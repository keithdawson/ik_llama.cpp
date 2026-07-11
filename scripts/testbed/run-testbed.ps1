# Host orchestrator for the fake-NUMA testbed (docs/numa-testbed.md).
#
# Runs everything inside the ik-numa-testbed container (docker/numa-testbed.Containerfile).
# Source is bind-mounted at /src; build output goes to the ik-build named volume (object
# files over a Windows bind mount are gRPC-FUSE slow); models to ik-models; ccache to
# ik-ccache. Benchmark runs are confined to 8 physical cores via --cpuset-cpus (computed
# once from lscpu inside the image, cached in testbed-results/cpuset.cache); builds and
# downloads run unconfined.
#
# Usage:
#   .\scripts\testbed\run-testbed.ps1 build-image
#   .\scripts\testbed\run-testbed.ps1 build                         # compile /src -> /build/main
#   .\scripts\testbed\run-testbed.ps1 download-model                # gemma MoE + smoke model
#   .\scripts\testbed\run-testbed.ps1 smoke                         # functional gate
#   .\scripts\testbed\run-testbed.ps1 ab -Config scripts/testbed/configs/baseline.json
#   .\scripts\testbed\run-testbed.ps1 ab-branches -RefA main -RefB my-experiment -Config <cfg>
#   .\scripts\testbed\run-testbed.ps1 shell                         # interactive, pinned cores
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("build-image", "build", "download-model", "smoke", "ab", "ab-branches", "shell", "detect-cpus")]
    [string]$Command,

    [string]$Config,
    [string]$RefA,
    [string]$RefB,
    [string]$Image = "ik-numa-testbed",
    [int]$FakeNodes = 2,
    [string]$XgmiGbps = "60",
    [string]$SmokeModel = "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    [int]$Cores = 8
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$CacheFile = Join-Path $RepoRoot "testbed-results\cpuset.cache"

$CmakeConfigure = "cmake -B {0} -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON " +
    "-DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON " +
    "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"

function Get-CpuSet {
    if (Test-Path $CacheFile) {
        return (Get-Content $CacheFile -Raw).Trim()
    }
    Write-Host "detecting physical cores via lscpu inside $Image ..."
    $lines = docker run --rm $Image lscpu '-p=CPU,CORE'
    if ($LASTEXITCODE -ne 0) { throw "lscpu probe failed - run 'build-image' first" }
    $coreMap = [ordered]@{}
    foreach ($line in $lines) {
        if ($line -match '^\s*#') { continue }
        $cpu, $core = $line.Split(',')
        if (-not $coreMap.Contains($core)) { $coreMap[$core] = $cpu } # first CPU of each core = physical
    }
    $picked = @($coreMap.Values | Select-Object -First $Cores)
    if ($picked.Count -lt $Cores) {
        Write-Warning "only $($picked.Count) physical cores visible; using all of them"
    }
    $cpuset = $picked -join ","
    New-Item -ItemType Directory -Force -Path (Split-Path $CacheFile) | Out-Null
    Set-Content -Path $CacheFile -Value $cpuset
    Write-Host "benchmark cpuset: $cpuset (cached in $CacheFile)"
    return $cpuset
}

function Invoke-Testbed {
    param(
        [string[]]$Cmd,          # command to exec in the container
        [switch]$Pinned,         # confine to the benchmark cpuset
        [switch]$Interactive,
        [string[]]$ExtraMounts = @()
    )
    $dockerArgs = @("run", "--rm")
    if ($Interactive) { $dockerArgs += "-it" }
    if ($Pinned) { $dockerArgs += @("--cpuset-cpus", (Get-CpuSet)) }
    # NOTE: never pass --memory here - llama's mirror free-RAM check reads /proc/meminfo,
    # which shows the whole WSL2 VM; a cgroup cap it can't see would OOM-kill silently.
    $dockerArgs += @(
        "-v", "${RepoRoot}:/src",
        "-v", "ik-build:/build",
        "-v", "ik-models:/models",
        "-v", "ik-ccache:/ccache",
        "-w", "/src",
        "-e", "GGML_NUMA_FAKE=$FakeNodes",
        "-e", "GGML_NUMA_XGMI_GBPS=$XgmiGbps"
    )
    $dockerArgs += $ExtraMounts
    $dockerArgs += $Image
    $dockerArgs += $Cmd
    docker @dockerArgs
    if ($LASTEXITCODE -ne 0) { throw "container command failed ($LASTEXITCODE): $($Cmd -join ' ')" }
}

function Build-Tree {
    param([string]$Src, [string]$BuildDir)
    Invoke-Testbed -Cmd @("bash", "-lc", ("cd {0} && {1} && cmake --build {2} --config Release -j`$(nproc)" -f `
        $Src, ($CmakeConfigure -f $BuildDir), $BuildDir))
}

function Get-WorktreePath([string]$Ref) {
    $safe = $Ref -replace '[^A-Za-z0-9._-]', '_'
    return Join-Path (Split-Path -Parent $RepoRoot) "ik-wt-$safe"
}

switch ($Command) {
    "build-image" {
        docker build -f (Join-Path $RepoRoot "docker\numa-testbed.Containerfile") -t $Image $RepoRoot
        if ($LASTEXITCODE -ne 0) { throw "image build failed" }
    }
    "detect-cpus" {
        if (Test-Path $CacheFile) { Remove-Item $CacheFile }
        Get-CpuSet | Out-Host
    }
    "build" {
        Build-Tree -Src "/src" -BuildDir "/build/main"
    }
    "download-model" {
        Invoke-Testbed -Cmd @("bash", "/src/scripts/testbed/download-model.sh")
    }
    "smoke" {
        Invoke-Testbed -Pinned -Cmd @("python3", "/src/scripts/testbed/testbed-ab.py", "smoke",
            "--model", $SmokeModel, "--bin", "/build/main/bin/llama-cli")
    }
    "ab" {
        if (-not $Config) { throw "ab requires -Config <path to json>" }
        $cfgInContainer = "/src/" + ((Resolve-Path $Config).Path.Substring($RepoRoot.Length + 1) -replace '\\', '/')
        Invoke-Testbed -Pinned -Cmd @("python3", "/src/scripts/testbed/testbed-ab.py", "run", "--config", $cfgInContainer)
    }
    "ab-branches" {
        if (-not ($RefA -and $RefB -and $Config)) { throw "ab-branches requires -RefA, -RefB and -Config" }
        # host-side worktrees so both revisions are plain directories the container can mount.
        # convention: the config's variant binaries point at /build/a/bin/... and /build/b/bin/...
        $wtA = Get-WorktreePath $RefA
        $wtB = Get-WorktreePath $RefB
        foreach ($pair in @(@($wtA, $RefA), @($wtB, $RefB))) {
            $path, $ref = $pair
            if (-not (Test-Path $path)) {
                # --detach so a ref that is already checked out elsewhere (e.g. the main
                # worktree's branch) can still be materialized
                git -C $RepoRoot worktree add --detach $path $ref
                if ($LASTEXITCODE -ne 0) { throw "git worktree add $path $ref failed" }
            } else {
                git -C $path checkout --detach $ref
                if ($LASTEXITCODE -ne 0) { throw "git checkout $ref in $path failed" }
            }
        }
        $mounts = @("-v", "${wtA}:/src-a", "-v", "${wtB}:/src-b")
        Invoke-Testbed -ExtraMounts $mounts -Cmd @("bash", "-lc", ("cd /src-a && {0} && cmake --build /build/a -j`$(nproc)" -f ($CmakeConfigure -f "/build/a")))
        Invoke-Testbed -ExtraMounts $mounts -Cmd @("bash", "-lc", ("cd /src-b && {0} && cmake --build /build/b -j`$(nproc)" -f ($CmakeConfigure -f "/build/b")))
        $cfgInContainer = "/src/" + ((Resolve-Path $Config).Path.Substring($RepoRoot.Length + 1) -replace '\\', '/')
        Invoke-Testbed -Pinned -ExtraMounts $mounts -Cmd @("python3", "/src/scripts/testbed/testbed-ab.py", "run", "--config", $cfgInContainer)
    }
    "shell" {
        Invoke-Testbed -Pinned -Interactive -Cmd @("bash")
    }
}
