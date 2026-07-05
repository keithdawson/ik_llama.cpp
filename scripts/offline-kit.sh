#!/usr/bin/env bash
# offline-kit.sh - build an offline RPM bundle for an airgapped Rocky Linux 9 server.
#
# Runs dependency resolution inside a fresh Rocky 9 container (so the closure matches
# a minimal server, not this workstation), downloads every RPM needed to build and run
# ik_llama.cpp plus optional CUDA/driver/docker components, creates a local dnf repo,
# and packs it together with a generated install.sh into one tarball to carry over.
#
# On the airgapped server:   tar xzf offline-kit-*.tar.gz && cd offline-kit && sudo ./install.sh
#
# Usage:
#   ./scripts/offline-kit.sh make [options]
#
# Options:
#   --out DIR       output directory            (default: ./offline-kit)
#   --cuda VER      CUDA toolkit version, e.g. 12.9, or 'none'   (default: 12.9)
#   --with-driver   include NVIDIA open-kernel-module driver RPMs (+ dkms, kernel-devel)
#   --with-docker   include docker-ce engine RPMs
#   --with-nvctk    include nvidia-container-toolkit RPMs (GPU access inside containers)
#   --dry-run       resolve and print download URLs only; nothing is downloaded
#   --image IMG     resolver container image     (default: rockylinux/rockylinux:9)
#
# Requires: docker (or podman) with internet access. Works from Linux, WSL, or Git Bash.
set -euo pipefail

OUT=./offline-kit
CUDA_VER=12.9
WITH_DRIVER=0
WITH_DOCKER=0
WITH_NVCTK=0
DRY=0
IMAGE=rockylinux/rockylinux:9

cmd=${1:-}; shift || true
if [ "$cmd" != "make" ]; then
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
while [ $# -gt 0 ]; do
    case "$1" in
        --out)         OUT=$2; shift 2 ;;
        --cuda)        CUDA_VER=$2; shift 2 ;;
        --with-driver) WITH_DRIVER=1; shift ;;
        --with-docker) WITH_DOCKER=1; shift ;;
        --with-nvctk)  WITH_NVCTK=1; shift ;;
        --dry-run)     DRY=1; shift ;;
        --image)       IMAGE=$2; shift 2 ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

RUNTIME=$(command -v docker || command -v podman) || { echo "need docker or podman"; exit 1; }

mkdir -p "$OUT/rpms"
OUT_ABS=$(cd "$OUT" && pwd)

# ---------------------------------------------------------------- package manifest
# Base: everything needed to build ik_llama.cpp (gcc 11 system + gcc-toolset-13, cmake,
# OpenMP) and operate it (numactl/numastat, perf + friends for docs/numa-tuning.md,
# python3 for scripts/numa-ab.py), plus debugging & quality-of-life tools.
cat > "$OUT_ABS/packages-base.txt" <<'EOF'
gcc
gcc-c++
make
cmake
git
binutils
glibc-devel
libstdc++-devel
libgomp
gcc-toolset-13
gcc-toolset-13-gcc-c++
numactl
numactl-libs
numactl-devel
perf
sysstat
kernel-tools
pciutils
dmidecode
python3
python3-pip
jq
tmux
rsync
tar
unzip
zip
wget
gdb
strace
patch
which
vim-enhanced
htop
ccache
epel-release
EOF

EXTRA_PKGS=""
if [ "$CUDA_VER" != "none" ]; then
    EXTRA_PKGS+=" cuda-toolkit-${CUDA_VER//./-}"
    echo "$CUDA_VER" > "$OUT_ABS/HAS_CUDA"
fi
if [ "$WITH_DRIVER" = 1 ]; then
    # open kernel modules (required for Blackwell); dkms builds need kernel-devel matching
    # the RUNNING kernel. The bundled kernel-devel tracks the container's (current) Rocky
    # minor, which may be newer than the target's - so bundle the matching kernel too, and
    # install.sh offers to update+reboot when versions mismatch.
    EXTRA_PKGS+=" nvidia-open dkms kernel kernel-devel kernel-headers"
    touch "$OUT_ABS/HAS_DRIVER"
fi
if [ "$WITH_DOCKER" = 1 ]; then
    EXTRA_PKGS+=" docker-ce docker-ce-cli containerd.io docker-compose-plugin"
    touch "$OUT_ABS/HAS_DOCKER"
fi
if [ "$WITH_NVCTK" = 1 ]; then
    EXTRA_PKGS+=" nvidia-container-toolkit"
    touch "$OUT_ABS/HAS_NVCTK"
fi

# ---------------------------------------------------------------- resolver container
DL_FLAGS="--resolve --alldeps"
[ "$DRY" = 1 ] && DL_FLAGS="$DL_FLAGS --urls"

CONTAINER_SCRIPT=$(cat <<EOS
set -euo pipefail
dnf -y -q install dnf-plugins-core createrepo_c epel-release >/dev/null
if [ -n "${EXTRA_PKGS// }" ]; then
    if [ "$CUDA_VER" != "none" ] || [ "$WITH_DRIVER" = 1 ]; then
        dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
    fi
    if [ "$WITH_DOCKER" = 1 ]; then
        dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
    fi
    if [ "$WITH_NVCTK" = 1 ]; then
        curl -sL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
            -o /etc/yum.repos.d/nvidia-container-toolkit.repo
        # metadata signature check would prompt for a key import; we verify nothing here
        # (the airgapped installer runs --nogpgcheck anyway)
        sed -i 's/^repo_gpgcheck=1/repo_gpgcheck=0/; s/^gpgcheck=1/gpgcheck=0/' \
            /etc/yum.repos.d/nvidia-container-toolkit.repo
    fi
fi
echo "== resolving: base packages +${EXTRA_PKGS:-}"
dnf -y download $DL_FLAGS --destdir /out/rpms \$(grep -v '^#' /out/packages-base.txt) $EXTRA_PKGS
if [ "$DRY" != 1 ]; then
    cp -r /etc/pki/rpm-gpg /out/gpg-keys 2>/dev/null || true
    createrepo_c --general-compress-type=gz /out/rpms
    echo "== repo created: \$(ls /out/rpms/*.rpm | wc -l) RPMs, \$(du -sh /out/rpms | cut -f1)"
fi
EOS
)

# Git Bash / MSYS: give docker a Windows-style host path and disable path mangling
HOST_OUT="$OUT_ABS"
if command -v cygpath >/dev/null 2>&1; then
    HOST_OUT=$(cygpath -w "$OUT_ABS")
    export MSYS_NO_PATHCONV=1
fi

"$RUNTIME" run --rm -v "$HOST_OUT:/out" "$IMAGE" bash -c "$CONTAINER_SCRIPT"

if [ "$DRY" = 1 ]; then
    echo "== dry run complete (no files downloaded)"
    exit 0
fi

# ---------------------------------------------------------------- generated installer
cat > "$OUT_ABS/install.sh" <<'EOF'
#!/usr/bin/env bash
# Installs the offline kit on an airgapped Rocky Linux 9 server. Run as root.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run as root (sudo ./install.sh)"; exit 1; }
DIR=$(cd "$(dirname "$0")" && pwd)

REPOFLAGS=(--disablerepo='*' --repofrompath="offlinekit,file://$DIR/rpms"
           --enablerepo=offlinekit --setopt=offlinekit.gpgcheck=0 --nogpgcheck)

echo "== installing base packages"
# shellcheck disable=SC2046
dnf "${REPOFLAGS[@]}" -y install $(grep -v '^#' "$DIR/packages-base.txt")

if [ -f "$DIR/HAS_CUDA" ]; then
    V=$(cat "$DIR/HAS_CUDA")
    echo "== installing CUDA toolkit $V"
    dnf "${REPOFLAGS[@]}" -y install "cuda-toolkit-${V//./-}"
    cat > /etc/profile.d/cuda.sh <<'EOP'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
EOP
    echo "   (new shells get nvcc on PATH via /etc/profile.d/cuda.sh)"
fi

if [ -f "$DIR/HAS_DRIVER" ]; then
    RUNNING=$(uname -r)
    if ! ls "$DIR"/rpms/kernel-devel-"$RUNNING"* >/dev/null 2>&1 && \
       ! rpm -q "kernel-devel-$RUNNING" >/dev/null 2>&1; then
        echo "NOTE: bundled kernel-devel does not match the running kernel ($RUNNING)."
        echo "      Installing the bundled kernel + kernel-devel; REBOOT into the new kernel"
        echo "      before expecting the dkms nvidia module to load."
        dnf "${REPOFLAGS[@]}" -y install kernel
    fi
    echo "== installing NVIDIA open-kernel-module driver (dkms)"
    dnf "${REPOFLAGS[@]}" -y install nvidia-open dkms kernel-devel kernel-headers
    echo "   reboot after install; verify with nvidia-smi"
fi

if [ -f "$DIR/HAS_DOCKER" ]; then
    echo "== installing docker-ce"
    dnf "${REPOFLAGS[@]}" -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable --now docker || true
fi

if [ -f "$DIR/HAS_NVCTK" ]; then
    echo "== installing nvidia-container-toolkit"
    dnf "${REPOFLAGS[@]}" -y install nvidia-container-toolkit
    command -v nvidia-ctk >/dev/null && nvidia-ctk runtime configure --runtime=docker || true
fi

echo
echo "== post-install checks"
for c in "gcc --version" "g++ --version" "cmake --version" "numastat -V" "perf --version" "python3 --version"; do
    printf '  %-18s ' "${c%% *}:"; $c 2>/dev/null | head -1 || echo MISSING
done
[ -f "$DIR/HAS_CUDA" ] && { printf '  %-18s ' nvcc:; /usr/local/cuda/bin/nvcc --version 2>/dev/null | grep release || echo "MISSING (check /usr/local/cuda)"; }
echo
echo "Done. Build with:"
echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89"
echo "  cmake --build build -j \$(nproc)"
echo "  (gcc-toolset-13 available via: scl enable gcc-toolset-13 bash)"
EOF
chmod +x "$OUT_ABS/install.sh"

cat > "$OUT_ABS/README.md" <<EOF
# Offline dependency kit for Rocky Linux 9 (built $(date -u +%Y-%m-%d))

Self-contained dnf repo with everything needed to build and run ik_llama.cpp
(numa-mirror fork) on an airgapped Rocky 9 server.

Components: base toolchain + tooling$([ "$CUDA_VER" != none ] && echo ", CUDA toolkit $CUDA_VER")$([ "$WITH_DRIVER" = 1 ] && echo ", NVIDIA open driver (dkms)")$([ "$WITH_DOCKER" = 1 ] && echo ", docker-ce")$([ "$WITH_NVCTK" = 1 ] && echo ", nvidia-container-toolkit")

## Install on the server

    tar xzf offline-kit-*.tar.gz
    cd offline-kit
    sudo ./install.sh

The installer only reads from the bundled repo (network is never touched).
Package list: packages-base.txt; full RPM inventory: MANIFEST.txt.

## Notes

- RPMs resolved against a minimal Rocky 9 container, so the closure is a superset
  of what a stock server needs. Already-installed packages are skipped by dnf.
- Some RPMs may be slightly newer than the server's 9.6 snapshot; dnf treats those
  as normal in-place updates.
- The NVIDIA driver (if included) uses the open kernel modules required by
  Blackwell GPUs and builds via dkms - kernel-devel must match the RUNNING kernel
  (install.sh checks and warns).
- AMD uProf is NOT included (EULA-gated download). Fetch the RPM manually from
  https://www.amd.com/en/developer/uprof.html and bring it alongside this kit.
EOF

(cd "$OUT_ABS/rpms" && ls *.rpm | sort) > "$OUT_ABS/MANIFEST.txt"

STAMP=$(date +%Y%m%d)
TARBALL="offline-kit-rocky9-$STAMP.tar.gz"
echo "== packing $TARBALL"
tar -C "$(dirname "$OUT_ABS")" -czf "$(dirname "$OUT_ABS")/$TARBALL" "$(basename "$OUT_ABS")"
echo "== done: $(dirname "$OUT_ABS")/$TARBALL ($(du -sh "$(dirname "$OUT_ABS")/$TARBALL" | cut -f1))"
echo "   RPMs: $(wc -l < "$OUT_ABS/MANIFEST.txt")  (inventory in MANIFEST.txt)"
