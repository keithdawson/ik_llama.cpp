# CLAUDE.md — ik_llama.cpp numa-mirror fork

Guidance for working in this repo. It is also the standalone index for operating the
fork on the airgapped target server ("Pandora") — everything referenced here ships in
this source tree; nothing depends on external chat history or network access.

## What this fork is

Fork of ik_llama.cpp adding `--numa mirror`: one node-local copy of the model weights
(+ CPU KV cache) per NUMA node, with node-pinned threads, a NUMA-hierarchical barrier,
and per-token KV replication — so a dual-socket server runs at full local memory
bandwidth on both sockets. Built for a 2× EPYC 9665 (2 nodes, 12 DDR5 channels each,
2.25 TB) target serving large MoE models (e.g. GLM 5.2 Q5_K_XL ≈ 500 GB; the 2× mirror
= 1 TB, which fits).

## Document map (read these, in this order, on the server)

| File | Contents |
|---|---|
| [README.md](README.md) top section | `--numa mirror` usage, hybrid GPU flags, **complete env-knob table**, auto waiting-policy behavior |
| [docs/numa-tuning.md](docs/numa-tuning.md) | **The server playbook**: waiting policy + spin-count tuning, `pandora-tune.sh` run order and where each answer goes, ~500 GB model sizing notes, placement verification (`numastat`/`numa_maps`), cross-socket traffic measurement, cgroup recipes for co-existing with vLLM |
| [docs/numa-testbed.md](docs/numa-testbed.md) | The local fake-NUMA Docker testbed + **measured experiment verdicts** (what was validated, rejected, or deferred to the real machine, with numbers) |

## Scripts

| Script | Runs where | Purpose |
|---|---|---|
| `scripts/pandora-tune.sh` | **on the server** | one subcommand per knob; sweeps and prints the value + where to apply it. Start with `census`. Plain bash, no network/python needed |
| `scripts/tune-spincount.sh` | on the server | `GOMP_SPINCOUNT` sweep for mirror+GPU hybrid runs |
| `scripts/numa-ab.py` | on the server | original A/B harness (cgroups, perf DF counters, numastat sampling) |
| `scripts/build-zen.sh` | either | CPU build with the AVX-512 flag set for Zen 4/5 |
| `scripts/offline-kit.sh` | dev box | builds the airgapped Rocky 9 RPM bundle |
| `scripts/testbed/run-testbed.ps1` | dev box (Windows) | fake-NUMA Docker testbed orchestrator (build-image/build/smoke/ab/ab-branches) |
| `scripts/testbed/testbed-ab.py` | testbed container | A/B runner + smoke gate (byte-identity + counter sanity) |

## Validated defaults (already in the code — no action needed)

- **Waiting policy auto-set**: mirror + `-ngl>0` defaults `OMP_WAIT_POLICY=PASSIVE` +
  `GOMP_SPINCOUNT=5000` unless either is set by the user (+16% tg, pp-neutral on the
  testbed). Tune the exact spin count per machine with `tune-spincount.sh`.
- **Hierarchical barrier gate** at `n_batch > 32` (`GGML_NUMA_HIER_BATCH_MAX` to sweep).
- **CPU-count-weighted thread split**: nodes with fewer pinnable CPUs (e.g. after
  `GGML_NUMA_RESERVE_CPUS`) automatically get proportionally fewer threads; identical
  to the historical even split when nodes are symmetric.
- 7 upstream fixes cherry-picked (gemma-4 `-ngl 0` silent failure, CUDA MUL/contiguity,
  CUDA discrepancies) — see `git log --grep=cherry` / commits around `ba68f029`.

## Default-off knobs awaiting on-target validation

`GGML_NUMA_NT_COPY`, `GGML_NUMA_HIER_BATCH_MAX` (≠32), `GGML_NUMA_PIN=cpu`,
`GGML_NUMA_RESERVE_CPUS`, `GGML_NUMA_COPY_THREADS` (≠16), `GGML_NUMA_HUGETLB`,
`--numa-bind-compute`. Each has a `pandora-tune.sh` subcommand; the testbed verdicts
(docs/numa-testbed.md table) say what to expect.

Expert sharding — only for MoE models too large to mirror, and **not a speedup over
mirroring**: `--numa-mirror dense,kv` (one copy of the routed experts, pinned per node)
plus `GGML_NUMA_SHARD_STEAL` / `GGML_NUMA_SHARD_STEAL_COST` (rebalances lopsided routing;
bit-identical). Measure the remote-read penalty `r` first with
`pandora-tune.sh shard` (kept out of `all`); background in
[docs/numa-tuning.md](docs/numa-tuning.md#expert-sharding-and-rebalancing---numa-mirror-dense).
`GGML_NUMA_SHARD_SCHED=0` exists only for that measurement.

## Server quickstart (airgapped)

```sh
# 0. first session only: clear any build directory that rode along in the package
#    (Windows .exe artifacts are useless here). Exec bits on our scripts are committed
#    as 100755 now, so chmod is only needed if the transfer medium stripped them
#    (a plain .zip of the working folder does; the tar.gz carry archive does not).
rm -rf build
chmod +x scripts/*.sh scripts/testbed/*.sh 2>/dev/null || true   # harmless belt-and-braces

# 1. build (CPU; add -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89;120" for the Ada/Blackwells)
./scripts/build-zen.sh

# 2. sanity: mirroring must actually engage, fallbacks must be 0
./scripts/pandora-tune.sh census -m /models/model.gguf -t 188

# 3. tune (see docs/numa-tuning.md for the full ordered table)
./scripts/pandora-tune.sh threads      -m ...
./scripts/pandora-tune.sh barrier-gate -m ...
./scripts/pandora-tune.sh spincount    -m ... -g <gpu node> -- -ngl 99 -ot exps=CPU -fa on

# 4. apply the printed values (env vars in the serving unit / flags on the CLI), then
./scripts/pandora-tune.sh verify -m ...
```

## Gotchas (each of these cost real debugging time)

- **The mirror free-RAM check fails silently**: if `MemAvailable < weights + 2 GB` at
  mirror time, weights are NOT mirrored and the run continues degraded. Always confirm
  the `duplicated N weight tensors` log line (the `census` subcommand checks this).
  Hugetlb pools and co-resident vLLM both shrink `MemAvailable`.
- **Never run mirror under `numactl --cpunodebind`** (ggml re-pins past it; use
  `systemd-run --scope -p AllowedCPUs=...`), and never set `--cpuset-mems` /
  `AllowedMemoryNodes` (mirrors must allocate on every node).
- This fork's **`-fa` takes a value**: `-fa on`, not bare `-fa`.
- Instruct MoE models can produce degenerate text on raw untemplated prompts that
  *looks* like a backend bug — always sanity-check generation with the model's chat
  template before blaming the compute path.
- CUDA builds need the driver visible at **link** time for the VMM symbols (`--gpus all`
  when building in a container).
- Testbed only: WSL2 SMT siblings are adjacent pairs, so 8 physical cores =
  `--cpuset-cpus 0,2,4,6,8,10,12,14`; Git Bash needs `MSYS_NO_PATHCONV=1` for docker
  path args; keep `*.sh` LF (enforced via .gitattributes).
- **Packaging for the sneakernet**: build the carry archive with
  `git archive --format=tar.gz -o pandora-numa-kit-$(git rev-parse --short HEAD).tar.gz HEAD`
  — a plain folder copy drags along git-ignored junk (notably `build/` full of Windows
  `.exe` files, which cost a confused session on the server). Binaries are never
  carried; the server builds from source (quickstart step 1).

## Branch layout

- `numa-mirror` — stable branch, deploy from here.
- `numa-testbed` — where testbed + tuning work lands first; merged into `numa-mirror`
  once validated.
- `kv-unified` — experimental elastic slot context sharing for the server.

`SERVER CONFIGURATION.txt` (repo root, git-excluded via `.git/info/exclude`) holds the
target hardware specs — never commit it.
