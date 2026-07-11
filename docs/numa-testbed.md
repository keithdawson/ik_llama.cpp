# Fake-NUMA testbed (local A/B rig for `--numa mirror`)

A Docker-Desktop testbed that exercises the whole NUMA-mirror stack on a single-node
desktop (Ryzen 9800X3D, 8C/16T, 48 GB) so mirror-path changes can be A/B-tested before
they ride to the real dual-EPYC target. Companion tooling lives in `scripts/testbed/`;
the real-hardware harness remains `scripts/numa-ab.py` + `docs/numa-tuning.md`.

## How it works

`GGML_NUMA_FAKE=N` makes `ggml_numa_init` fabricate an N-node topology by splitting the
CPUs the process is allowed on (the container's `--cpuset-cpus` slice) into N contiguous
blocks. Everything downstream keys off the node table, so the **real** mirror machinery
runs: per-node weight/KV copies, node-pinned threads, the hierarchical barrier, per-token
KV replication, resyncs. Only `mbind` is skipped (all pages genuinely live on one node).

Two proxies stand in for the physics a desktop cannot reproduce:

- `GGML_NUMA_XGMI_GBPS=<GB/s>` paces the **explicit cross-node copy paths** (mirror
  population, resync, per-token KV replication) to a modeled interconnect bandwidth.
  Default in the runner: 60 (ballpark for one xGMI direction on EPYC 9005; sweep it).
- `GGML_NUMA_STATS=1|2` counts what crossed the "interconnect": per-node local-vs-fallback
  pointer resolutions, replicated/populated bytes, hier/flat barrier passes, throttle
  sleep time. `1` dumps `numa_stats: key=value` lines to stderr at exit; `2` also dumps
  and resets after every `llama_print_timings` (per-phase attribution).

### What the testbed does and does not model

| Effect | Modeled? |
|---|---|
| Mirror code paths (allocation, population, redirection, replication, pinning, barrier) | yes, bit-for-bit the real code |
| Explicit cross-node copy cost (population, KV replicate, resync) | yes, via the GBPS throttle |
| Remote-DRAM latency on *implicit* reads (a thread reading another node's pages) | **no** — all memory is equally fast; the stats counters flag such accesses instead |
| Separate L3 per node | **no** — the 9800X3D is one CCD; both "nodes" share one 96 MB L3 |
| Per-node memory bandwidth scaling | **no** — both "nodes" share one DDR5 controller |

Consequences for reading results:
- **Wall-clock-valid locally (W)**: pure CPU-cost changes (resolver overhead, copy
  parallelism/streaming, hugepages) and anything on the throttled copy paths.
- **Counter-proxy-only (C)**: locality/contention changes (barrier gating, node-aware
  scheduling, placement). Locally you prove correctness + "remote bytes avoided";
  the t/s verdict needs the dual-socket machine.
- A local *regression* in W metrics is disqualifying; a local *no-change* in C metrics
  is expected and fine.

## One-time setup

1. **Raise the WSL2 memory cap** (2x-mirroring a ~15 GB model + KV needs ~35 GB):
   edit `C:\Users\<you>\.wslconfig`:

   ```ini
   [wsl2]
   memory=40GB
   processors=16
   ```

   then `wsl --shutdown` and restart Docker Desktop.

2. Build the image and the tree, fetch models:

   ```powershell
   .\scripts\testbed\run-testbed.ps1 build-image
   .\scripts\testbed\run-testbed.ps1 build            # cmake -> ik-build volume /build/main
   .\scripts\testbed\run-testbed.ps1 download-model   # gemma MoE + Qwen2.5-0.5B smoke model
   ```

   Build output lives in the `ik-build` named volume (`/build/...`), models in
   `ik-models` (`/models`), compiler cache in `ik-ccache` — never on the Windows bind
   mount, which is slow for many small files.

3. Gate every change with the smoke suite (inert-when-off, activation, greedy-identity,
   counter sanity):

   ```powershell
   .\scripts\testbed\run-testbed.ps1 smoke
   ```

## Running A/B experiments

```powershell
# variants within one binary (flags/env):
.\scripts\testbed\run-testbed.ps1 ab -Config scripts\testbed\configs\baseline.json

# two git revisions (worktrees ..\ik-wt-<ref>, built to /build/a and /build/b;
# the config's variant "binary" fields must point at /build/a/bin/... and /build/b/bin/...):
.\scripts\testbed\run-testbed.ps1 ab-branches -RefA numa-mirror -RefB my-experiment -Config <cfg>
```

The runner (`scripts/testbed/testbed-ab.py`) does per-variant warmup, then **alternating
A/B/A/B timing reps** (stats off), then one instrumented rep per variant, and writes
JSON + a markdown table to `testbed-results/<timestamp>-<name>/`. Re-render anytime with
`python3 scripts/testbed/testbed-ab.py report <dir>`.

Benchmark runs are confined to the first 8 physical cores (`--cpuset-cpus`, autodetected
via `lscpu -p=CPU,CORE` and cached in `testbed-results/cpuset.cache`; delete the file or
run `detect-cpus` to re-probe). Do **not** add `--memory` to the container: the mirror
free-RAM check reads `/proc/meminfo` (the whole WSL2 VM) and cannot see cgroup limits.

## Environment variables

| Var | Values | Meaning |
|---|---|---|
| `GGML_NUMA_FAKE` | 2..8 | fabricate N NUMA nodes from the allowed CPUs (testbed only) |
| `GGML_NUMA_XGMI_GBPS` | float GB/s | cap explicit cross-node copies (0/unset = off) |
| `GGML_NUMA_STATS` | 0/1/2 | mirror-path counters: off / dump at exit / also per-timings |
| `GGML_NUMA_PIN` | `cpu` | per-CPU thread pinning instead of per-node (see numa-tuning.md) |

All are inert when unset; a build with these changes and no env vars behaves identically
to one without them.

## GPU-hybrid mode (Pandora shape: dense on GPU, experts on CPU)

Build the CUDA image and tree (needs `--gpus all` even for the build, so `libcuda` links):

```powershell
docker build -f docker/numa-testbed.Containerfile --build-arg BASE=nvidia/cuda:12.8.0-devel-ubuntu24.04 -t ik-numa-testbed-cuda .
docker run --rm --gpus all -v ${PWD}:/src -v ik-build:/build -v ik-ccache:/ccache ik-numa-testbed-cuda bash -lc "cmake -B /build/cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 <zen flags> && cmake --build /build/cuda -j16"
```

Canonical hybrid flags: `-ngl 99 -ot exps=CPU -fa on --numa mirror --numa-gpu-node 1`
(this fork's `-fa` requires a value). With KV on the GPU, `kv_repl_bytes` must be 0 and
only expert tensors get mirrored — both verified. Config: `configs/gpu-hybrid.json`
(currently the Qwen1.5-MoE model: **gemma-4 produces garbage on any CUDA build of this
fork** — CPU-only builds are fine; needs upstream gemma-4 fixes cherry-picked).

**Hybrid rule (auto-set by `llama_init_from_gpt_params` when mirror + `-ngl` > 0):
`OMP_WAIT_POLICY=PASSIVE` + `GOMP_SPINCOUNT=25000`.** Setting either env var yourself
disables the auto-default. The history, measured on gemma (testbed, 8 cores):

| waiting policy (mirror hybrid) | pp vs no-numa | tg vs no-numa |
|---|---|---|
| default (long spin) | -12% | -5% (driver starved by spinning pinned workers) |
| fully PASSIVE (0 spins) | -7…-10% | +4…+8% (every CPU segment pays thread-wake latency) |
| **PASSIVE + spin 25k (~100 µs)** | **+0.2%** | **+15.9%** (27.9 t/s) |
| spin 250k | -1.4% | -4% (back toward starvation) |

Controls that pin down the mechanism: PASSIVE *without* mirror collapses TG -36%
(unpinned sleeping threads wake on random cores) — the win is specifically
pinned + briefly-spin-then-yield. `GGML_NUMA_RESERVE_CPUS=N[@node]` (drops N CPUs
from a node's pinning set; the thread block split is CPU-count weighted so the node
gets proportionally fewer threads) did **not** help locally once waiting was fixed —
B/C/D within noise on TG, reserve costs PP compute — but on a 96-core socket the
2-core insurance is nearly free; re-test on the real machine.

Per-node resolve counters show a ~20-25% node0-biased count skew during hybrid PP —
that is low-parallelism ops (nth < n_threads) always landing on the lowest thread ids
(= node 0), not a data-placement bug (fallbacks stay 0).

## Experiment log convention

One directory per experiment under `testbed-results/`, plus a row appended to
`testbed-results/LOG.md`: date, experiment id (from the backlog in the project plan),
config file, W/C validity class, tg/pp delta, counter deltas, verdict
(roll up / park for Pandora / reject). Rolled-up winners must keep the smoke suite green
and are re-based onto the cumulative branch so later experiments measure against the
current best.

## Sanity checklist after any mirror-path change

1. `smoke` passes (includes byte-identical greedy output vs the no-NUMA baseline).
2. Stats rep: `resolve_hit` split ≈ evenly across nodes, `resolve_fallback == 0`,
   `kv_repl_bytes ≈ n_tokens × Σ_layers(K-row + V-row bytes) × (nodes−1)`.
3. Throttle monotonicity: tg t/s must not *rise* when `GGML_NUMA_XGMI_GBPS` drops
   (60 → 8 → 1) unless the change specifically removes cross-node traffic — in which
   case `kv_repl_bytes`/`populate_bytes` must show it.
