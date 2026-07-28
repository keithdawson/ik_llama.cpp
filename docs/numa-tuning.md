# NUMA tuning and measurement guide

How to verify memory placement, measure cross-socket traffic, and A/B the hybrid
NUMA-GPU knobs (`--numa-gpu-node`, `--numa-bind-compute`) on a multi-socket server.
Written for a dual-socket AMD EPYC running a RHEL-family distro, but everything except
the AMD-specific counters applies to Intel boxes too.

A scripted harness that automates most of this is in [`scripts/numa-ab.py`](../scripts/numa-ab.py).

## The knobs being tested

| Flag | What it places | Trade-off |
|---|---|---|
| `--numa mirror` | one copy of weights (+ KV if on CPU) per node | N× RAM for mirrored data |
| `--numa-gpu-node N` | ggml context arenas (tensor metadata) → node N | negligible; prerequisite for the next flag |
| `--numa-bind-compute` | scheduler CPU compute buffers (intermediate tensor data — what the GPU DMAs from/to) → node N | GPU transfers become node-local, but expert threads on *other* nodes write outputs remotely |
| `GGML_NUMA_PIN=cpu` (env) | pins each mirror thread to one *specific* CPU instead of "anywhere on its node" | no cross-CCD thread migration (each EPYC CCD has its own L3), physical cores before SMT siblings — but the scheduler loses the freedom to dodge other load on the socket |
| `--numa-mirror dense,kv` | routed experts get **one** copy, pinned per node, instead of a copy on every node | the only way to run an MoE too large to mirror; costs TG throughput to routing imbalance — see [Expert sharding and rebalancing](#expert-sharding-and-rebalancing---numa-mirror-dense) |
| `GGML_NUMA_SHARD_STEAL=1` (env) | rebalances each MoE op's experts across nodes when routing lands lopsided | recovers most of that imbalance and stays bit-identical, but needs `GGML_NUMA_SHARD_STEAL_COST` set from a measured `r` |

`--numa-bind-compute` and `GGML_NUMA_PIN=cpu` are the experiments: whether they help
depends on batch size, cache behavior, and what else runs on the box. Measure, don't
assume — `numa-ab.py matrix` (below) sweeps them for you.

Build note: make sure the build has **OpenMP** enabled (it is by default — verify with
`grep GGML_OPENMP build/CMakeCache.txt`, or `ldd build/bin/llama-server | grep gomp`).
Without OpenMP, worker threads are created and joined on *every* graph compute, which
is real per-token overhead at high thread counts.

## Waiting policy (mirror + GPU offload)

When `--numa mirror` runs with GPU layers, the pinned OpenMP workers must spin only
*briefly* and then sleep while the GPU works: spinning through GPU segments starves the
CUDA driver thread (measured -12% pp / -5% tg on the testbed), while fully passive
waiting makes every CPU expert segment pay thread-wake latency (~-10% pp). The binary
therefore **auto-sets `OMP_WAIT_POLICY=PASSIVE` + `GOMP_SPINCOUNT=5000`** whenever
mirror + `-ngl > 0` and *neither* variable is already in the environment (it logs one
line when it does). 5000 is the measured optimum on the dual-EPYC target; a desktop
Zen 5 preferred 25000 (more cores → shorter spin wins, since idle spinners cost more
when there are ~190 of them). Re-tune per machine:

```sh
./scripts/tune-spincount.sh -m /path/model.gguf -t <your -t> -- <your real serving flags>
```

The script sweeps `GOMP_SPINCOUNT` over {0, 1k, 2.5k, 5k, 7.5k, 10k, 25k, 100k} against
a ~3k-token prompt and prints a pp/tg table plus the best value (override with `SPINS=`).

**What to do with the answer:** pick the spin count with the best tg whose pp is also
within noise of the best pp row (they usually agree; if not, favor tg for a serving
box). If it's 5000, do nothing — the built-in default already matches. If it differs,
set both variables explicitly in the serving environment — an explicit setting disables
the auto-default:

```sh
# wrapper script / shell
export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=<best>

# systemd unit
Environment=OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=<best>

# docker compose
environment: { OMP_WAIT_POLICY: PASSIVE, GOMP_SPINCOUNT: "<best>" }
```

Re-tune if the thread count, model, or GPU/CPU split changes materially — the optimum
tracks how long the per-layer GPU segments are relative to thread wake latency.

## Expert sharding and rebalancing (`--numa-mirror dense`)

Only relevant for MoE models **too large to mirror**. If `2 × weights` fits in RAM, use
`--numa mirror` and skip this section — sharding is a way to fit, not a way to go faster.

### What sharding does

`--numa-mirror dense,kv` mirrors everything *except* the routed experts. Those keep a single
copy, with expert `e` pinned to node `e % n_nodes`, so the footprint becomes
`dense × n_nodes + experts × 1` instead of `everything × n_nodes`. The MoE kernels then compute
expert `e` only on that node's threads, so its weight reads stay node-local.

Shared experts (`*_shexp`) are **not** sharded — they are active on every token, so they stay
mirrored (and can sit on the GPU instead; `--cpu-moe` only matches `*_exps`). The boot log says
which: `NUMA shard: N shared-expert tensors (X GiB) mirrored, not sharded`.

### The cost you are buying: routing imbalance

GLM-5.2 routes each token to **8 of 256** experts. Under `e % 2` those 8 split by parity —
sometimes 4/4, often 5/3, sometimes 6/2. Every MoE op ends at a barrier, so **the op runs at
the pace of the busier node** and the lighter one idles.

That is what `GGML_NUMA_STATS=1` reports as `moe_node_skew_mean` (`1.00` = perfect split,
`2.00` = one node did everything). Measured on the testbed: **TG ≈ 1.25–1.32**, PP ≈ 1.07 —
prompt processing averages out because every expert gets rows, token generation does not.

> Do **not** read `moe_experts=` (the cumulative per-node totals) as the balance figure. It
> averages out to ~1.01 over a run and looks perfect even when every individual op was lopsided.
> `moe_node_skew_mean` is the number that matters.

The skew counter works with sharding **off**, so a plain `--numa mirror` run predicts what
sharding would cost on your model before you commit to it.

### Rebalancing it away: `GGML_NUMA_SHARD_STEAL=1` (default off)

The idle node can take work off the busy one instead of waiting. The obvious way — let a node
finish and then grab leftover work — fits badly here: a node's threads all cooperate on *one*
expert, splitting its rows, so "node B takes expert X" needs all ~64 of B's threads to agree
mid-flight. That needs a synchronization point (the cost we are trying to remove), and the row
split would depend on which threads arrived first, so output would drift run to run.

Instead the assignment is decided **up front**. How much work each expert represents is already
known before any of it starts — `matrix_row_counts[e]`, the number of token-rows routed to
expert `e`, is filled by thread 0 and published by the barrier that already precedes the expert
loop. So every thread runs the same greedy pass over the same table and reaches the same
assignment, with no atomics, no extra barrier, and no coordination.

> Two workers splitting a pile of tasks can grab them as they free up — which needs constant
> checking so they don't collide — or both read the whole list at the start and independently
> work out the same split. Same rule, same list, same plan, no talking.

Each expert is still computed start to finish by one node; only *which* node can change. The
arithmetic is untouched, so **output stays bit-identical** and the smoke gate's byte-identity
check still applies.

### Setting `GGML_NUMA_SHARD_STEAL_COST` (`r`)

Moving expert `X` off its owner makes its weight reads cross-socket. `r` is how much slower that
is (`1.0` = free, `2.0` = twice the cost). Worked through a 5/3 split, one unit per expert:

| | busy node | idle node | op takes |
|---|---|---|---|
| do nothing | 5 | 3 | **5** |
| move one, `r = 1.0` | 4 | 3 + 1.0 = 4 | **4** ✅ |
| move one, `r = 1.5` | 4 | 3 + 1.5 = 4.5 | **4.5** ✅ |
| move one, `r = 2.0` | 4 | 3 + 2.0 = 5 | **5** — no gain |
| move one, `r = 3.0` | 4 | 3 + 3.0 = 6 | **6** ❌ worse |

The greedy pass runs exactly this comparison and moves only when the predicted time strictly
drops. So the algorithm is **self-limiting**: if remote reads are expensive on your machine it
declines to move at all rather than making things worse. Setting `r` too *high* costs you
nothing but the missed opportunity; setting it too *low* is the failure mode, because it
over-moves and pays more in remote reads than it recovers in idle time.

Effect on the testbed (`skew_after`, remote penalty included):

| `r` | Qwen1.5-MoE | gemma-4-26B |
|---|---|---|
| 1.0 | 1.34 → **1.00** | 1.25 → **1.00** |
| 1.3 | → 1.05 | → 1.04 |
| 1.5 (default) | → 1.09 | → 1.07 |
| 2.0 | → 1.24 | → 1.13 |

### Measuring `r` on the real machine

**This has not been measured on Pandora yet**, which is why the knob ships off. The testbed
cannot answer it: under a fake topology `mbind` is a no-op, so every "remote" read is really
local. Three runs at identical model, threads, context and prompt:

```sh
A: --numa mirror                                      # all local, balanced
B: --numa-mirror dense,kv  GGML_NUMA_SHARD_SCHED=0    # same schedule, ~half the reads remote
C: --numa-mirror dense,kv                             # node-aware, local, imbalanced
```

`GGML_NUMA_SHARD_SCHED=0` keeps stage-1 placement but restores the old schedule, so A and B do
the *same* work in the *same* split and differ only in locality — which is what isolates the
penalty instead of confounding it with imbalance.

- **`r ≈ B/A`** (on the TG number; that is where the skew bites).
- **`C/A`** is what node-aware scheduling nets today, after paying the skew.

Then set `GGML_NUMA_SHARD_STEAL_COST` to the measured `r`, enable
`GGML_NUMA_SHARD_STEAL=1`, and confirm with `GGML_NUMA_STATS=1`:

```
numa_stats: moe_rebalance=1 cost=1.50 moves=648 (0.59 per op) skew_after=1.087 (from 1.340)
```

`moe_node_skew_mean` deliberately keeps scoring the untouched `e % n_nodes` split, so
before → after stays comparable across runs. Both knobs are env vars — no rebuild to change
either.

## Pandora validation suite (`scripts/pandora-tune.sh`)

Every knob that the fake-NUMA testbed shipped needs one confirmation sweep on the real
machine. `pandora-tune.sh` runs them and prints, per knob, the winning value and where
to put it. General form:

```sh
# CPU-only serving shape:
./scripts/pandora-tune.sh <sub> -m /models/GLM-5.2-Q5_K_XL.gguf -t 188 -r 2

# hybrid shape (dense on the Blackwells, experts on CPU): append your real flags
./scripts/pandora-tune.sh <sub> -m ... -g 0 -- -ngl 99 -ot exps=CPU -fa on
```

Recommended order and where each answer goes:

| # | Subcommand | Sweeps | Answer goes to |
|---|---|---|---|
| 1 | `census` | (one instrumented run) | gate for everything else: **"duplicated N weight tensors" must appear**, fallbacks 0, expert skew report |
| 2 | `threads` | `-t` 96–192 | `-t` on the serving command line |
| 3 | `barrier-gate` | `GGML_NUMA_HIER_BATCH_MAX` -1…512 | env, only if ≠ 32 |
| 4 | `pin` | `GGML_NUMA_PIN` node/cpu | env, only if cpu wins |
| 5 | `copy` | `GGML_NUMA_COPY_THREADS` × `GGML_NUMA_NT_COPY` | env; affects load + resync only |
| 6 | `spincount` (hybrid) | `GOMP_SPINCOUNT` 0–100k | env, only if ≠ 5000 (see Waiting policy above) |
| 7 | `reserve` (hybrid) | `GGML_NUMA_RESERVE_CPUS` 0–4@gpu-node | env, only if ≠ 0 |
| 8 | `bind-compute` (hybrid) | `--numa-bind-compute` | CLI flag, only if on wins |
| 9 | `hugetlb` (optional) | `GGML_NUMA_HUGETLB` | env; only bother on a long-uptime box (testbed: tg -6.5%) |
| 10 | `verify` | (final config) | record pp/tg as the tuned reference |

`all` runs 1–5 (plus 7–8 when hybrid flags are passed). Lists are overridable via env
(`GATE_LIST`, `COPY_THREADS_LIST`, `NT_LIST`, `THREADS_LIST`, `RESERVE_LIST`).

The expert-sharding knobs (`--numa-mirror dense`, `GGML_NUMA_SHARD_STEAL[_COST]`) have **no
`pandora-tune.sh` subcommand yet** — they only matter for models that cannot be mirrored, and
their one measurement is the three-run A/B in
[Expert sharding and rebalancing](#expert-sharding-and-rebalancing---numa-mirror-dense).

### Measured on Pandora — GLM 5.2, 2× EPYC 9665 (2026-07)

Results from the real machine, as opposed to the fake-NUMA testbed. These are the values to
start from; re-derive only if the hardware or model changes.

| knob | result | notes |
|---|---|---|
| `threads` | **`-t 128`** | coarse sweep 96 / 128 / 144 / …; 128 won. Below the 192 physical cores, as expected for bandwidth-bound TG — do not assume "all cores" |
| `spincount` | **5000** | coarse matrix; **2500 and 10000 were both significantly worse**, so the optimum is fairly sharp. This is now the built-in default (was 25000, which was the 8-core testbed optimum). A denser sweep around 5000 was started but not finished |
| `census` expert skew | **1.74** `max_over_mean` | per-expert routing skew for GLM 5.2. Comfortably shard-friendly (>>3 would mean a few hot experts dominate). Sits next to gemma-4's 1.78, which measured a *node* skew of ~1.25 |
| `numactl` / `GGML_NUMA_RESERVE_CPUS` | not exercised | untouched so far; the reserve knob is hybrid-only insurance |
| CUDA on the server | **resolved** | the long-standing `ggml_cuda_init: failed to initialize CUDA` with no reason string turned out to be a **Resizable BAR misconfiguration**, not a driver or library problem. Check ReBAR/above-4G-decoding in firmware before chasing `libcuda` stubs or `nvidia_uvm` |

Not yet measured, and it gates the expert rebalancer (`GGML_NUMA_SHARD_STEAL`): the
**remote-read penalty `r`** — how much slower an expert matmul is when its weights sit on the
other socket. Get it with three runs at identical model/threads/prompt:

```sh
A: --numa mirror                                     # all local, balanced
B: --numa-mirror dense,kv  GGML_NUMA_SHARD_SCHED=0   # same schedule, ~half remote  -> r = B/A
C: --numa-mirror dense,kv                            # node-aware, local, imbalanced
```

Then set `GGML_NUMA_SHARD_STEAL_COST` to the measured `r` before enabling the rebalancer.

### Sizing notes for a ~500 GB model (GLM 5.2 Q5_K_XL, 700B/A40B)

- **Mirror fits**: 2 × 500 GB ≈ 1 TB of the 2.3 TB. The free-RAM check needs
  `MemAvailable > weights + 2 GB` *after* the weights are loaded — with ~1.8 TB free
  that's comfortable, but anything else big running (vLLM) counts against it. The
  census run confirms mirroring actually happened.
- **Run cost**: the first load pulls 500 GB from disk (minutes); afterwards the file is
  page-cached (fits in leftover RAM), so each rep ≈ cached load + mirror populate
  (500 GB at the `copy`-tuned rate; ~25 s at 20 GB/s) + the benchmark itself. Budget
  roughly 5–10 min per rep; the full suite at `-r 2` is an overnight job — run
  subcommands selectively.
- **Census first**: verify GLM's expert routing skew (`max_over_mean`); gemma-4 and
  Qwen1.5-MoE both measured near-uniform (≤ 1.5·mean per expert), which validates
  mirroring over expert-sharding. If GLM measures similarly, mirror is settled.
- Do **not** wrap runs in `numactl --cpunodebind` (mirror re-pins and escapes it), and
  never set `--cpuset-mems`/`AllowedMemoryNodes` — mirrors must allocate on every node.

## 1. Verifying placement: `numastat` and `numa_maps`

### System-wide allocation counters

```
numastat            # cumulative per-node counters since boot
```

- `numa_hit` — allocations that landed on the preferred node.
- `numa_miss` / `numa_foreign` — allocations that had to go to a different node than
  intended. A steadily climbing `numa_miss` during model load means a node ran out of
  free pages and allocations spilled.
- `other_node` — allocations made by a process running on a different node.

These count **page allocations, not memory accesses** — good for catching misplaced
buffers, useless for measuring runtime traffic (use §3 for that).

### Per-process placement (the important one)

```
numastat -p $(pidof llama-server)
```

Shows the process's resident memory split per node. What you want to see with
`--numa mirror` on 2 nodes:

- The two weight mirrors roughly **balanced**: e.g. a 250 GB model shows ~250 GB on
  node 0 *and* ~250 GB on node 1.
- With `--numa-bind-compute` on `--numa-gpu-node 1`: the compute buffer
  (its size is printed at startup: `CPU compute buffer size = ... MiB`) counted
  entirely under node 1.

For a per-mapping breakdown (which exact buffer sits where):

```
grep -E 'N0=|N1=' /proc/$(pidof llama-server)/numa_maps | sort -t= -k2 -rn | head
```

Each line is one mapping; `N0=<pages> N1=<pages>` is its per-node page count
(multiply by `kernelpagesize_kB`, usually 4 KiB, or 2048 KiB for THP). The giant
anonymous mappings are the weight mirrors — each should be ~100 % on a single node.
A mapping split ~50/50 between nodes is first-touch memory written by threads on
both sockets; before this feature, that's exactly what the compute buffer looked like.

## 2. Verifying the core split

```
for t in /proc/$(pidof llama-server)/task/*; do
  taskset -pc $(basename $t) 2>/dev/null
done | sort | uniq -c | sort -rn | head
```

You should see two big groups of threads, one pinned to node 0's allowed CPUs and one
to node 1's. If threads show the *full* node CPU lists even though you launched under
a restricted core set, you used `numactl` — its affinity mask is **overridden** by
ggml's own per-node pinning. Use a cgroup instead (see §4).

## 3. Measuring cross-socket traffic

### On AMD EPYC: AMD uProf (`AMDuProfPcm`)

`pcm-numa` (Intel PCM) reads Intel uncore counters and **does not work on EPYC** —
use it on Intel boxes only. The AMD equivalent ships with AMD uProf (install the
`amduprof` RPM; on an airgapped box, bring it on the media — no other deps).

```
# 30 s of per-socket memory bandwidth (data-fabric counters, needs root + msr module)
AMDuProfPcm -m memory -a -d 30 -o /tmp/uprof-mem.csv

# per-die DRAM read/write bandwidth breakdown
AMDuProfPcm -m memory,ipc -a -d 30
```

Read it as: run tg with the A config, note per-socket DRAM bandwidth; both sockets
near their local ceiling = mirror is doing its job. Then compare xGMI/remote traffic
between A and B configs. (Column names vary by uProf version; the harness saves the
raw CSV rather than parsing it.)

### Portable fallback: `perf` with Zen data-fabric events

```
# does the kernel expose the events? (Zen 4/5 naming)
perf list 2>/dev/null | grep -i dram_io

# local vs far (other-socket) DRAM demand fills, system-wide, 10 s
perf stat -a -e ls_dmnd_fills_from_sys.dram_io_near,ls_dmnd_fills_from_sys.dram_io_far -- sleep 10
```

`dram_io_far / (near + far)` is your remote-access ratio for demand loads. With
mirror working correctly during tg it should be low single-digit percent; if it's
~50 % something isn't reading its local copy. (Event names differ across kernel
versions; `numa-ab.py` probes a candidate list and uses whatever the kernel accepts.)

### GPU transfer pressure

```
nvidia-smi dmon -s put -i <gpu-idx>    # rxpci/txpci columns, MB/s
```

Compare PCIe rx/tx during pp with `--numa-bind-compute` on vs. off. The hypothesis
being tested: same volume, lower effective latency when the DMA source pages are on
the GPU's socket.

## 4. Core-split recipes (co-existing with vLLM)

**Do not use `numactl --cpunodebind` / `--physcpubind` on llama.cpp in mirror mode.**
numactl only sets an inherited affinity mask, and ggml's worker threads immediately
re-pin themselves to full node CPU sets, escaping it. Cgroups are enforced by the
kernel: explicit affinity requests get intersected with the cgroup's allowed set, so
per-node pinning still works but stays inside your allocation.

```
# vLLM: hard-confined to its cores and node-0 memory
docker run --gpus '"device=0,1,2,3"' \
    --cpuset-cpus 0-15,192-207 --cpuset-mems 0 ... vllm ...

# llama.cpp: everything except vLLM's cores, spanning BOTH nodes (mirror needs both)
sudo systemd-run --scope --collect -p AllowedCPUs=16-191,208-383 \
    ./build/bin/llama-server -m model.gguf --numa mirror --numa-gpu-node 1 \
    -ngl 999 --cpu-moe -t 160 ...
```

(Example core numbering for 2× 96-core with SMT: node 0 = 0-95 + 192-287,
node 1 = 96-191 + 288-383 — confirm with `lscpu -e` or
`cat /sys/devices/system/node/node*/cpulist`.)

Notes:

- Never set `AllowedMemoryNodes` / `--cpuset-mems` on llama.cpp — the mirrors need
  to allocate on every node.
- The thread block-split assigns the first half of `-t` to node 0 and the second half
  to node 1. If vLLM takes 16 node-0 cores, node 0 has 80 free physical cores, so
  `-t 160` saturates node 0 exactly while node 1 idles 16 — start there and sweep.
- `echo 0 > /proc/sys/kernel/numa_balancing` before benchmarking, as always.

## 5. A/B protocol

1. Baseline correctness once per build:
   `llama-perplexity` over a few chunks with and without `--numa-gpu-node` /
   `--numa-bind-compute` — all variants must produce matching perplexity.
2. For each config, run identical `llama-cli` workloads (fixed prompt length, fixed
   `-n`, `--ignore-eos`) and record prompt-eval and eval t/s from
   `llama_print_timings`. 3+ repetitions; tg spread should be within ~2 %.
3. Capture `numastat -p` + `numa_maps` mid-run to confirm each config actually
   placed memory where you think it did (§1) — a benchmark of a misplaced buffer is
   a benchmark of nothing.
4. Only then compare traffic counters (§3) to explain *why* the numbers moved.

`scripts/numa-ab.py` automates steps 2–4 and prints a comparison table.

### Sweeping a whole test matrix

`numa-ab.py matrix` runs the cartesian product of thread counts × pin modes × flag
sets, then tabulates. Use a `{t}` placeholder where the thread count goes:

```
sudo ./scripts/numa-ab.py matrix --name sweep --reserve-node0 16 \
    --threads 128,160,176 --pin node,cpu \
    --extra '--numa-gpu-node 1|--numa-gpu-node 1 --numa-bind-compute' -- \
    ./build/bin/llama-cli -m model.gguf --numa mirror -t {t} -ngl 999 --cpu-moe \
    -p "<long prompt>" -n 128 --ignore-eos
```

That example is 3 × 2 × 2 = 12 runs; the report at the end shows throughput, remote
DRAM-fill ratio, per-node placement, and which env/flags each run used.

## Notes on cold-path costs

Mirror population at load and KV resync (context shift / K-shift, session restore)
use a parallel node-pinned copy (`ggml_numa_memcpy_to_node`, up to 16 threads pinned
to the destination node) instead of a single-threaded memcpy, so copies run at
roughly interconnect speed rather than one core's memcpy speed. Context shifts on a
large populated cache still copy the whole K cache once — expect a brief stall on
that event, not a per-token cost.
