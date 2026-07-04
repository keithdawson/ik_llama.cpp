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

`--numa-bind-compute` and `GGML_NUMA_PIN=cpu` are the experiments: whether they help
depends on batch size, cache behavior, and what else runs on the box. Measure, don't
assume — `numa-ab.py matrix` (below) sweeps them for you.

Build note: make sure the build has **OpenMP** enabled (it is by default — verify with
`grep GGML_OPENMP build/CMakeCache.txt`, or `ldd build/bin/llama-server | grep gomp`).
Without OpenMP, worker threads are created and joined on *every* graph compute, which
is real per-token overhead at high thread counts.

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
