# Pandora day-1 runbook — numa-mirror tuning

Print-out for the server session. Everything below is offline-safe; full details in
`docs/numa-tuning.md`, complete knob table in `README.md`, gotchas in `CLAUDE.md`.
Model path assumed `/models/GLM-5.2-Q5_K_XL.gguf` — adjust to taste.

## 0. One-time setup (first session only)

```sh
cd <repo root>
chmod +x scripts/*.sh          # zip drops exec bits
rm -rf build                   # Windows .exe junk from the carry package — delete it
```

## 1. Build (CPU-only is enough for today)

```sh
./scripts/build-zen.sh                                   # ~5-10 min
./build/bin/llama-cli --version                          # must run
# AVX-512 check: kernels live in the SHARED LIB, not the thin llama-cli binary —
# objdump the lib, or just look for the load-time banner
# "======================================= HAVE_FANCY_SIMD is defined"
# and "AVX512_VNNI = 1" in the system_info line of any run's log.
objdump -d "$(find build -name 'libggml*.so' | head -1)" | grep -c vpdpbusd   # want a big number
```

If cmake/gcc are missing, install from the offline-kit RPM repo first.

## 2. System prep (every boot)

```sh
echo 0 | sudo tee /proc/sys/kernel/numa_balancing
```

- Do NOT wrap anything in `numactl --cpunodebind` (mirror re-pins past it).
- Never set `--cpuset-mems` / `AllowedMemoryNodes` (mirrors must allocate on BOTH nodes).

## 3. GATE: census (do not tune until this passes)

```sh
./scripts/pandora-tune.sh census -m /models/GLM-5.2-Q5_K_XL.gguf -t 188
```

First run reads 500 GB from disk — expect several minutes of apparent hang. Then check:

- [ ] line `duplicated N weight tensors across 2 nodes` printed
      (missing = mirror silently skipped: MemAvailable < weights + 2 GB. Free memory
      — vLLM, hugetlb pools — and rerun.)
- [ ] `resolve_fallback=0` on both nodes
- [ ] moe census `max_over_mean` = ________  (informational only — no bearing on mirror
      correctness. ~1.0 = perfectly uniform routing, up to ~2.5 typical; it gauges the
      hypothetical expert-sharding alternative, nothing else. GLM measured 1.74 = fine.)

## 4. Sweeps (in this order; each prints "==> RESULT" with the value + where it goes)

Reps: `-r 2`. Budget ~5–10 min per rep after the first cached load. Record each answer.

| # | Command | Record result | Apply where |
|---|---------|---------------|-------------|
| 1 | `./scripts/pandora-tune.sh threads -m <model> -r 2` | `-t` = ________ | serving command line |
| 2 | `./scripts/pandora-tune.sh barrier-gate -m <model> -t <best> -r 2` | `GGML_NUMA_HIER_BATCH_MAX` = ________ | env, only if ≠ 32 |
| 3 | `./scripts/pandora-tune.sh pin -m <model> -t <best> -r 2` | `GGML_NUMA_PIN` = ________ | env, only if `cpu` won |
| 4 | `./scripts/pandora-tune.sh copy -m <model> -t <best> -r 2` | `GGML_NUMA_COPY_THREADS` = ______ , `GGML_NUMA_NT_COPY` = ______ | env; affects load/resync only |

CPU-only serving? Skip to step 6. Running hybrid (GPU dense + CPU experts)? Do step 5.

## 5. Hybrid sweeps (needs the CUDA build)

```sh
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
      -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89;120"
cmake --build build-cuda -j
```

`-g` = the GPU's NUMA node (`nvidia-smi topo -m`; Blackwells = node 0, Ada = node 1).
Reminder: this fork's `-fa` takes a value → `-fa on`.

| # | Command | Record | Apply where |
|---|---------|--------|-------------|
| 5a | `./scripts/pandora-tune.sh spincount -m <model> -b build-cuda/bin/llama-cli -t <best> -g <N> -- -ngl 99 -ot exps=CPU -fa on` | `GOMP_SPINCOUNT` = ________ | env, only if ≠ 25000 (auto-default; pair with `OMP_WAIT_POLICY=PASSIVE`) |
| 5b | `./scripts/pandora-tune.sh reserve   ... same flags ...` | `GGML_NUMA_RESERVE_CPUS` = ______@____ | env, only if ≠ 0 |
| 5c | `./scripts/pandora-tune.sh bind-compute ... same flags ...` | on / off = ________ | `--numa-bind-compute` CLI flag, only if `on` won |

## 6. Final check + apply

```sh
./scripts/pandora-tune.sh verify -m <model> -t <best>     # record: pp ______  tg ______
```

Put the recorded env values in the serving unit, e.g.:

```ini
# systemd
Environment=OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=<val> GGML_NUMA_PIN=<val> ...
```

or `export` lines in the wrapper script / `environment:` in docker compose. Flags
(`-t`, `--numa mirror`, `--numa-gpu-node N`, `--numa-bind-compute`) go on the
llama-server command line.

While a server runs, sanity-check placement from another shell:

```sh
numastat -p $(pgrep llama-server)     # both nodes should each hold ~= the model size
```

## If something looks wrong

- A pp or tg column reads 0: the timing-line parse missed. The raw lines are saved in
  `pandora-tune-results/last-timings.log` (or `tune-spincount-timings.log`) — eyeball
  the `eval time` line there; the throughput number sits right before "tokens per
  second" and the parser expects exactly that phrase.

- Garbage/weird text ≠ backend bug until proven: instruct models degenerate on raw
  prompts — retest with the model's chat template before debugging.
- Sweeps too slow? Narrow lists via env: `GATE_LIST="32 128"`, `THREADS_LIST="160 176 192"`,
  `COPY_THREADS_LIST="16 32"`, `RESERVE_LIST="0 2"`, `SPINS="0 25000 100000"`.
- Any run without `duplicated N weight tensors` in its log ran UNMIRRORED — its numbers
  are garbage; fix memory pressure and redo.
