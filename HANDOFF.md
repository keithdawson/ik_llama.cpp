# HANDOFF — pick up here

Written 2026-07-28 (updated same day, second session). Read this first, then `CLAUDE.md`
(operator index), `docs/numa-tuning.md` (server playbook), `docs/numa-testbed.md`
(testbed + verdicts), `SERVE-GLM-NOTES.md` (GLM/Open-WebUI/MTP/CUDA-init troubleshooting).

---

## 1. STATE: upstream merge is DONE and pushed; expert sharding stages 1-2 are DONE

### Upstream merge — committed, verified, PR open

`upstream/main` @ `b054a8b9` merged into the fork (111 commits since the `6c00e87a` fork
point). Commit `d06027e8` on branch `upstream-merge`, pushed to `keithdawson`.

**PR: https://github.com/keithdawson/ik_llama.cpp/pull/1** (base `numa-mirror`, 113 commits).
Nothing else is needed for the merge — review and merge it on GitHub, or
`git checkout numa-mirror && git merge --ff-only upstream-merge` locally.

Conflicts resolved (all three exactly as the previous handoff planned; no NUMA code was inside
any conflict hunk):

- `ggml/include/ggml-cuda.h` — took upstream's `ggml_backend_cuda_init(int, const void *,
  const void * model)` / `ggml_backend_cuda_invalidate_graphs(const void * model)`. All call
  sites across the tree already matched the new arity.
- `ggml/src/ggml.c` — `ggml_compute_forward_sum_rows_f32` only; took upstream's any-dim superset.
- `src/llama.cpp` — kept BOTH the upstream DSA `kr_l` restore and our NUMA KV resync, resync
  last. Inline note left: if `kr_l` ever joins `llama_mirror_kv_cache`, it needs a resync loop.

Verification (the previous handoff expected build breakage from API churn — there was none):

- CPU AVX-512 build: clean, 0 errors.
- **CUDA build `-DCMAKE_CUDA_ARCHITECTURES="89;120"`: clean, 0 errors**, all targets incl.
  `llama-server`. This is what exercises the merged GLM-DSA / indexer CUDA sources.
- Testbed smoke gate: all checks pass, zero resolve fallbacks, byte-identical greedy output.

### Expert-affinity sharding — stages 1 and 2 implemented on branch `numa-shard`

Branch `numa-shard` (based on the merge commit) has:

- `9f0c6cac` stage 1 — placement: `--numa-mirror dense,kv`
- `b02d9c75` stage 2 — node-aware MoE scheduling

`--numa-mirror dense,kv` mirrors every weight except the routed experts; those keep a single
copy with expert `e` pinned to node `e % n_nodes`, and the MoE kernels compute expert `e` only
on that node's threads. Footprint becomes `dense × n_nodes + experts × 1`.

Output is **byte-identical** to baseline on both testbed MoE models, across the fused default,
`-no-fmoe` and `-no-fug` paths. The existing smoke gate is unaffected.

**Read the measurement carefully — the obvious counter lies.** Cumulative per-node expert
totals come out at max/mean ≈ 1.01 and look perfectly balanced. They are not the number that
matters: every MoE op ends at a barrier, so each op runs at the pace of whichever node drew
more of its routed rows. The census now reports that per op, weighted by routed rows:

| model | TG (batch 1) | PP (batch 512) |
|---|---|---|
| Qwen1.5-MoE-A2.7B | `moe_node_skew_mean` **1.32**, max **2.00** | **1.07**, max 1.41 |
| gemma-4-26B-A4B | **1.25**, max **2.00** | — |

`max = 2.00` means ops where one node held every routed row and the other idled. That is the
10-30% TG haircut the plan predicted, now measured. **`--numa-mirror dense` is for models that
do not fit mirrored — it is not a speedup over mirroring.**

The skew counter is computed whether or not sharding is enabled, so a plain `--numa mirror`
run predicts the cost for a given model. **Do this for Kimi K3 before committing to sharding.**

---

## 2. NEXT: pick one

### (a) Cross-node work stealing — the mitigation for the skew above
A node that finishes its experts helps with the other's (remote reads), converting idle into
slower-but-useful work. Needs a second barrier or an atomic per-expert claim in the loops in
`ggml/src/ggml.c` (`ggml_numa_expert_scope` is the hook — it currently answers a pure
`e % n_nodes` question and would instead consult a claim table).

Deliberately not attempted yet: the testbed's `mbind` is a **no-op**, so it cannot show the
remote-read cost being traded against the idle time, and the design would be tuned blind.
This wants Pandora numbers first.

### (b) Stage 3 — tensor-parallel expert split
Split each expert's up/gate by output rows and down by input rows so every node does half of
every expert → perfect balance, all-local reads. `ffn_down` contracts over the split dimension,
so it needs a small cross-node all-reduce per expert per token (~`n_embd` floats). **Not
bit-identical** (different summation order), so the smoke gate's byte-identity check needs a
tolerance-based variant for this stage.

User asked to study for the reduction design:
- `D:\AI-projects\getVLLMworking\b12x-latest\` — B12X PCIe allreduce kernels
- `D:\AI-projects\getVLLMworking\vllm\` — vLLM custom all-reduce (chunking, sync, one-shot vs
  two-shot thresholds)
- `D:\AI-projects\getVLLMworking\blackwell-llm-toolkit\` — reference recipes

Relevant transfer: small-payload cross-device reduction over a slow-ish link is structurally
the same problem as cross-socket reduction over xGMI.

---

## 3. Open items / unfinished (unchanged from the previous session)

- **Finer `GOMP_SPINCOUNT` sweep** around 5000 — user started it on the server, left before it
  finished. `SPINS="2500 4000 5000 6000 7500" ./scripts/tune-spincount.sh …`
- **Kimi K3 arch support is UNVERIFIED.** No Kimi arch in `src/llama-arch.cpp` (K2 rode on
  deepseek2). `GGML_TYPE_MXFP4 = 39` exists so the quant type is fine — the *architecture* is
  the question. Cheapest test: point llama-cli at a K3 GGUF and look for an unknown-architecture
  error. (The upstream merge did not obviously add it.)
- **Server CUDA init still unresolved.** `ggml_cuda_init: failed to initialize CUDA` with no
  reason string, on bare metal. Next step is `scripts/cuda-probe.cu`
  (`nvcc -o /tmp/cuda-probe scripts/cuda-probe.cu && /tmp/cuda-probe`). Interpretation table in
  `SERVE-GLM-NOTES.md`. Ranked suspects: stub `libcuda.so` on `LD_LIBRARY_PATH`, then
  driver-module↔library mismatch (reboot), then nvidia_uvm.
- **MTP not yet confirmed working** — retest with `llama-server` (`llama-cli` ignores
  `--spec-type`). Requires `-np 1`.
- GLM serving on the server works CPU-only with `--jinja` at ~12.3 t/s. Always pass an explicit
  `-c`.

## 4. Branch state

| ref | commit | note |
|---|---|---|
| `upstream-merge` | `d06027e8` | merge commit; **pushed**, PR #1 open against `numa-mirror` |
| `numa-shard` | `b02d9c75` | stages 1-2 of expert sharding, based on the merge |
| `numa-testbed` | `1e81bd4e` | now an ancestor of the merge |
| `numa-mirror` | `8b90ba20` | advances when PR #1 merges |
| `upstream/main` | `b054a8b9` | merge target |

Remotes: `keithdawson` = our fork (push here), `upstream` = ikawrakow/ik_llama.cpp,
`origin` = mikechambers84 (unused).

## 5. Testbed cheat sheet

Everything runs in Docker (`ik-numa-testbed`, CUDA variant `ik-numa-testbed-cuda`), volumes
`ik-build` (/build), `ik-models` (/models), `ik-ccache` (/ccache). Git Bash needs
`MSYS_NO_PATHCONV=1` for docker path args. Benchmark cpuset is `0,2,4,6,8,10,12,14` (WSL2 SMT
siblings are adjacent pairs). Orchestrator: `scripts/testbed/run-testbed.ps1`.
Models: gemma-4-26B-A4B UD-IQ4_XS, Qwen1.5-MoE-A2.7B IQ4_XS, Qwen2.5-0.5B.

Never pass `--memory` to the containers (the mirror free-RAM check reads `/proc/meminfo`, sees
the whole WSL2 VM, and would OOM-kill silently instead).

Two traps that cost time this session:

- **`-fmoe` is not a flag.** Fused MoE is on by **default**; only `-no-fmoe` exists. Passing
  `-fmoe` makes llama-cli exit with `error: unknown argument`, which looks exactly like "the
  feature silently didn't engage" if you only grep the log for your own log lines.
- **Guard A/B comparisons against empty output.** gemma-4 with `--no-mmap` OOM-killed under
  page-cache pressure; both sides produced nothing and `cmp` happily reported "identical".
  Check the byte count before trusting a match.
