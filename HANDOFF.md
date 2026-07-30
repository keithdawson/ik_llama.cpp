# HANDOFF — pick up here

Written 2026-07-28. Read this first, then `CLAUDE.md` (operator index),
`docs/numa-tuning.md` (server playbook), `docs/numa-testbed.md` (testbed + verdicts),
`SERVE-GLM-NOTES.md` (GLM/Open-WebUI/MTP/CUDA-init troubleshooting).

---

## 1. STATE

### Upstream merge — merged and deployed

`upstream/main` @ `b054a8b9` (111 commits since the `6c00e87a` fork point) is merged as
`d06027e8`. **PR #1 is merged**; `numa-mirror` and `numa-testbed` both carry it.
`upstream-merge` and `numa-shard` have been deleted — their work is in `numa-testbed`.

Verified before landing: CPU AVX-512 build clean, **CUDA build `89;120` clean** (all targets
incl. `llama-server`), smoke gate passing with byte-identical output.

### Expert-affinity sharding — stages 1-2 done, on `numa-testbed`

`--numa-mirror dense,kv` mirrors every weight except routed experts; those keep one copy with
expert `e` pinned to node `e % n_nodes`, and the MoE kernels compute expert `e` only on that
node's threads. Footprint = `dense × n_nodes + experts × 1`. Byte-identical output on both
testbed MoE models across fused-default / `-no-fmoe` / `-no-fug`.

**Read the skew number, not the totals.** Cumulative per-node expert counts come out ~1.01 and
look perfectly balanced; that is an artifact of averaging. Every MoE op ends at a barrier, so
each op runs at its busiest node's pace. `GGML_NUMA_STATS=1` now reports the per-op,
row-weighted truth:

| model | TG (batch 1) | PP (batch 512) |
|---|---|---|
| Qwen1.5-MoE-A2.7B | `moe_node_skew_mean` **1.32**, max **2.00** | **1.07**, max 1.41 |
| gemma-4-26B-A4B | **1.25**, max **2.00** | — |

`max = 2.00` = ops where one node held every routed row and the other idled. The counter works
with sharding *off*, so a plain `--numa mirror` run predicts the cost for any model.

**`--numa-mirror dense` is for models that don't fit mirrored — not a speedup over mirroring.**

---

## 2. RESEARCH FINDINGS — read before planning K3 work

### Kimi K3 will not load today, but the port is mostly graph plumbing (revised 2026-07-29)

From `moonshotai/Kimi-K3` `config.json` and the llama.cpp work:

| | value |
|---|---|
| architectures / model_type | `KimiK3ForConditionalGeneration` / `kimi_k3` |
| layers | 93 (layer 0 dense, `first_k_dense_replace=1`) |
| hidden size | 7168 |
| routed experts | **896**, top-**16**, `moe_intermediate_size` 3072 |
| **shared experts** | **2** (always active) |
| attention | **24 MLA (full) + 69 KDA (Kimi Delta Attention, linear/recurrent)** |
| extras | AttnRes (cross-layer residuals, softmax mixture over depth), Stable LatentMoE (router projects to latent 3584), multimodal vision, 1M context |
| quant | `mxfp4-pack-quantized` (compressed-tensors, group 32); **self-attention, shared experts, MLPs and lm_head excluded from quantization** → those stay bf16 |

Status as of **2026-07-29** (rechecked; the first read of this was too pessimistic):
- **ik_llama has no Kimi arch** — not `kimi_k3`, not `kimi_linear`. `upstream/main` @ `6647db9c`
  is one commit past our merge base and has nothing K3. Related upstream branches: `ik/mxfp4`,
  `ik/mxfp4_r8` (quant kernels), `ik/fix_kimi2_parse` (**K2**, not K3).
- **Mainline llama.cpp has not merged it either.** Open PR
  [ggml-org/llama.cpp#26185](https://github.com/ggml-org/llama.cpp/pull/26185): +1939/-30 over
  21 files, 8 commits, still moving (updated 2026-07-29), community-validated on real
  checkpoints, awaiting maintainer review. Groundwork:
  [#26041](https://github.com/ggml-org/llama.cpp/discussions/26041).

**The port is much smaller than "an architecture project".** The giveaway is that PR #26185
touches **no `ggml/` files at all** — it is built from ops mainline already had, and *this fork
already has those ops too*, inherited from two models it already supports:

| K3 needs | we have | via |
|---|---|---|
| KDA (gated delta attention) | `ggml_delta_net(q,k,v,g,beta,state,saved_steps)` | Qwen3-Next |
| AttnRes cross-layer residuals | `ggml_hc_pre` / `ggml_hc_post` | DeepSeek-V4 (mainline: `ggml_dsv4_hc_pre`) |
| MLA | ✓ | deepseek2 / GLM-DSA |
| MXFP4 | `GGML_TYPE_MXFP4 = 39` — commented *"so we are compatible with mainline"* | ik/mxfp4 |
| hybrid linear+full attention KV cache | `llm_arch_is_hybrid` | Qwen3-Next |

Qwen3-Next is itself a hybrid linear+full attention MoE, so K3's structural pattern is already
exercised here. Posted MXFP4 GGUFs should be **byte-compatible** — that type ID was matched on
purpose.

Remaining work, roughly in order of cost:
1. `LLAMA_MAX_EXPERTS` 512 → ≥896 (`src/llama-hparams.cpp:10`; mainline went to 1024).
2. **`situ` activation is genuinely missing**, plus `situ_beta` / `situ_linear_beta`.
3. Arch registration: `"kimi-k3"`, 5 new KV keys (`attn_res.block_size`,
   `activation.situ_beta`, `activation.situ_linear_beta`, `expert_latent_length`,
   `kda.gate_lower_bound`) and 7 new tensors (`ssm_g`, `attn_res_score`, `ffn_res_score`,
   `output_res_score`, `ffn_routed_{up,down,norm}`).
4. **Graph build — the real work**: port the 645-line `src/models/kimi-k3.cpp` onto this fork's
   `src/graphs/build_*.cpp` API. Different API, so a rewrite rather than a copy.
5. Conversion (390-line `conversion/kimi_k3.py`; mainline restructured into a `conversion/`
   package while we still have a monolithic `convert_hf_to_gguf.py`) — **skippable if we load
   community GGUFs instead of converting.**
6. Chat template + reasoning/tool parser (`models/templates/Kimi-K3.jinja` + `common/chat.cpp`);
   ours is the `chat-peg-parser`, theirs is not.

**Our expert sharding already lands correctly on K3 — no changes needed.** Verified through the
conversion: `MODEL_TENSOR.FFN_{GATE,UP,DOWN}_EXP` (singular) maps to the *plural* name string
`blk.{bid}.ffn_*_exps`, and the converter stacks experts into one `[n_expert, rows, cols]`
tensor. So routed experts get sharded, while `ffn_routed_{up,down,norm}` (the latent-MoE
projections — per-layer, shared across experts, so dense-like) and `_shexp` fall into the
mirrored set, which is right. K3 is also the case that *needs* sharding: ~1.5 TB at mxfp4 means
mirroring wants ~3 TB and does not fit, whereas sharding is ~1.5 TB + dense×2.

**Risks:** the PR is unmerged and still changing, so porting now means re-porting if reviewers
ask for changes. And one known upstream bug hits this deployment directly — **prefix-cache reuse
corrupts recurrent KDA state in multi-turn conversations**, which for an Open-WebUI setup is not
cosmetic. Low-bit (≤Q2) quality degradation is also reported.

**Recommended sequence:** wait for #26185 to merge before porting the graph, but do items 1-3
now (stable regardless of PR churn), then point `llama-cli` at a posted GGUF to turn the
remaining gap into a concrete unknown-tensor/KV error list instead of speculation.

Also new upstream: `ik/cpu_chunked_experts` (+236 in `ggml.c`, plus `getrows.cu`). It does
**not** touch `mul_mat_id`, so it should not conflict with our stage-2 edits — but it is MoE
CPU work, so check it before the next upstream merge.

### GLM-5.2 does have shared experts — one, not two

From `zai-org/GLM-5.2` `config.json`: `GlmMoeDsaForCausalLM` / `glm_moe_dsa`, 78 layers
(`first_k_dense_replace=3`, `num_nextn_predict_layers=1`), hidden 6144, **256 routed experts,
top-8, `n_shared_experts: 1`**, `moe_intermediate_size` 2048, plus the DSA indexer fields
(`index_topk 2048`, `index_n_heads 32`, `kv_lora_rank 512`, `q_lora_rank 2048`).

`LLM_ARCH_GLM_DSA` in this fork matches `glm_moe_dsa`, and the merge brought a lot of upstream
GLM-DSA perf work. So GLM-5.2 is the supported model; K3 is not.

Shared-expert weight is small in both cases — roughly **2.8B params for GLM-5.2**
(1 × 75 layers × 3 × 6144 × 2048) and **12.2B for K3** (2 × 92 layers × 3 × 7168 × 3072, bf16
≈ 24 GB since K3 excludes them from quantization). Either fits on one Blackwell, and mirroring
them per node is cheap.

### Shared experts already land where they need to — verified, not assumed

Both halves of the requirement are already satisfied; the code now says so explicitly.

1. **GPU offload works.** `--cpu-moe` / `--n-cpu-moe` expand to the pattern
   `blk\.N\.(ffn_(up|down|gate|gate_up)_exps\.(weight|scale))` (`src/llama.cpp` ~4234), which
   matches **only routed experts**. `_shexp` tensors are not matched, so they follow `-ngl` onto
   the GPU. Same for `-ot exps=CPU` — `"exps"` is not a substring of `"shexp"`.
2. **CPU mirroring works.** `llama_numa_is_routed_expert()` excludes `_shexp` (and
   `ffn_norm_exps`), so shared experts sit in the mirrored dense set. Verified against the GGUF:
   Qwen1.5-MoE has 411 tensors = 339 mirrored + 72 sharded, and all **96** `_shexp` tensors are
   in the mirrored 339. The loader now logs
   `NUMA shard: 96 shared-expert tensors (0.42 GiB) mirrored, not sharded`.

**Why this matters for the reduction:** a replicated shared expert is computed redundantly on
each node from local weights and added locally, so it never enters a cross-node all-reduce.
Sharding it instead would pin a permanently-hot tensor to one node *and* drag it into the
reduction. Keep it replicated when stage 3 lands.

### One upstream bug fixed on the way

`gguf-py` could not be imported at all — upstream `b054a8b9` defines
`INDEXER_{K_NORM,PROJ,ATTN_K,ATTN_Q_B}` twice in `MODEL_TENSOR` (GLM-DSA's and openPangu's),
so every `from gguf import ...` raised `TypeError: 'INDEXER_K_NORM' already defined`. Worse,
`TENSOR_NAMES` is a flat dict, so openPangu's entries *overwrote* GLM-DSA's — GLM-DSA
conversion would have emitted `blk.N.attn_indexer_*` instead of `blk.N.indexer.*`. Fixed by
giving openPangu its own `PANGU_INDEXER_*` members (commit `1758bd39`). Not ours: our pre-merge
side had none of these.

---

## 2b. DONE: cost-aware expert rebalancing (`GGML_NUMA_SHARD_STEAL`, default off)

The "work stealing" idea, built as **deterministic rebalancing** rather than opportunistic
stealing. Each MoE op re-derives its expert→node assignment from `matrix_row_counts`, which
`ith==0` already fills and the pre-loop barrier already publishes. Because every thread runs the
same greedy pass over the same data, they agree without atomics or an extra barrier, each expert
is still computed in full by one node, and **output stays bit-identical** — which opportunistic
stealing would have cost us.

Moving an expert off its owner makes its reads remote, so a move only pays when the busy node
sheds more than the idle node takes on. That ratio is `GGML_NUMA_SHARD_STEAL_COST` (`r`,
default 1.5). The algorithm is self-limiting — at high `r` it simply declines to move:

| `r` | Qwen1.5-MoE | gemma-4-26B |
|---|---|---|
| 1.0 | 1.34 → **1.00** | 1.25 → **1.00** |
| 1.3 | → 1.05 | → 1.04 |
| 1.5 | → 1.09 | → 1.07 |
| 2.0 | → 1.24 (few moves) | → 1.13 |

Stats print `moe_rebalance=1 cost=… moves=… skew_after=… (from …)`. Note `moe_node_skew_mean`
always scores the *untouched* `owner()` split so before/after stay comparable.

**Default off pending the `r` measurement** — per this repo's convention for unvalidated knobs.
An `r` set too low over-moves and pays more in remote reads than it saves in idle.

---

## 3. LATER: stage 3 — tensor-parallel expert split

Only worth it if rebalancing leaves too much on the table. It removes the remaining skew
entirely, at the cost of giving up bit-identity: split **each** expert across nodes instead of
assigning whole experts, so every node does an equal share of every expert.

- `ffn_up` / `ffn_gate`: split by **output rows**. Each node computes its own row block from
  local weights. No communication.
- `ffn_down`: contracts over the split dimension, so each node produces a **partial sum** over
  `n_embd` and they must be **all-reduced** across nodes, once per expert per token.
- Shared experts stay **replicated** (above) — computed redundantly, added locally, never
  reduced.

Consequences to plan for:
- **Not bit-identical** (different summation order). The smoke gate's byte-identity check needs
  a tolerance-based variant for this path — do not weaken the existing check, add a sibling.
- The all-reduce is small (~`n_embd` floats per expert per token) but frequent, which is
  exactly the regime where implementation quality dominates.

Study material the user flagged, all local:
- `D:\AI-projects\getVLLMworking\b12x-latest\` — B12X PCIe allreduce kernels
- `D:\AI-projects\getVLLMworking\vllm\` — vLLM custom all-reduce: chunking, sync strategy,
  one-shot vs two-shot thresholds
- `D:\AI-projects\getVLLMworking\blackwell-llm-toolkit\` — reference recipes

Transfer is structural: small-payload reduction over a slow-ish link (PCIe/NVLink there,
xGMI here) — one-shot for small payloads, two-shot/ring for large, careful barrier placement.

**Alternative worth pricing first:** cross-node work stealing on top of stage 2 (a node that
finishes its experts helps the other's, eating remote reads instead of idling). Much smaller
change, no reduction, still bit-identical. It converts idle into slower-but-useful work rather
than eliminating imbalance. Blocked on real hardware to tune, since the testbed's `mbind` is a
no-op and cannot show the remote-read cost being traded.

---

## 4. Open items / unfinished

- **Finer `GOMP_SPINCOUNT` sweep** around 5000 — the coarse matrix confirmed **5000** (both
  2500 and 10000 were significantly worse, so the optimum is sharp); a denser sweep was started
  on the server but not finished.
  `SPINS="4000 4500 5000 5500 6000" ./scripts/tune-spincount.sh …`
- **Remote-read penalty `r` unmeasured** — needed to set `GGML_NUMA_SHARD_STEAL_COST` and to
  justify enabling the rebalancer at all. Procedure in `docs/numa-tuning.md`
  ("Measured on Pandora"). Until then `GGML_NUMA_SHARD_STEAL` stays default-off.
- ~~Server CUDA init~~ **RESOLVED**: it was a **Resizable BAR** misconfiguration, not a driver
  or library problem. CUDA works on the server now. Worth remembering that the symptom was
  `ggml_cuda_init: failed to initialize CUDA` with *no reason string* while `nvidia-smi` was
  perfectly happy — check ReBAR / above-4G decoding before chasing `libcuda` stubs.
- **MTP not confirmed** — retest with `llama-server` (`llama-cli` ignores `--spec-type`),
  requires `-np 1`.
- GLM serving works CPU-only with `--jinja` at ~12.3 t/s. Always pass an explicit `-c`.

## 5. Branch state

| ref | commit | note |
|---|---|---|
| `numa-mirror` | `d06027e8` | upstream merge — **deploy from here** |
| `numa-testbed` | `1758bd39`+ | merge + shard stages 1-2 + gguf-py fix |
| `kv-unified` | `0b60444a` | experimental, untouched |

Remotes: `keithdawson` = our fork (push here), `upstream` = ikawrakow/ik_llama.cpp,
`origin` = mikechambers84 (unused).

## 6. Testbed cheat sheet

Docker (`ik-numa-testbed`, CUDA variant `ik-numa-testbed-cuda`), volumes `ik-build` (/build),
`ik-models` (/models), `ik-ccache` (/ccache). Git Bash needs `MSYS_NO_PATHCONV=1` for docker
path args. Benchmark cpuset `0,2,4,6,8,10,12,14` (WSL2 SMT siblings are adjacent pairs).
Orchestrator `scripts/testbed/run-testbed.ps1`. Models: gemma-4-26B-A4B UD-IQ4_XS,
Qwen1.5-MoE-A2.7B IQ4_XS, Qwen2.5-0.5B.

Never pass `--memory` to the containers (the mirror free-RAM check reads `/proc/meminfo`, sees
the whole WSL2 VM, and would OOM-kill silently instead).

Traps that cost real time:

- **`-fmoe` is not a flag.** Fused MoE is on by **default**; only `-no-fmoe` exists. Passing
  `-fmoe` exits with `error: unknown argument`, which looks exactly like "the feature silently
  didn't engage" if you only grep the log for your own log lines.
- **Guard A/B comparisons against empty output.** gemma-4 with `--no-mmap` OOM-killed under
  page-cache pressure; both sides produced nothing and `cmp` reported "identical".
- **`gguf-py` needs numpy** in the testbed container (`pip install numpy
  --break-system-packages`); it is not preinstalled.
