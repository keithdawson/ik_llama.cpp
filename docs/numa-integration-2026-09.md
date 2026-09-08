# NUMA upstream integration — 2026-09-07

Integrated upstream `fe215a8` (2026-09-03) into the fork's `numa-testbed` lineage, including its existing working changes. The stable deployment branch is `numa-mirror`.

## Changes

- Preserve weight/KV mirroring, expert-affinity sharding, shared-expert replication, deterministic rebalancing, and NUMA statistics.
- Default hybrid mirror runs to `OMP_WAIT_POLICY=PASSIVE` and `GOMP_SPINCOUNT=7000`, following the user's additional Pandora testing. Explicit environment settings remain respected. GOMP_SPINCOUNT is a spin count, not milliseconds.
- Default mirror workers to individual CPU affinity. Use `GGML_NUMA_PIN=node` for node-wide affinity.
- Retain `--numa-exclude-cpus` and pageable CPU expert allocation from the original uncommitted work; reject exclusions that empty a node and bound parsed ranges.
- Route mirror runs around upstream's new global expert-chunk scheduler, which otherwise bypasses local mirror pointers and expert ownership. Non-mirror runs retain upstream chunking.
- Resolve cache-restore conflict by keeping both NUMA resynchronization and upstream recurrent/cache restoration work. Preserve distinct GLM-DSA and openPangu GGUF metadata names.
- Update pinning comparisons and reject empty smoke-test output.

## Validation

Full Linux CPU Release build passed, including llama-cli, llama-server, and test targets. Default upstream expert chunking was enabled during testing.

Dense Qwen2.5-0.5B smoke gate passed: mirrored and baseline generation identical; both fake nodes resolved local weights; zero fallback counters; KV replication active; node and individual CPU affinity outputs identical.

Qwen1.5-MoE-A2.7B IQ4_XS smoke gate passed: whole experts assigned to both nodes, shared experts mirrored, baseline/sharded/rebalanced generation identical. Rebalancing moved 804 experts at cost 1.0 and 162 at cost 2.0.

GGUF imports and distinct GLM/openPangu indexer names passed.

18 of 20 selected CTest tests passed. BERT tokenization and the ChatGLM4 chat-template expectation also fail on a separately compiled, unmodified upstream fe215a8 checkout. Initial Windows line-ending fixture failures were resolved by restoring repository LF endings.

## Model and hardware limits

**Kimi K3 is not implemented by this merge.** Upstream fe215a8 still lacks its architecture registration and graph. The model-level expert limit also remains 512. KDA primitives alone do not establish K3 support. A separate architecture port and real GGUF validation are required; the older HANDOFF assessment must not be treated as a current compatibility guarantee.

Existing sharding distributes whole routed experts across the two NUMA nodes. It does not split each individual expert tensor across both nodes and reduce partial results. That tensor-parallel scheme remains future work.

Fake-NUMA tests validate scheduling and output, not physical page locality, xGMI performance, or target throughput. CUDA builds and large-model inference on Pandora were not validated in this session. Keep rebalancing off until its remote-read cost is measured on Pandora.

## Pandora launch settings

For a supported MoE model too large for full weight mirroring:

```sh
export OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=7000
export GGML_NUMA_PIN=cpu
export GGML_NUMA_STATS=1
./build/bin/llama-server -m /models/model.gguf --numa mirror --numa-mirror dense,kv -t 128 -c 32768 --jinja
```

128 threads is the older documented Pandora starting point, not a new benchmark result. Set context and GPU offload flags for the actual model. Confirm expert placement on both nodes and zero unintended fallbacks before throughput measurements. Build the carried source on Pandora with its offline toolchain; this source archive does not bundle compilers or model weights.

## Project summaries reviewed

- airgapped-coding-clis: offline coding-client deployment connected to the LAN inference service.
- cache_calculator: model architecture, KV-cache, and GPU capacity calculations (no top-level summary file).
- EXL3: offline inference staging, EXL3/TabbyAPI and vLLM deployment, hardware/toolchain handoff.
- ik_llama.cpp-numa_mirror: dual-socket CPU/hybrid inference with mirror and expert-sharding work.
- modtran_at_home: offline atmospheric transmission/radiance modeling.
- pandora-tools: CUDA/PCIe/NCCL diagnostic sources; summarized by PANDORA_START_HERE.md.
- project_tracker: offline FastAPI/SQLite project dashboard and LAN LLM integration.
- rtx6kpro: community hardware, model-serving, and benchmarking reference.
- xray_spectrum_calc: offline X-ray attenuation, fluorescence, and filter optimization.

Shared deployment constraint: stage on this Windows machine and physically carry source/artifacts to the airgapped Rocky Linux Pandora server.