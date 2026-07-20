# GLM 5.2 + Open WebUI looping — diagnosis & fix sheet (print & carry)

**Symptom:** in Open WebUI the model loops, appearing to answer its own "tool calls".

**Most likely cause:** llama-server was launched without `--jinja`. This fork's built-in
template matcher only knows GLM-4-era templates (`[gMASK]<sop>` style); GLM 5.2's
template (thinking + tool syntax) doesn't match, so a wrong/generic fallback template is
applied. The model then sees malformed turns, emits its own role/tool markers
(`<|user|>`, `<|observation|>`, `<tool_call>`) as plain text, and role-plays both sides
— which reads exactly like "it thinks its response is a tool call and response."
Open WebUI's native function-calling and auto title/tag generation amplify the mess.

Work through the steps in order; each isolates one layer.

## 1. Confirm what template the server is using

```sh
curl -s http://localhost:<port>/props | head -c 2000
```

Look at the `chat_template` field. If it is NOT a long jinja template mentioning
tool/thinking handling (i.e. it's empty or some short generic thing), the fallback is
active → step 2 fixes it.

## 2. Relaunch llama-server with the template flags

```sh
./build/bin/llama-server -m /models/GLM-5.2-Q5_K_XL.gguf \
    --numa mirror -t <tuned> -c <ctx> \
    --jinja \
    --host 0.0.0.0 --port <port>
```

`--jinja` makes the server render the GGUF's own embedded template verbatim instead of
the built-in matcher. Optionally add `--reasoning-format` / `--think-tokens` handling
(see `--help`) so `<think>` blocks are stripped or wrapped for the client.

## 3. Test WITHOUT Open WebUI (isolates server vs UI)

```sh
curl -s http://localhost:<port>/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role":"user","content":"What is the capital of France? One sentence."}],
  "max_tokens": 200, "temperature": 0.7
}'
```

- Clean one-sentence answer (possibly with a think block) → server is fixed; any
  remaining weirdness is Open WebUI → step 4.
- Still emits `<|user|>` / `<|observation|>` / tool markers and keeps going → jinja
  rendering itself is at fault → step 5.

## 4. Open WebUI settings that cause exactly this

- **Function calling: set to "Default", not "Native"** (per-model settings). Native
  injects an OpenAI `tools` array; the template then renders a tool section and GLM
  answers in its native tool syntax, which OWUI doesn't parse back → looping tool
  theater. "Default" keeps tool prompting inside OWUI.
- **Detach all Tools/Functions** from this model in the workspace/model editor.
- **Admin Settings → Interface: set the Task Model** (title/tags/follow-up generation)
  to something else or off — otherwise every chat fires 2-3 hidden extra prompts at
  the 500 GB model.
- Advanced params for the model: temperature ~0.7–1.0, top_p 0.95, repeat penalty 1.0
  (GLM-recommended neighborhood; the defaults are fine once the template is right).
- If responses still run into a fake next turn, add stop strings `<|user|>` and
  `<|observation|>` in the model's advanced params — a band-aid; correct jinja + EOT
  normally makes it unnecessary.

## 5. If --jinja itself renders wrong (fallback)

Upstream fixed a jinja engine bug after our fork point that matters for loop-heavy
templates like GLM's: `997b289d "jinja: give each for-loop iteration a fresh scope
(#2018)"`. If step 3 still misbehaves with `--jinja`, cherry-pick that commit on the
dev box, rebuild, and re-carry:

```sh
git fetch upstream && git cherry-pick 997b289d
```

Also available as a band-aid without rebuilding: extract the template from the GGUF on
any machine (`gguf-py` or a hex-friendly editor), fix/simplify it, and pass
`--chat-template-file fixed.jinja`.

## MTP (multi-token prediction) — supported, opt-in, worth ~1.3-1.8x tg

The fork implements GLM's MTP head (build_glm4_moe_mtp) as a speculative stage.
Enable on the server command line:

```sh
--spec-type mtp:n_max=1,p_min=0.0
```

(the old `-mtp` shorthand intentionally errors and points to this syntax). Stronger
variant for coding chats — n-gram self-speculation first, MTP fallback:

```sh
--spec-type ngram-mod:n_max=64,n_min=2,ngram_size_n=8 --spec-type mtp:n_max=1,p_min=0.0
```

Caveats / checks, in order:

1. **Prerequisite**: the GGUF must contain the MTP/nextn tensors (unsloth quants
   normally do). The load log shows the MTP setup if present; a model without them
   can't form the stage.
2. **Verify losslessness once**: speculative decoding must produce IDENTICAL output to
   a non-speculative run for the same greedy request. Same curl with and without the
   flag, diff the text. Any difference = broken, turn it off.
3. **NUMA mirror interaction**: composes cleanly in principle (MTP weights are host
   weights -> mirrored; verify batches of 2-4 stay under the hier-barrier gate 32) but
   this exact combination was never testbed-exercised. First run: confirm the
   `duplicated N weight tensors` line and `resolve_fallback=0` as usual.
4. **Crash caveat**: upstream was actively fixing `--spec-type mtp` server crashes just
   after our fork point (regression + fix both postdate our tree, so we should be
   clean). If the server crashes with the flag, it's a cherry-pick job on the dev box,
   not a config problem — note the exact error.
5. **Measure it**: record tg with/without on a real coding prompt. Acceptance (and
   thus speedup) is best at low temperature and on structured/code text; expect
   roughly 1.3-1.8x when healthy (12.3 -> ~16-22 t/s). If tg gets WORSE, acceptance is
   poor for your workload — drop the ngram stage first, then MTP entirely.
6. `-mtprot` / `--mtp-requantize-output-tensor` exists as an extra knob for the MTP
   output tensor quant; leave default unless chasing quality issues.
7. Open WebUI needs nothing special — speculation is server-internal and invisible to
   the API.

## Boot-log readings (observed on the server, both explained)

- **`objdump ... llama-cli | grep -c vpdpbusd` = 0 is a FALSE ALARM** on this build:
  it links shared libs, so the AVX-512 kernels are in `libggml.so`, not the thin cli
  binary. Real checks: the load banner
  `======================================= HAVE_FANCY_SIMD is defined` and
  `AVX512_VNNI = 1` in the `system_info:` line. Only if the banner is missing:
  clean rebuild (`rm -rf build`) under the newest gcc-toolset. Missing it would
  mostly hurt pp, not tg.
- **`Free memory 0 MiB on device 0 is less the required compute buffer size 131072 MiB`
  is cosmetic on CPU-only builds**: the device free-memory query is a hardcoded stub
  (returns ~0) and the warning belongs to the multi-GPU split planner. BUT the 128 GiB
  estimate means the server was launched **without `-c`** → context defaulted to GLM's
  max (~200k). Set `-c` explicitly (e.g. `-c 65536`); with mirror the CPU KV cache is
  duplicated per node, so a maxed context wastes mirrored RAM on every node.

## Hybrid: `ggml_cuda_init: failed to initialize CUDA` (then 0 layers on GPU)

`cudaGetDeviceCount()` returned an error, so ggml disables the ENTIRE CUDA backend and
everything falls to CPU. Build and flags are fine; this is a runtime/driver issue.
`nvidia-smi` working does NOT rule it out (it only needs the base driver). **Read the
string after the colon — it is dispositive:**

| Error string | Cause | Fix |
|---|---|---|
| `initialization error` / `unknown error` / `OS call failed` | `nvidia_uvm` not loaded / `/dev/nvidia-uvm` missing (common on fresh headless boot) | `ls -l /dev/nvidia-uvm`; `sudo nvidia-modprobe -u -c=0`; persist via boot unit + `nvidia-persistenced` |
| `CUDA driver version is insufficient for CUDA runtime version` | driver older than the built toolkit | compare `nvidia-smi` "CUDA Version: X.Y" vs build toolkit; rebuild against ≤ that, or update driver |
| `no CUDA-capable device is detected` | empty `CUDA_VISIBLE_DEVICES`, or a container without GPU passthrough | set CUDA_VISIBLE_DEVICES; if dockerized, run with `--gpus all` + nvidia-container-toolkit |
| `forward compatibility ... non supported HW` | driver/compat mismatch for the Blackwells | align driver + CUDA compat package |

Always: **isolate the Ada** for llama (mixed sm_120 Blackwell + sm_89 Ada enumeration
can abort init if a Blackwell is new-silicon-flaky or held by vLLM). Use the UUID, not
the index — index ordering isn't stable across the two architectures:

```sh
nvidia-smi -L                                  # note the Ada 6000's UUID
CUDA_VISIBLE_DEVICES=GPU-<ada-uuid> ./build-cuda/bin/llama-cli -ngl 99 -ot exps=CPU -fa on ...
```

This is also the correct production setup — leaves the 4 Blackwells for vLLM.

### If NO reason string prints after "failed to initialize CUDA" (bare-metal, from source)

The code always appends `: <reason>`, so a blank reason means the runtime never reached
a real driver. `nvidia-smi` working does not cover this. Check in order:

```sh
# A. nvidia_uvm module + device nodes (nvidia-smi does NOT load uvm; #1 on fresh boots)
lsmod | grep nvidia_uvm ; ls -l /dev/nvidia-uvm
sudo nvidia-modprobe -u -c=0            # creates the uvm nodes, then retry

# B. stub libcuda.so trap (the "built CUDA from source" gotcha)
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -i stubs   # must print NOTHING
ldd ./build-cuda/bin/llama-cli | grep -i cuda           # must resolve to /usr/lib64, not a stubs/ dir
# (the toolkit's stubs/libcuda.so is link-time only; on the runtime path it has no driver behind it)

# C. driver vs toolkit (Blackwells need >=570-series)
nvidia-smi | grep "CUDA Version" ; nvcc --version        # driver max-CUDA must be >= build toolkit

# D. kernel's own view (Blackwell adapter-init failures)
dmesg | grep -iE 'nvrm|nvidia' | tail -20
```

**Decisive step — the standalone probe** (`scripts/cuda-probe.cu`) calls
`cudaGetDeviceCount` with no llama logging in the way, so you get the exact code even
when llama prints a blank/terse reason:

```sh
nvcc -o /tmp/cuda-probe scripts/cuda-probe.cu && /tmp/cuda-probe
ldd ./build-cuda/bin/llama-cli | grep -i cuda   # must be /usr/lib64, NOT a .../stubs/ dir
```

| Probe output | Meaning → action |
|---|---|
| `DRIVER TOO OLD` / `err=35 insufficient driver` | driver older than build toolkit → rebuild vs matching toolkit or update driver |
| `err=999 unknown error` | driver bad state — usually module↔libcuda.so mismatch (**reboot**) or an Xid; `dmesg \| grep -iE 'nvrm\|xid'` |
| `err=100 no CUDA-capable device` | bad `CUDA_VISIBLE_DEVICES`, or a stub `libcuda.so` on the path |
| `err=0 count=5`, all list | CUDA is fine standalone → llama-specific: stub lib on its `LD_LIBRARY_PATH`, or mixed-arch enum → run with `CUDA_VISIBLE_DEVICES=GPU-<ada-uuid>` |
| `err=0` but Ada missing / <5 devices | that GPU won't enumerate → `dmesg` for its adapter-init failure |

Context note: this box runs a **desktop** (not headless), so the driver stack is
healthy (X renders on it) — that lowers the odds of the nvidia_uvm case (A) and raises
the stub-lib (B) and driver module↔library mismatch (reboot) cases.

CPU-only serving is unaffected by all of this — hybrid is an optional speedup on top.

## Notes

- Do NOT enable `--dsa` (GLM sparse attention) on this build — a post-fork upstream fix
  (`96472464`, wrong RoPE type under DSA) isn't in our tree; without it DSA corrupts
  tokens. Default is off; leave it off until that's cherry-picked.
- The earlier lesson generalizes: degenerate/looping text on an instruct MoE is a
  template problem until proven otherwise — the compute path was byte-identity-tested.
- Keep the `--numa mirror` + tuned `-t` + auto waiting-policy flags exactly as tuned;
  none of this changes the NUMA setup.
