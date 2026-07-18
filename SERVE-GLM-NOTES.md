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

## Notes

- Do NOT enable `--dsa` (GLM sparse attention) on this build — a post-fork upstream fix
  (`96472464`, wrong RoPE type under DSA) isn't in our tree; without it DSA corrupts
  tokens. Default is off; leave it off until that's cherry-picked.
- The earlier lesson generalizes: degenerate/looping text on an instruct MoE is a
  template problem until proven otherwise — the compute path was byte-identity-tested.
- Keep the `--numa mirror` + tuned `-t` + auto waiting-policy flags exactly as tuned;
  none of this changes the NUMA setup.
