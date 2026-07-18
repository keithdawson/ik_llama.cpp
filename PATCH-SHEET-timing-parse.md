# Hand-patch sheet: fix tg=0 in the tuning scripts (print & carry)

**Why:** the sweep scripts parsed throughput as the 4th-from-last field of the
`llama_print_timings` lines. GLM appends extra fields to the *eval* line, so the tg
column reads 0 (pp was unaffected — those numbers are good). The fix parses
position-independently: take the number right before the words `tokens per`.

Two files, same idea. Make a backup first:

```sh
cp scripts/pandora-tune.sh scripts/pandora-tune.sh.bak
cp scripts/tune-spincount.sh scripts/tune-spincount.sh.bak
```

---

## Patch 1 of 2 — `scripts/pandora-tune.sh`, inside `run_bench()`

FIND this awk program (the only place `NF-3` appears in the file):

```
            awk '/llama_print_timings: prompt eval time/ { pp=$(NF-3) }
                 /llama_print_timings: *eval time/       { tg=$(NF-3) }
                 END { print pp+0, tg+0 }')
```

REPLACE those 3 lines with (keep the trailing `)` — it closes the `< <(...)`):

```
            awk '
                /llama_print_timings/ && /prompt eval time/ {
                    for (i = 2; i <= NF; i++) if ($i == "tokens" && $(i+1) == "per") pp = $(i-1)
                }
                /llama_print_timings/ && !/prompt eval/ && / eval time/ {
                    for (i = 2; i <= NF; i++) if ($i == "tokens" && $(i+1) == "per") tg = $(i-1)
                }
                END { print pp+0, tg+0 }')
```

---

## Patch 2 of 2 — `scripts/tune-spincount.sh`, inside `run_once()`

FIND:

```
    awk '/llama_print_timings: prompt eval time/ { pp=$(NF-3) }
         /llama_print_timings: *eval time/       { tg=$(NF-3) }
         END { print pp+0, tg+0 }'
```

REPLACE with (note: NO trailing `)` here, unlike patch 1):

```
    awk '
        /llama_print_timings/ && /prompt eval time/ {
            for (i = 2; i <= NF; i++) if ($i == "tokens" && $(i+1) == "per") pp = $(i-1)
        }
        /llama_print_timings/ && !/prompt eval/ && / eval time/ {
            for (i = 2; i <= NF; i++) if ($i == "tokens" && $(i+1) == "per") tg = $(i-1)
        }
        END { print pp+0, tg+0 }'
```

---

## Verify (instant, no model load needed)

```sh
bash -n scripts/pandora-tune.sh && bash -n scripts/tune-spincount.sh && echo SYNTAX_OK
```

Then feed the parser fake timing lines, including a GLM-style suffixed eval line:

```sh
printf '%s\n%s\n' \
 'llama_print_timings: prompt eval time = 9 ms / 3000 tokens ( 0.4 ms per token, 2430.11 tokens per second)' \
 'llama_print_timings:  eval time = 9 ms / 128 runs ( 81.2 ms per token, 12.30 tokens per second) extra glm stuff' |
awk '
    /llama_print_timings/ && /prompt eval time/ {
        for (i = 2; i <= NF; i++) if ($i == "tokens" && $(i+1) == "per") pp = $(i-1)
    }
    /llama_print_timings/ && !/prompt eval/ && / eval time/ {
        for (i = 2; i <= NF; i++) if ($i == "tokens" && $(i+1) == "per") tg = $(i-1)
    }
    END { print pp+0, tg+0 }'
```

Expected output: `2430.11 12.3` — if you see that, the patch is right; rerun any sweep
where you need the tg column (barrier-gate and pin are the tg-sensitive ones; your pp
numbers from the broken runs were parsed correctly and stand).

Notes:
- The git version of this fix additionally logs every run's raw timing lines to
  `pandora-tune-results/last-timings.log`; the hand-patched copy just lacks that
  nicety until the next full source carry.
- Census interpretation correction that came with this fix: `max_over_mean` is
  informational only (gauges the hypothetical expert-sharding alternative; ~1.0
  uniform, ≤2.5 typical). Your 1.74 is fine — no action, mirror unaffected.
