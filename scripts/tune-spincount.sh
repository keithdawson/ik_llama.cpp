#!/usr/bin/env bash
# Tune GOMP_SPINCOUNT for NUMA-mirror + GPU hybrid runs on this machine.
#
# Background (docs/numa-tuning.md "Waiting policy"): with --numa mirror and GPU offload,
# pinned OpenMP workers must briefly spin then sleep while the GPU runs. The built-in
# default (OMP_WAIT_POLICY=PASSIVE, GOMP_SPINCOUNT=25000) was tuned on a desktop Zen 5;
# this script finds the best value for the machine it runs on and prints what to set.
#
# Usage (run on the target machine, e.g. Pandora):
#   ./scripts/tune-spincount.sh -m /path/model.gguf [-b build/bin/llama-cli] \
#       [-t 94] [-r 3] [-n 64] [-- <extra llama-cli args, e.g. -ot exps=CPU -ngl 99>]
#
# Defaults assume the hybrid shape: -ngl 99 -ot exps=CPU -fa on --numa mirror.
# Pass your real serving flags after `--` to tune against the actual workload shape.
set -euo pipefail

BIN=build/bin/llama-cli
MODEL=""
THREADS=""
REPS=3
NPRED=64
SPINS=${SPINS:-"0 5000 10000 25000 50000 100000 250000"}

while getopts "b:m:t:r:n:h" opt; do
    case $opt in
        b) BIN=$OPTARG ;;
        m) MODEL=$OPTARG ;;
        t) THREADS=$OPTARG ;;
        r) REPS=$OPTARG ;;
        n) NPRED=$OPTARG ;;
        h) grep '^#' "$0" | head -16; exit 0 ;;
        *) exit 1 ;;
    esac
done
shift $((OPTIND - 1))
EXTRA=("$@")
if [ ${#EXTRA[@]} -eq 0 ]; then
    EXTRA=(-ngl 99 -ot exps=CPU -fa on)
fi
[ -n "$MODEL" ] || { echo "error: -m <model.gguf> required" >&2; exit 1; }
[ -x "$BIN" ] || { echo "error: $BIN not executable" >&2; exit 1; }

TARGS=()
[ -n "$THREADS" ] && TARGS=(-t "$THREADS")

# ~3k-token prompt so prompt processing is actually measured
PROMPT=$(mktemp)
trap 'rm -f "$PROMPT"' EXIT
for i in $(seq 1 160); do
    printf 'The quick brown fox jumps over the lazy dog while the industrious ant carries provisions across the meadow toward winter storage. ' >> "$PROMPT"
done

run_once() { # $1 = spincount ; prints "pp tg"
    # position-independent parse (value before "tokens per second"): some models append
    # extra fields to the eval line (e.g. GLM MTP stats). Raw lines are kept in
    # tune-spincount-timings.log for offline diagnosis if a column reads 0.
    OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=$1 \
    "$BIN" -m "$MODEL" --numa mirror "${TARGS[@]}" "${EXTRA[@]}" \
        -c 4096 -n "$NPRED" --temp 0 --seed 1 --no-display-prompt -f "$PROMPT" 2>&1 |
    awk -v raw="tune-spincount-timings.log" '
        /llama_print_timings/ {
            print >> raw
            if (/prompt eval time/) {
                for (i = 2; i <= NF; i++) if ($i == "tokens" && $(i+1) == "per") pp = $(i-1)
            } else if (/ eval time/) {
                for (i = 2; i <= NF; i++) if ($i == "tokens" && $(i+1) == "per") tg = $(i-1)
            }
        }
        END { print pp+0, tg+0 }'
}

echo "spincount sweep: $SPINS  (reps=$REPS, bin=$BIN)"
echo "warmup..." >&2
run_once 25000 > /dev/null

printf '%-10s %12s %12s\n' "spincount" "pp t/s" "tg t/s"
best_spin=""; best_tg=0
for s in $SPINS; do
    pp_sum=0; tg_sum=0
    for r in $(seq 1 "$REPS"); do
        read -r pp tg < <(run_once "$s")
        pp_sum=$(awk "BEGIN{print $pp_sum+$pp}")
        tg_sum=$(awk "BEGIN{print $tg_sum+$tg}")
    done
    pp_avg=$(awk "BEGIN{printf \"%.1f\", $pp_sum/$REPS}")
    tg_avg=$(awk "BEGIN{printf \"%.2f\", $tg_sum/$REPS}")
    printf '%-10s %12s %12s\n' "$s" "$pp_avg" "$tg_avg"
    if awk "BEGIN{exit !($tg_avg > $best_tg)}"; then best_tg=$tg_avg; best_spin=$s; fi
done

echo
echo "best tg: GOMP_SPINCOUNT=$best_spin ($best_tg t/s)"
echo "-> if pp at that value is also within noise of the best pp row, adopt it:"
echo "   export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=$best_spin"
echo "   (see docs/numa-tuning.md 'Waiting policy' for where to persist this)"
