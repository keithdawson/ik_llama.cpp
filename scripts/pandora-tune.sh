#!/usr/bin/env bash
# Validation & tuning suite for the dual-socket target ("Pandora").
# Sweeps every env knob the fake-NUMA testbed shipped, on the REAL machine, and prints
# for each one the winning value and exactly where to apply it. Companion doc:
# docs/numa-tuning.md "Pandora validation suite" (incl. expected runtimes for ~500 GB
# models — each rep reloads the model; after the first run it comes from page cache).
#
# Usage:
#   ./scripts/pandora-tune.sh <subcommand> -m /path/model.gguf [options] [-- <extra llama-cli args>]
#
# Subcommands (CPU-only unless marked hybrid):
#   census        one instrumented run: expert-routing histogram + mirror sanity checks
#   spincount     GOMP_SPINCOUNT sweep (hybrid; wraps scripts/tune-spincount.sh)
#   barrier-gate  GGML_NUMA_HIER_BATCH_MAX sweep      (tg + pp)
#   copy          GGML_NUMA_COPY_THREADS x NT_COPY    (mirror populate GB/s)
#   pin           GGML_NUMA_PIN node vs cpu           (tg + pp; CCD/L3 locality)
#   threads       coarse -t sweep                     (tg + pp)
#   reserve       GGML_NUMA_RESERVE_CPUS sweep        (hybrid: pass your -ngl/-ot flags)
#   bind-compute  --numa-bind-compute on/off          (hybrid)
#   hugetlb       GGML_NUMA_HUGETLB on/off            (optional; needs vm.nr_hugepages, see -h)
#   verify        single run with your current env; prints pp/tg + placement sanity
#   all           census, barrier-gate, copy, pin, threads (+hybrid subs if -ngl in extra args)
#
# Options:
#   -m FILE   model gguf (required)
#   -b PATH   llama-cli binary            (default build/bin/llama-cli)
#   -t N      thread count for non-thread sweeps (default: nproc minus 2)
#   -r N      reps per value              (default 2; first-ever run also warms page cache)
#   -g N      GPU NUMA node for hybrid subs (default 0 - the Blackwells' socket)
#   -n N      tokens to generate          (default 128)
#
# Every sweep honors your ambient environment; values under test are set per-run.
# Do NOT run this under numactl --cpunodebind: mirror mode re-pins and escapes it.
set -euo pipefail

BIN=build/bin/llama-cli
MODEL="" ; THREADS="" ; REPS=2 ; GPUNODE=0 ; NPRED=128
CMD=${1:-} ; shift || true
while getopts "b:m:t:r:g:n:h" opt; do
    case $opt in
        b) BIN=$OPTARG ;;
        m) MODEL=$OPTARG ;;
        t) THREADS=$OPTARG ;;
        r) REPS=$OPTARG ;;
        g) GPUNODE=$OPTARG ;;
        n) NPRED=$OPTARG ;;
        h) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) exit 1 ;;
    esac
done
shift $((OPTIND - 1))
EXTRA=("$@")
[ -n "$CMD" ]   || { echo "usage: $0 <subcommand> -m model.gguf [-h for help]" >&2; exit 1; }
[ -n "$MODEL" ] || { echo "error: -m <model.gguf> required" >&2; exit 1; }
[ -x "$BIN" ]   || { echo "error: $BIN not executable" >&2; exit 1; }
[ -n "$THREADS" ] || THREADS=$(( $(nproc) - 2 ))

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RESULTS=${RESULTS:-pandora-tune-results}
mkdir -p "$RESULTS"

# ~3k-token prompt so pp is measured for real
PROMPT=$(mktemp)
trap 'rm -f "$PROMPT"' EXIT
for i in $(seq 1 160); do
    printf 'The quick brown fox jumps over the lazy dog while the industrious ant carries provisions across the meadow toward winter storage. ' >> "$PROMPT"
done

is_hybrid=0
case " ${EXTRA[*]:-} " in *" -ngl "*) is_hybrid=1 ;; esac

# run_bench "ENV1=v ENV2=v" [extra args...]  -> echoes "pp tg" (averaged over REPS)
run_bench() {
    local envs=$1; shift
    local pp_sum=0 tg_sum=0 pp tg r
    for r in $(seq 1 "$REPS"); do
        read -r pp tg < <(env $envs "$BIN" -m "$MODEL" -t "$THREADS" "${EXTRA[@]}" "$@" \
                -c 4096 -n "$NPRED" --temp 0 --seed 1 --no-display-prompt -f "$PROMPT" 2>&1 |
            awk '/llama_print_timings: prompt eval time/ { pp=$(NF-3) }
                 /llama_print_timings: *eval time/       { tg=$(NF-3) }
                 END { print pp+0, tg+0 }')
        pp_sum=$(awk "BEGIN{print $pp_sum+$pp}"); tg_sum=$(awk "BEGIN{print $tg_sum+$tg}")
    done
    awk "BEGIN{printf \"%.1f %.2f\", $pp_sum/$REPS, $tg_sum/$REPS}"
}

# run one load-only instrumented run; echoes the full numa_stats block
run_stats() {
    local envs=$1; shift
    env $envs GGML_NUMA_STATS=1 "$BIN" -m "$MODEL" -t "$THREADS" "${EXTRA[@]}" "$@" \
        -c 4096 -n 1 --temp 0 -p Hi 2>&1 | grep -a '^numa_stats:'
}

table_header() { printf '%-24s %12s %12s\n' "$1" "pp t/s" "tg t/s"; }
warmup() {
    echo "warmup run (first-ever load also pulls the model into page cache)..." >&2
    run_bench "" --numa mirror > /dev/null
}

pick_best() { # stdin: "label pp tg" lines -> prints best-tg label (pp used as tiebreak)
    sort -k3,3gr -k2,2gr | head -1 | awk '{print $1}'
}

sub_census() {
    echo "== census: one instrumented mirror run =="
    local out
    out=$(GGML_NUMA_STATS=1 "$BIN" -m "$MODEL" -t "$THREADS" "${EXTRA[@]}" --numa mirror \
            -c 4096 -n 32 --temp 0 --seed 1 --no-display-prompt -f "$PROMPT" 2>&1 |
          grep -aE '^numa_stats:|duplicated .* weight tensors|mirrored .* KV' || true)
    echo "$out" | grep -av 'moe_expert_rows' || true
    echo "$out" | grep -a 'moe_expert_rows' > "$RESULTS/census-expert-rows.csv" || true
    echo
    echo "==> RESULT census:"
    echo "  - REQUIRED: a 'duplicated N weight tensors' line must appear above. If it does not,"
    echo "    the mirror free-RAM check failed (MemAvailable < weights size + 2 GB) and you are"
    echo "    running UNMIRRORED - fix memory pressure before tuning anything else."
    echo "  - resolve_fallback must be 0 on every node."
    echo "  - max_over_mean < ~1.5 on moe stats means expert routing is near-uniform (mirroring"
    echo "    is the right architecture; sharding would only save RAM, not time)."
    echo "  - full per-expert histogram saved to $RESULTS/census-expert-rows.csv"
}

sub_barrier_gate() {
    echo "== barrier-gate: GGML_NUMA_HIER_BATCH_MAX sweep (tg wants hier; big-batch pp may not) =="
    warmup
    table_header "hier_batch_max"
    local best; best=$(for v in ${GATE_LIST:- -1 16 32 128 512}; do
        read -r pp tg < <(run_bench "GGML_NUMA_HIER_BATCH_MAX=$v" --numa mirror)
        printf '%-24s %12s %12s\n' "$v" "$pp" "$tg" >&2
        echo "$v $pp $tg"
    done | pick_best)
    echo
    echo "==> RESULT barrier-gate: GGML_NUMA_HIER_BATCH_MAX=$best"
    echo "  APPLY: only if != 32 (the built-in default): add GGML_NUMA_HIER_BATCH_MAX=$best to the"
    echo "  serving environment (systemd Environment= / docker compose environment: / wrapper export)."
}

sub_copy() {
    echo "== copy: GGML_NUMA_COPY_THREADS x GGML_NUMA_NT_COPY (mirror populate bandwidth) =="
    echo "each run is load-only (-n 1); GB/s = populate_bytes / populate_us"
    printf '%-24s %12s\n' "threads,nt" "GB/s"
    local best="" best_gbps=0
    for nt in ${NT_LIST:-0 1}; do
        for th in ${COPY_THREADS_LIST:-8 16 24 32 48}; do
            local sum=0 r
            for r in $(seq 1 "$REPS"); do
                local line gbps
                line=$(run_stats "GGML_NUMA_COPY_THREADS=$th GGML_NUMA_NT_COPY=$nt" --numa mirror | grep -a populate_bytes)
                gbps=$(echo "$line" | awk -F'[= ]' '{b=0;u=1} {for(i=1;i<=NF;i++){if($i=="populate_bytes")b=$(i+1); if($i=="populate_us")u=$(i+1)}} END{printf "%.2f", b/u/1000}')
                sum=$(awk "BEGIN{print $sum+$gbps}")
            done
            local avg; avg=$(awk "BEGIN{printf \"%.2f\", $sum/$REPS}")
            printf '%-24s %12s\n' "$th,nt=$nt" "$avg"
            if awk "BEGIN{exit !($avg > $best_gbps)}"; then best_gbps=$avg; best="threads=$th nt=$nt"; fi
        done
    done
    echo
    echo "==> RESULT copy: best populate = $best_gbps GB/s at $best"
    echo "  APPLY: if best threads != 16, set GGML_NUMA_COPY_THREADS=<n>; if nt=1 won,"
    echo "  set GGML_NUMA_NT_COPY=1. Both in the serving environment. This affects model-load"
    echo "  mirror population and KV resyncs (K-shift/state-restore), not steady-state tg."
}

sub_pin() {
    echo "== pin: GGML_NUMA_PIN node vs cpu (per-CPU pinning stops cross-CCD L3 migration) =="
    warmup
    table_header "pin"
    local best; best=$(for v in node cpu; do
        local envs=""
        [ "$v" = cpu ] && envs="GGML_NUMA_PIN=cpu"
        read -r pp tg < <(run_bench "$envs" --numa mirror)
        printf '%-24s %12s %12s\n' "$v" "$pp" "$tg" >&2
        echo "$v $pp $tg"
    done | pick_best)
    echo
    echo "==> RESULT pin: $best"
    echo "  APPLY: if 'cpu' won, add GGML_NUMA_PIN=cpu to the serving environment."
}

sub_threads() {
    echo "== threads: coarse -t sweep (mirror splits threads across both sockets) =="
    warmup
    table_header "-t"
    local best; best=$(for v in ${THREADS_LIST:-96 128 160 176 192}; do
        read -r pp tg < <(THREADS=$v run_bench "" --numa mirror -t "$v")
        printf '%-24s %12s %12s\n' "$v" "$pp" "$tg" >&2
        echo "$v $pp $tg"
    done | pick_best)
    echo
    echo "==> RESULT threads: -t $best"
    echo "  APPLY: as the -t argument of llama-server/llama-cli. Keep it even so both sockets"
    echo "  get equal blocks (the split is CPU-count weighted if you also reserve CPUs)."
}

sub_reserve() {
    [ "$is_hybrid" = 1 ] || { echo "reserve is a hybrid test: pass your -ngl/-ot flags after --" >&2; exit 1; }
    echo "== reserve: GGML_NUMA_RESERVE_CPUS on the GPU node (cores for the CUDA driver) =="
    warmup
    table_header "reserve"
    local best; best=$(for v in ${RESERVE_LIST:-0 1 2 4}; do
        local envs=""
        [ "$v" != 0 ] && envs="GGML_NUMA_RESERVE_CPUS=$v@$GPUNODE"
        read -r pp tg < <(run_bench "$envs" --numa mirror --numa-gpu-node "$GPUNODE")
        printf '%-24s %12s %12s\n' "$v" "$pp" "$tg" >&2
        echo "$v $pp $tg"
    done | pick_best)
    echo
    echo "==> RESULT reserve: GGML_NUMA_RESERVE_CPUS=$best@$GPUNODE"
    echo "  APPLY: only if != 0: add it to the serving environment. Reserved CPUs are excluded"
    echo "  from compute pinning; the thread split rebalances automatically."
}

sub_bind_compute() {
    [ "$is_hybrid" = 1 ] || { echo "bind-compute is a hybrid test: pass your -ngl/-ot flags after --" >&2; exit 1; }
    echo "== bind-compute: --numa-bind-compute off vs on (sched CPU buffers on the GPU socket) =="
    warmup
    table_header "bind-compute"
    local best; best=$(for v in off on; do
        local flag=()
        [ "$v" = on ] && flag=(--numa-bind-compute)
        read -r pp tg < <(run_bench "" --numa mirror --numa-gpu-node "$GPUNODE" "${flag[@]}")
        printf '%-24s %12s %12s\n' "$v" "$pp" "$tg" >&2
        echo "$v $pp $tg"
    done | pick_best)
    echo
    echo "==> RESULT bind-compute: $best"
    echo "  APPLY: if 'on' won, add --numa-bind-compute to the llama-server/llama-cli command line."
}

sub_hugetlb() {
    echo "== hugetlb: GGML_NUMA_HUGETLB off vs on =="
    echo "NOTE: needs a preallocated pool >= mirror size (weights + KV copies):"
    echo "  sysctl -w vm.nr_hugepages=<bytes/2MiB>   # ~256000 for a 500 GB mirror"
    echo "The pool SHRINKS MemAvailable: verify the census 'duplicated N weight tensors' line"
    echo "still appears, or the mirror silently skips. Testbed verdict was tg -6.5%; this is"
    echo "only worth trying on a long-uptime box where THP has fragmented."
    warmup
    table_header "hugetlb"
    for v in 0 1; do
        local envs=""
        [ "$v" = 1 ] && envs="GGML_NUMA_HUGETLB=1"
        read -r pp tg < <(run_bench "$envs" --numa mirror)
        printf '%-24s %12s %12s\n' "$v" "$pp" "$tg"
    done
    echo
    echo "==> RESULT hugetlb: adopt GGML_NUMA_HUGETLB=1 only if it won BOTH columns; remember to"
    echo "  persist vm.nr_hugepages via sysctl.d if so, else reset it to 0."
}

sub_verify() {
    echo "== verify: one run with your current environment + flags =="
    read -r pp tg < <(run_bench "" --numa mirror)
    echo "pp=$pp t/s   tg=$tg t/s"
    echo "placement sanity (run in another shell while a server is up):"
    echo "  numastat -p \$(pgrep llama-server)   # both nodes should hold ~= the model size"
}

case $CMD in
    census)       sub_census ;;
    spincount)    exec "$SCRIPT_DIR/tune-spincount.sh" -b "$BIN" -m "$MODEL" -t "$THREADS" -r "$REPS" -n "$NPRED" -- "${EXTRA[@]}" ;;
    barrier-gate) sub_barrier_gate ;;
    copy)         sub_copy ;;
    pin)          sub_pin ;;
    threads)      sub_threads ;;
    reserve)      sub_reserve ;;
    bind-compute) sub_bind_compute ;;
    hugetlb)      sub_hugetlb ;;
    verify)       sub_verify ;;
    all)
        sub_census;       echo
        sub_barrier_gate; echo
        sub_copy;         echo
        sub_pin;          echo
        sub_threads;      echo
        if [ "$is_hybrid" = 1 ]; then
            sub_reserve;      echo
            sub_bind_compute; echo
            echo "run 'spincount' separately (it has its own sweep script)."
        fi
        ;;
    *) echo "unknown subcommand: $CMD (use -h)" >&2; exit 1 ;;
esac
