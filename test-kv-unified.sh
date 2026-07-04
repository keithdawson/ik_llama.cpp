#!/bin/bash
# Functional test for --kv-unified (elastic slot context). Run from the build tree root.
# Not part of the test suite; used during development on the kv-unified branch.
set -u
BIN=${BIN:-./build-linux/bin/llama-server}
MODEL=${MODEL:-$HOME/models/stories260K.gguf}
PORT=8199

PROMPT=$(python3 -c 'print("Once upon a time there was a little girl named Lily. " * 32)')

start_server() {
    $BIN -m "$MODEL" -c 512 -np 2 -t 2 --port $PORT $1 > /tmp/srv-$TAG.log 2>&1 &
    SPID=$!
    for i in $(seq 1 60); do
        curl -s localhost:$PORT/health > /dev/null 2>&1 && return 0
        sleep 0.3
    done
    echo "server failed to start"; cat /tmp/srv-$TAG.log | tail -5; return 1
}

stop_server() { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; }

query() {
    curl -s localhost:$PORT/completion -d "{\"prompt\": $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$1"), \"n_predict\": ${2:-8}}"
}

summarize() {
    python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print("  tokens_evaluated:", d.get("tokens_evaluated"), " truncated:", d.get("truncated"), " stop:", d.get("stopped_limit") or d.get("stop_type", "?"))
except Exception as e:
    print("  BAD RESPONSE:", e)
'
}

TAG=1
echo "=== test 1: default mode (fixed 256/slot quota) ==="
start_server "" || exit 1
query "$PROMPT" | summarize
stop_server

TAG=2
echo "=== test 2: --kv-unified (slot may use whole 512 pool) ==="
start_server "--kv-unified" || exit 1
query "$PROMPT" | summarize
stop_server

varied() { # varied() SEED NWORDS -> non-repetitive prompt (repetitive ones no-op the shift matcher)
    python3 -c "
import random; random.seed($1)
words = 'cat dog tree house sun moon river stone bird fish apple cake road cloud star grass hill lamp door book'.split()
print(' '.join(random.choice(words) for _ in range($2)))"
}

TAG=3
echo "=== test 3: --kv-unified idle-cache eviction (A done, B needs A's cells) ==="
start_server "--kv-unified" || exit 1
query "$(varied 1 220)" 8 > /tmp/respA.json          # A completes, idles with ~230 cached cells
echo "-- response A:"; cat /tmp/respA.json | summarize
query "$(varied 2 220)" 8 > /tmp/respB.json          # B needs ~230 of the 512-cell pool
echo "-- response B:"; cat /tmp/respB.json | summarize
echo "-- evictions:"; grep -c "evicting idle slot" /tmp/srv-$TAG.log
stop_server

TAG=4
echo "=== test 4: --kv-unified active shift (B arrives while A still generating) ==="
start_server "--kv-unified" || exit 1
query "$(varied 3 200)" 400 > /tmp/respA.json &      # A keeps generating past B's arrival
APID=$!
sleep 1
query "$(varied 4 200)" 8 > /tmp/respB.json
wait $APID
echo "-- response A:"; cat /tmp/respA.json | summarize
echo "-- response B:"; cat /tmp/respB.json | summarize
echo "-- shifts:";    grep -c "slot context shift" /tmp/srv-$TAG.log
echo "-- evictions:"; grep -c "evicting idle slot" /tmp/srv-$TAG.log
echo "-- errors:";    grep -c '"code":500' /tmp/respA.json /tmp/respB.json
stop_server
echo "done"
