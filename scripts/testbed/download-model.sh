#!/usr/bin/env bash
# Download the testbed models into /models (the ik-models named volume).
# Runs inside the ik-numa-testbed container:
#   .\scripts\testbed\run-testbed.ps1 download-model
#
# Override the defaults with env vars if the repo id / quant path differs:
#   HF_REPO / HF_FILE      the ~15 GB MoE model used for real A/B runs
#   SMOKE_REPO/SMOKE_FILE  small model for fast functional smoke tests
set -euo pipefail

MODELS_DIR=${MODELS_DIR:-/models}

# unsloth dynamic-quant GGUF; adjust HF_FILE if the quant sits in a subfolder
# (e.g. "UD-IQ4_XS/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf")
HF_REPO=${HF_REPO:-unsloth/gemma-4-26B-A4B-it-GGUF}
HF_FILE=${HF_FILE:-gemma-4-26B-A4B-it-UD-IQ4_XS.gguf}

SMOKE_REPO=${SMOKE_REPO:-Qwen/Qwen2.5-0.5B-Instruct-GGUF}
SMOKE_FILE=${SMOKE_FILE:-qwen2.5-0.5b-instruct-q4_k_m.gguf}

if command -v hf >/dev/null 2>&1; then
    DL="hf download"
else
    DL="huggingface-cli download"
fi

echo "==> smoke model: $SMOKE_REPO :: $SMOKE_FILE"
$DL "$SMOKE_REPO" "$SMOKE_FILE" --local-dir "$MODELS_DIR"

echo "==> main model: $HF_REPO :: $HF_FILE (~15 GB, be patient)"
$DL "$HF_REPO" "$HF_FILE" --local-dir "$MODELS_DIR"

echo "==> done:"
ls -lh "$MODELS_DIR"
