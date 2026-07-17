#!/usr/bin/env bash
#
# quickbench.sh — fast screening pass for a new model candidate: a fresh
# unsloth/NVFP4 quant, someone else's trimmed model, or anything you want a
# quick read on before committing to a full eval.
#
# Runs tier3 (sanity) + a 1-repeat eval (all 10 domains) + a single-gpu_util
# capacity pass (orchestrator / coding_agent / chat_agent), then refreshes
# results/LEADERBOARD.md.
#
# The candidate model's own OpenAI-compatible server must already be serving
# at --endpoint before you run this — quickbench doesn't start it for you
# (models, quant formats, and serving flags vary too much to assume one
# vllm-serve invocation fits everyone's candidate). If you're also running
# another resident model that needs the memory freed up, stop/start that
# service yourself around this call.
#
# Usage:
#   ./quickbench.sh <label> <model> <endpoint> [gpu_util]
#
# Example — screening a new unsloth NVFP4 quant:
#   ./quickbench.sh qwen3-nvfp4-v2 unsloth/Qwen3-32B-NVFP4 \
#       http://localhost:8001/v1 0.85
#
# Example — someone benchmarking their own model:
#   ./quickbench.sh my-model my-org/my-model http://localhost:8000/v1
#
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $(basename "$0") <label> <model> <endpoint> [gpu_util]" >&2
  exit 1
fi

LABEL="$1"
MODEL="$2"
ENDPOINT="$3"
GPU_UTIL="${4:-0.85}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "=== [1/3] tier3 sanity (coding / tool-call / long-context) — $LABEL ==="
./spark_bench_plus.py tier3 --label "$LABEL" --model "$MODEL" --endpoint "$ENDPOINT"

echo "=== [2/3] eval (1 repeat, full 10-domain suite) — $LABEL ==="
./spark_bench_plus.py eval --label "$LABEL" --model "$MODEL" --endpoint "$ENDPOINT" --repeats 1

echo "=== [3/3] capacity (gpu_util=$GPU_UTIL, orchestrator/coding_agent/chat_agent) — $LABEL ==="
./spark_bench_plus.py capacity --label "$LABEL" --model "$MODEL" --endpoint "$ENDPOINT" \
  --gpu-utils "$GPU_UTIL" --skip-restart \
  --profiles orchestrator,coding_agent,chat_agent \
  --concurrency 1,2,4,8,16

echo "=== refreshing leaderboard ==="
./leaderboard.py --write

echo "[quickbench] done — see results/LEADERBOARD.md"
