#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Keep this ablation in a separate checkpoint/log directory by default.
export PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-justrl_qwen3_1p7b_js_top5_kl}
export JS_TOKEN_FILTER_ENABLED=${JS_TOKEN_FILTER_ENABLED:-True}
export JS_TOP_FRACTION=${JS_TOP_FRACTION:-0.05}
export USE_KL_LOSS=${USE_KL_LOSS:-True}
export ADAPTIVE_KL_LOSS_COEF=${ADAPTIVE_KL_LOSS_COEF:-True}

# Use more of each 80GB GPU for vLLM KV cache and prefill batching.
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}

exec bash "${SCRIPT_DIR}/train_justrl_qwen.sh"
