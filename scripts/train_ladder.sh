#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Usage:
#   bash scripts/train_ladder.sh 20_30
#   JS_LADDER_SPLIT=20_30 bash scripts/train_ladder.sh
#
# The positional argument takes precedence over JS_LADDER_SPLIT. When neither
# is provided, the script defaults to 90_100 (the original top-10% setting).
if [ "$#" -gt 1 ]; then
  echo "Usage: bash scripts/train_ladder.sh [START_END]" >&2
  exit 2
fi
LADDER_SPLIT=${1:-${JS_LADDER_SPLIT:-90_100}}

# Before running on another machine, uncomment and fill in these paths:
# export ACTOR_MODEL_PATH="/path/to/Qwen3-1.7B"
# export REWARD_MODEL_PATH="/path/to/JustRL-DeepSeek-1.5B"
# export TEACHER_REF_MODEL_PATH="/path/to/DeepSeek-R1-Distill-Qwen-1.5B"
# export TRAIN_DATASET="/path/to/skywork-or1-math-dapo-original.parquet"
# export TEST_DATASET="['/path/to/aime24.parquet','/path/to/aime25.parquet']"
# export OUTPUT_ROOT="/path/to/checkpoints"
# export LOG_ROOT="/path/to/logs"

# Keep every ladder interval in a separate checkpoint/log directory by default.
export PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NUM_NODES=${NUM_NODES:-1}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-justrl_qwen3_1p7b_js_ladder_${LADDER_SPLIT}_kl}
export JS_TOKEN_FILTER_ENABLED=${JS_TOKEN_FILTER_ENABLED:-True}
export JS_TOP_FRACTION=${JS_TOP_FRACTION:-0.10}
export JS_TOKEN_SELECTION_MODE=ladder
export JS_LADDER_SPLIT=${LADDER_SPLIT}
export USE_KL_LOSS=${USE_KL_LOSS:-True}
export ADAPTIVE_KL_LOSS_COEF=${ADAPTIVE_KL_LOSS_COEF:-True}

# Match the 8 x H200 top-JS training and validation settings.
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-128}
export TEST_FREQ=${TEST_FREQ:-20}
export SAVE_FREQ=${SAVE_FREQ:-20}
export VAL_N=${VAL_N:-32}

# Use more of each H200 GPU for vLLM KV cache and prefill batching.
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}

exec bash "${SCRIPT_DIR}/train_s2d_opd.sh"
