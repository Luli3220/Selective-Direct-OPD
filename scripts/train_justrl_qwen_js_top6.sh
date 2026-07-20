#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Run this grid-search point in an isolated output directory.
export PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-2}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-justrl_qwen3_1p7b_js_top6_kl_no_nvlink}

# Keep the original Direct-OPD KL setup; only change the JS token fraction.
export JS_TOKEN_FILTER_ENABLED=${JS_TOKEN_FILTER_ENABLED:-True}
export JS_TOP_FRACTION=${JS_TOP_FRACTION:-0.06}
export USE_KL_LOSS=${USE_KL_LOSS:-True}
export ADAPTIVE_KL_LOSS_COEF=${ADAPTIVE_KL_LOSS_COEF:-True}

# Use more of each 80GB GPU for vLLM KV cache and prefill batching.
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}

# The two GPUs have no NVLink/P2P path. Let NCCL use shared memory/PCIe instead.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}

exec bash "${SCRIPT_DIR}/train_justrl_qwen.sh"
