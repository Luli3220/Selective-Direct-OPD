#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

if [ "$#" -gt 1 ]; then
  echo "Usage: bash scripts/eval.sh [MODEL_PATH]" >&2
  exit 2
fi

if [ "$#" -eq 1 ]; then
  export ACTOR_MODEL_PATH="$1"
fi

export PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
export TEST_DATASET=${TEST_DATASET:-"['${REPO_ROOT}/eval/aime26.parquet','${REPO_ROOT}/eval/hmmt_nov_2025.parquet','${REPO_ROOT}/eval/hmmt_feb_2026.parquet']"}

# verl initializes a training dataset before entering its validation-only path.
export TRAIN_DATASET=${TRAIN_DATASET:-"${REPO_ROOT}/eval/aime24.parquet"}
export VAL_BEFORE_TRAIN=True
export VAL_ONLY=True
export REWARD_MODEL_ENABLE=False
export TEACHER_REF_REWARD_MODEL_ENABLE=False
export LOGGER=${LOGGER:-"['console']"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-s2d_opd_eval}

exec bash "${SCRIPT_DIR}/train_s2d_opd.sh"
