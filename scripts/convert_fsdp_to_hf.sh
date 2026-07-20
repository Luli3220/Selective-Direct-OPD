#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Fill in these two paths. CHECKPOINT_PATH may point to either global_step_xxx
# or global_step_xxx/actor.
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"${REPO_ROOT}/checkpoints/justrl_qwen3_1p7b/global_step_120"}
HF_OUTPUT_PATH=${HF_OUTPUT_PATH:-"${REPO_ROOT}/checkpoints/justrl_qwen3_1p7b/global_step_120_hg"}

# Positional arguments override the paths above:
#   bash scripts/convert_fsdp_to_hf.sh /path/to/global_step_xxx /path/to/hf_output
CHECKPOINT_PATH=${1:-"${CHECKPOINT_PATH}"}
HF_OUTPUT_PATH=${2:-"${HF_OUTPUT_PATH}"}
PYTHON_BIN=${PYTHON_BIN:-python}

if [ -f "${CHECKPOINT_PATH}/actor/fsdp_config.json" ]; then
  ACTOR_PATH="${CHECKPOINT_PATH}/actor"
elif [ -f "${CHECKPOINT_PATH}/fsdp_config.json" ]; then
  ACTOR_PATH="${CHECKPOINT_PATH}"
else
  echo "Error: no fsdp_config.json found under: ${CHECKPOINT_PATH}" >&2
  echo "Pass either a verl global_step directory or its actor directory." >&2
  exit 1
fi

if ! compgen -G "${ACTOR_PATH}/model_world_size_*_rank_*.pt" >/dev/null; then
  echo "Error: no FSDP model shards found under: ${ACTOR_PATH}" >&2
  exit 1
fi

if [ "${FORCE:-0}" != "1" ] && [ -f "${HF_OUTPUT_PATH}/config.json" ] && \
  { compgen -G "${HF_OUTPUT_PATH}/model*.safetensors" >/dev/null || \
    compgen -G "${HF_OUTPUT_PATH}/pytorch_model*.bin" >/dev/null; }; then
  echo "Hugging Face model already exists: ${HF_OUTPUT_PATH}"
  echo "Set FORCE=1 to run the merger again."
  exit 0
fi

export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"

MERGE_ARGS=(
  -m verl.model_merger merge
  --backend fsdp
  --local_dir "${ACTOR_PATH}"
  --target_dir "${HF_OUTPUT_PATH}"
)
if [ "${TRUST_REMOTE_CODE:-0}" = "1" ]; then
  MERGE_ARGS+=(--trust-remote-code)
fi

echo "FSDP actor: ${ACTOR_PATH}"
echo "HF output:  ${HF_OUTPUT_PATH}"
"${PYTHON_BIN}" "${MERGE_ARGS[@]}"

if [ ! -f "${HF_OUTPUT_PATH}/config.json" ] || \
  { ! compgen -G "${HF_OUTPUT_PATH}/model*.safetensors" >/dev/null && \
    ! compgen -G "${HF_OUTPUT_PATH}/pytorch_model*.bin" >/dev/null; }; then
  echo "Error: merger exited, but the Hugging Face model files are incomplete." >&2
  exit 1
fi

echo "Conversion complete: ${HF_OUTPUT_PATH}"
