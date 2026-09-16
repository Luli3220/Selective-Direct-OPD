# Setup

This repository includes the Selective-Direct-OPD training code and processed evaluation datasets. Model weights and the main training parquet must be downloaded separately.

## Environment

```bash
conda create -n selective-opd python=3.12 -y
conda activate selective-opd

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify pyarrow transformers
cd ..
```

The launch scripts use `/usr/bin/python3.12` by default. Override it when the environment uses another path:

```bash
export PYTHON_BIN="$(which python)"
```

## Required files

The default paths are:

```text
models/
  Qwen3-1.7B/
  JustRL-DeepSeek-1.5B/
  DeepSeek-R1-Distill-Qwen-1.5B/

datasets/
  train/skywork-or1-math-dapo-original.parquet

eval/
  aime24.parquet
  aime25.parquet
  aime26.parquet
  hmmt_nov_2025.parquet
  hmmt_feb_2026.parquet
```

Download the model weights from [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [JustRL-DeepSeek-1.5B](https://huggingface.co/hbx/JustRL-DeepSeek-1.5B), and [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B). The processed evaluation files are already included.

### Training data

Download the [Skywork-OR1 math parquet](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data/blob/main/data/math-00000-of-00001.parquet), then convert it from the repository root:

```bash
mkdir -p datasets/raw datasets/train

curl -L --fail \
  https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data/resolve/main/data/math-00000-of-00001.parquet \
  -o datasets/raw/skywork-or1-math.parquet

python scripts/prepare_skywork_math.py \
  --input datasets/raw/skywork-or1-math.parquet \
  --output datasets/train/skywork-or1-math-dapo-original.parquet
```

The converter applies the DAPO-style prompt used by Direct-OPD and validates the generated parquet.

## Training

Launch the default selective training configuration:

```bash
bash scripts/train_selective_direct_opd.sh
```

The repository also provides launchers for the top-10%, bottom-10%, random-token, and percentile-interval experiments under `scripts/`.

To redirect models, data, checkpoints, and logs:

```bash
MODEL_ROOT=/path/to/models \
DATA_ROOT=/path/to/datasets \
OUTPUT_ROOT=/path/to/checkpoints \
LOG_ROOT=/path/to/logs \
bash scripts/train_selective_direct_opd.sh
```

Common overrides:

```bash
ACTOR_MODEL_PATH=/path/to/student \
REWARD_MODEL_PATH=/path/to/post_rl_teacher \
TEACHER_REF_MODEL_PATH=/path/to/pre_rl_teacher \
TRAIN_DATASET=/path/to/train.parquet \
TOTAL_TRAINING_STEPS=300 \
GPUS_PER_NODE=8 \
NUM_NODES=1 \
bash scripts/train_selective_direct_opd.sh
```

AIME24 and AIME25 are used as validation sets during training. Checkpoints are saved under `${OUTPUT_ROOT}/${EXPERIMENT_NAME}`, and validation generations are written under `${CHECKPOINT_DIR}/outputs/validation_log/`.

## Held-out evaluation

Evaluate a Hugging Face-format model on AIME26 and the two HMMT sets:

```bash
bash scripts/eval.sh /path/to/model
```

The evaluation launcher uses the same rollout and validation path as training with `VAL_ONLY=True`. Use `TEST_DATASET` to select a subset or `VAL_N` to change the number of sampled responses per problem.

## Notes

- `LOGGER="['console']"` disables W&B logging.
- `MANAGE_RAY=True` starts and stops a local Ray head inside the launcher.
- `PYTHON_BIN`, `CUDA_VISIBLE_DEVICES`, and the batch-size variables can be overridden for the local environment.
