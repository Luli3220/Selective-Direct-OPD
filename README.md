# Not Every Token Is Worth Distilling: Selective Supervision for Direct-OPD

This repository contains the official code release for **Selective-Direct-OPD**.

## Quick Start

Install dependencies:

```bash
conda create -n selective-opd python=3.12 -y
conda activate selective-opd

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
python -m pip install math-verify pyarrow transformers
cd ..

export PYTHON_BIN="$(command -v python)"
```

See the [setup guide](docs/setup.md) for data preparation, paths, and launch options.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── docs/setup.md                         # Detailed setup instructions
├── eval/                                 # Validation and held-out datasets
├── scripts/
│   ├── train_selective_direct_opd.sh       # Configurable training entry point
│   ├── eval.sh                             # Validation-only evaluation entry point
│   ├── prepare_skywork_math.py             # Training-data converter
│   ├── train_ladder.sh                     # Select a divergence percentile interval
│   ├── train_justrl_qwen_js_top10.sh        # Highest-divergence 10% of tokens
│   ├── train_justrl_qwen_js_bottom10.sh     # Lowest-divergence 10% of tokens
│   ├── train_justrl_qwen_random_top10.sh    # Random 10% of tokens
│   └── convert_fsdp_to_hf.sh               # Export FSDP checkpoints to HF format
└── verl/
    ├── scripts/                           # Dependency installation and utilities
    ├── tests/                             # Bundled tests
    ├── requirements*.txt                  # Dependency specifications
    ├── pyproject.toml                     # Package/build configuration
    ├── setup.py
    ├── LICENSE                            # Upstream license
    └── verl/
        ├── trainer/                       # Training entry point, PPO loop and configs
        ├── workers/                       # Actor, rollout and reward-model workers
        ├── models/                        # Model implementations and integrations
        ├── utils/                         # Data loading, reward scoring and utilities
        └── model_merger/                  # Checkpoint export implementation
```

The token-selection implementation is in [`verl/verl/workers/actor/dp_actor.py`](verl/verl/workers/actor/dp_actor.py). Experiment settings and environment-variable overrides are defined in [`scripts/train_selective_direct_opd.sh`](scripts/train_selective_direct_opd.sh).

## Models and data

The datasets under `eval/` have two distinct roles: AIME24 and AIME25 are used for validation during training, while AIME26, HMMT November 2025, and HMMT February 2026 are reserved for held-out evaluation. Download the following model weights separately and place them under `models/`:

- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (student)
- [JustRL-DeepSeek-1.5B](https://huggingface.co/hbx/JustRL-DeepSeek-1.5B) (post-RL teacher)
- [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) (pre-RL teacher reference)

The default paths are:

```text
models/
├── Qwen3-1.7B/                            # Student
├── JustRL-DeepSeek-1.5B/                   # Post-RL teacher
└── DeepSeek-R1-Distill-Qwen-1.5B/           # Pre-RL teacher reference
datasets/
└── train/
    └── skywork-or1-math-dapo-original.parquet
eval/
├── aime24.parquet
├── aime25.parquet
├── aime26.parquet
├── hmmt_nov_2025.parquet
└── hmmt_feb_2026.parquet
```

The training data is not included. Follow the [setup guide](docs/setup.md#training-data) to prepare the Skywork-OR1 training parquet.

To use files stored elsewhere, set the following variables before training:

```bash
export ACTOR_MODEL_PATH="/path/to/Qwen3-1.7B"
export REWARD_MODEL_PATH="/path/to/JustRL-DeepSeek-1.5B"
export TEACHER_REF_MODEL_PATH="/path/to/DeepSeek-R1-Distill-Qwen-1.5B"
export TRAIN_DATASET="/path/to/train.parquet"
export TEST_DATASET="['/path/to/aime24.parquet','/path/to/aime25.parquet']"
```

## Running experiments

Run commands from the repository root after activating the environment and preparing the required files. For local console logging without a W&B account:

```bash
export PYTHON_BIN="$(command -v python)"
export LOGGER="['console']"

# Highest-divergence 10% of tokens
bash scripts/train_justrl_qwen_js_top10.sh

# Lowest-divergence 10% of tokens
bash scripts/train_justrl_qwen_js_bottom10.sh

# Random 10% of tokens (default selection seed: 42)
bash scripts/train_justrl_qwen_random_top10.sh

# A specified divergence percentile interval, e.g. the 20th–30th percentiles
bash scripts/train_ladder.sh 20_30
```

Choose one experiment per run. For a custom relative-selection experiment:

```bash
JS_TOKEN_SELECTION_MODE=relative \
JS_TOP_FRACTION=0.10 \
EXPERIMENT_NAME=selective_relative_top10 \
bash scripts/train_selective_direct_opd.sh
```

The scripts otherwise default to console and W&B logging. The console-only override above keeps experiment logs local. By default, `MANAGE_RAY=True` stops existing local Ray processes and starts a local Ray head; use a dedicated training machine or set `MANAGE_RAY=False` when using an already configured Ray runtime.

Checkpoints are saved under `${OUTPUT_ROOT}/${EXPERIMENT_NAME}`, validation generations under its `outputs/validation_log/` directory, and training logs under `${LOG_ROOT}/${EXPERIMENT_NAME}.log`.

## Evaluation

Evaluate a Hugging Face-format model on the three held-out datasets:

```bash
bash scripts/eval.sh /path/to/model
```

By default, the script evaluates AIME26, HMMT November 2025, and HMMT February 2026. It uses the same rollout and validation path as training with `VAL_ONLY=True`. Set `TEST_DATASET` to evaluate a subset and `VAL_N` to change the number of sampled responses per problem.

## Acknowledgments

This implementation builds on [Direct-OPD](https://github.com/BytedTsinghua-SIA/Direct-OPD) and [verl](https://github.com/volcengine/verl). We thank their contributors for releasing the code and infrastructure.
