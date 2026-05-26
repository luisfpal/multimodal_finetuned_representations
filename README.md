# Multimodal Finetuned Representations

Code for studying how visual instruction tuning changes the internal geometry of
vision-language models (VLMs). The repository extracts residual-stream
representations from multimodal models and compares them across layers, models,
modalities, and evaluation settings.

This README is a collaborator-facing snapshot. It documents the current setup and
selected experiment entrypoints. The full paper reproduction checklist for
**Visual Instruction Tuning Aligns Modalities through Abstraction** will be
expanded as the workflow is finalized.

## Supported Models

| Family              | Models                                       | Size    |
| ------------------- | -------------------------------------------- | ------- |
| LLaVA 1.5           | `lbasile/llava_7b`, pretrained variants      | 7B, 13B |
| LLaVA OneVision 1.5 | `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`   | 8B      |
| InternVL2           | `OpenGVLab/InternVL2-8B`, pretrained variant | 8B      |
| Cambrian            | `nyu-visionx/cambrian-8b`                    | 8B      |

## Setup

### Requirements

- Python 3.12
- NVIDIA GPU driver compatible with CUDA 12 wheels
- CUDA toolkit available on `PATH` so `nvcc --version` works
- [`uv`](https://github.com/astral-sh/uv) for dependency management

The dependency stack is pinned in `pyproject.toml`. In particular:

- `torch==2.9.0` from the CUDA 12.6 PyTorch wheel index
- `torchvision==0.24.0` from the CUDA 12.6 PyTorch wheel index
- `flash-attn==2.8.3` from a pinned prebuilt wheel for `torch2.9`, CUDA 12, and Python 3.12

### Environment Configuration

Create a local environment file from the example:

```bash
cp .env.example .env
```

Edit `.env` for your machine. The important values are:

- `PYTHONPATH`: absolute path to this repository
- `HF_HOME`: root Hugging Face cache directory
- `HF_HUB_CACHE`: Hugging Face model cache
- `HF_DATASETS_CACHE`: Hugging Face dataset cache
- `HF_TOKENS_CACHE`: Hugging Face token cache

The `ORFEO_*` variables are only needed for cluster-specific LLM judge tasks.
Leave them as placeholders if you are not using that infrastructure.

### Install Dependencies

If `nvcc --version` already works:

```bash
uv sync
source .venv/bin/activate
```

Then verify the core stack:

```bash
python -c "import torch, flash_attn; print(torch.__version__)"
```

### CUDA On HPC / No Root Access

Some clusters provide GPU drivers but do not expose a CUDA toolkit by default.
If `uv sync` fails while installing FlashAttention and `nvcc --version` does not
work, install a user-local CUDA toolkit:

```bash
bash scripts/setup/install_local_cuda.sh
source ~/.bashrc
uv sync
source .venv/bin/activate
```

For custom locations or CUDA versions:

```bash
bash scripts/setup/install_local_cuda.sh --help
bash scripts/setup/install_local_cuda.sh --install-dir "$HOME/scratch/cuda-12.2" --arch-list "8.0"
```

If using a non-default CUDA version, pass the NVIDIA runfile URL explicitly:

```bash
bash scripts/setup/install_local_cuda.sh \
  --cuda-version 12.6.0 \
  --runfile-url "<NVIDIA CUDA runfile URL>"
```

### Common Setup Checks

If installation fails, check these first:

1. `python --version` reports Python 3.12.
2. `nvcc --version` works in the same shell where you run `uv sync`.
3. `.env` points `PYTHONPATH` and Hugging Face caches to valid paths.
4. `pyproject.toml` still contains the pinned `flash-attn` wheel source.

## Repository Structure

```text
scripts/                    Python experiment and analysis code
  eval/                     Benchmark, compositionality, COCOQA, and logit-lens evaluation
  representations_interpretability/
                            Representation extraction and geometry metrics
  setup/                    Environment setup helpers
shell/                      Local and SLURM entrypoints for experiments
src/                        Core tracing pipeline
utils/                      Model, dataset, and metric utilities
plotting/                   Figure and analysis scripts
third_party/                External model code
```

## Selected Current Entrypoints

The scripts below cover some of the current experiments for the paper. This list
is intentionally partial and will be updated as the full reproduction workflow is
finalized.

| Purpose | Entry point |
| ------- | ----------- |
| LMMS-style benchmark evaluation | `shell/eval/run_local_evaluation.sh` |
| COCOQA evaluation | `shell/eval/run_local_cocoqa_evaluation.sh` |
| Compositionality evaluation | `shell/eval/launch_compositionality.sh` |
| Logit-lens evaluation | `shell/eval/launch_logitlens.sh` |
| Llama Vision multimodality similarity | `shell/representations_interpretability/llama_vision_similarity.sh` |
| Text-image representation similarity | `shell/representations_interpretability/multimodality_similarity.sh` |
| Representation classification metrics | `shell/representations_interpretability/representations_classification.sh` |

Before running a script, open it and review the configuration block at the top.
Several scripts define model lists, dataset lists, SLURM resources, output
directories, and cache locations directly in bash.

Some current scripts still contain author-local `PROJECT_ROOT` values such as
`/u/dssc/lbasile/scratch/plm`, while others assume the repository lives at
`$HOME/multimodal_finetuned_representations`. Adjust those paths before running
experiments on another machine.

## Running Experiments

Local scripts can be run directly after setup:

```bash
bash shell/eval/run_local_evaluation.sh "lbasile/llava_7b"
```

SLURM launchers should be configured first, then submitted:

```bash
bash shell/eval/launch_compositionality.sh
```

Logs are written under `logs/` by the launchers that submit SLURM jobs.

## Configuration Notes

- Do not commit real credentials in `.env`.
- Keep large model, dataset, representation, and evaluation caches outside the repository.
- Use `uv sync` to install from the lockfile. Do not install project dependencies with raw `pip`.
