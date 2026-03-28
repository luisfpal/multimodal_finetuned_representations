# Multimodal Finetuned Representations

Geometric interpretability study analyzing what changes — and where — inside the LLM backbone of vision-language models (VLMs) after visual instruction finetuning. The project extracts residual stream representations from multimodal models and computes similarity measures across layers to understand how finetuning reshapes internal representations.

Key techniques: residual stream tracing, layer patching (transplantation), intrinsic dimension analysis, and similarity measures (CKA, neighborhood overlap, information imbalance).

## Supported Models

| Family              | Models                                         | Size    |
| ------------------- | ---------------------------------------------- | ------- |
| LLaVA 1.5           | `lbasile/llava_7b`, pretrained variants      | 7B, 13B |
| LLaVA OneVision 1.5 | `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`   | 8B      |
| InternVL2           | `OpenGVLab/InternVL2-8B`, pretrained variant | 8B      |
| InternVL3.5         | `OpenGVLab/InternVL3_5-8B`                   | 8B      |
| Cambrian            | `nyu-visionx/cambrian-8b`                    | 8B      |

## Setup

### Requirements

- **Python 3.12** — Flash Attention 2.8.3 only provides pre-built wheels for this version
- **CUDA** >= 12.1 (tested on A100, H100)
- **[uv](https://github.com/astral-sh/uv)** package manager

### Quick Start

If your system already has a CUDA toolkit available (i.e., `nvcc --version` works):

```bash
uv sync
source .venv/bin/activate
```

That's it. The pre-built Flash Attention wheels install without compilation.

### Local CUDA Installation (HPC / No Root Access)

On HPC clusters where CUDA is not in your `PATH` by default, the `uv sync` step will fail because Flash Attention's pre-built wheels need `nvcc` to verify CUDA compatibility at install time. This does NOT mean you need to compile Flash Attention — you just need the CUDA toolkit accessible.

The included script installs a user-local CUDA toolkit (no root required):

```bash
# 1. Install CUDA toolkit locally (~4GB, one-time)
bash scripts/setup/install_local_cuda.sh

# 2. Reload shell environment (the script patches ~/.bashrc)
source ~/.bashrc

# 3. Now uv sync will work
uv sync
source .venv/bin/activate
```

The script supports options for custom paths and CUDA versions:

```bash
bash scripts/setup/install_local_cuda.sh --help
bash scripts/setup/install_local_cuda.sh --install-dir "$HOME/scratch/cuda-12.2" --arch-list "8.0"
```

### Troubleshooting

If `uv sync` still fails on Flash Attention:

1. Remove `flash-attn` from `pyproject.toml`
2. Run `uv sync`
3. Install manually: `uv pip install flash-attn==2.8.3 --no-build-isolation`

## Repository Structure

```
scripts/                    Python scripts for all experiments
  representations_interpretability/   Core analysis (similarity, intrinsic dim, etc.)
  eval/                               Benchmark evaluation (lmms-eval, compositionality)
  setup/                              Environment setup scripts
shell/                      SLURM launchers and worker scripts (mirrors scripts/)
src/                        Core pipeline (residual stream tracer)
utils/                      Shared utilities (model loading, datasets, metrics)
plotting/                   Visualization scripts and notebooks
third_party/                External model code (Cambrian)
```

## Running Experiments

All experiments run on SLURM via a two-layer pattern:

1. **Launcher** (`shell/*/launch_*.sh`) — generates SBATCH script, sets resources, submits job
2. **Worker** (`shell/*/[task].sh`) — sets experiment parameters, calls the Python script

```bash
# Submit a job
bash shell/representations_interpretability/launch_job.sh

# Monitor
tail -f logs/slurm-<JOB_ID>.out
```

| Experiment                   | Launcher                       | Python Script                                                            |
| ---------------------------- | ------------------------------ | ------------------------------------------------------------------------ |
| Cross-modal similarity       | `launch_job.sh`              | `scripts/representations_interpretability/multimodality_similarity.py` |
| Intrinsic dimension          | `launch_job.sh`              | `scripts/representations_interpretability/intrinsic_dimension.py`      |
| Cross-model comparison       | `launch_job.sh`              | `scripts/representations_interpretability/models_similarity.py`        |
| Benchmark evaluation         | `launch_lmms.sh`             | `scripts/eval/lmms_evaluation.py`                                      |
| Compositionality (ARO/BiVLC) | `launch_compositionality.sh` | `scripts/eval/compositionality_evaluation.py`                          |
For local testing (no SLURM), use the `run_local_*.sh` scripts:

```bash
bash shell/eval/run_local_evaluation.sh "lbasile/llava_7b"
```

## Configuration

The `.env` file at the project root sets `PYTHONPATH`, HuggingFace cache directories, and cluster-specific paths. It is sourced automatically by the launcher scripts.

> **Warning:** If this repository is made public, review `.env` to avoid exposing credentials or cluster-specific paths.

