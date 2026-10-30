# Multimodal Finetuned Representations

Visual instruction tuning is the step that takes a language model bolted to a
vision encoder and turns it into something that can actually talk about
images — the same kind of step that turns a base LLM into a chat assistant,
applied to vision. This repository studies what that step actually changes
inside the LLM backbone: whether it teaches the backbone a genuinely shared
representation space for images and text, or only a thin translation layer on
top of an otherwise-unchanged language model.

We extract residual-stream activations before and after finetuning, across
six model families, and compare them layer-by-layer with independent methods:
similarity measures (CKA, neighborhood overlap, information imbalance,
intrinsic dimension), causal layer-ablation benchmarking, cross-modal
retrieval, and linguistic probing. Agreement across independent methods is
what makes a finding more than an artifact of one measure.

Companion paper: **Visual Instruction Tuning Aligns Modalities through
Abstraction** ([arXiv:2606.03871](https://arxiv.org/abs/2606.03871)).

## Supported Models

| Family              | Models                                       | Size    |
| ------------------- | -------------------------------------------- | ------- |
| LLaVA 1.5           | `lbasile/llava_7b`, pretrained variants      | 7B, 13B |
| LLaVA OneVision 1.5 | `lmms-lab/LLaVA-OneVision-1.5-{4B,8B}-Instruct` | 4B, 8B |
| InternVL2           | `OpenGVLab/InternVL2-8B`, pretrained variant | 8B      |
| Cambrian            | `nyu-visionx/cambrian-8b`                    | 8B      |
| Llama-Vision (Mllama) | `Meta-Llama-3-11B/90B-Vision-Instruct`     | 11B, 90B |

<details>
<summary><strong>Setup</strong></summary>

### Requirements

- Python 3.12
- NVIDIA GPU driver compatible with CUDA 12 wheels
- CUDA toolkit available on `PATH` so `nvcc --version` works (see
  [Flash Attention & CUDA Setup](#flash-attention--cuda-setup) below if it
  doesn't yet)
- [`uv`](https://github.com/astral-sh/uv) for dependency management

The dependency stack is pinned in `pyproject.toml`. In particular:

- `torch==2.9.0` from the CUDA 12.6 PyTorch wheel index
- `torchvision==0.24.0` from the CUDA 12.6 PyTorch wheel index
- `flash-attn==2.8.3` from a pinned prebuilt wheel for `torch2.9`, CUDA 12, and
  Python 3.12

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

`ORFEO_*` lets `utils/llm_judge.py` mint a JWT against this project's HPC
(ORFEO) auth system, purely so `OPENAI_API_KEY`/`OPENAI_API_URL`/
`MODEL_VERSION` get set for lmms-eval's GPT-judge tasks (MMBench, mmvet,
llava-in-the-wild). On different infrastructure, either set those three
`OPENAI_*` variables directly to point at your own OpenAI-compatible endpoint
(`ensure_llm_judge_env()` no-ops once `OPENAI_API_KEY` is already set — the
ORFEO flow is only a fallback), or adapt `_acquire_orfeo_jwt()` in
`utils/llm_judge.py` to your own auth provider. Only the LLM-judge-scored evals
need this at all; everything else runs without it.

`REPRESENTATIONS_DIR` (defaults to `$HOME/scratch/representations` when unset)
points every representation-caching experiment — extraction, retrieval,
imbalance, probing, ablation — at the same on-disk cache root.

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

If this fails, read [Flash Attention & CUDA Setup](#flash-attention--cuda-setup)
below before doing anything else — almost every install failure on a new
machine traces back to one of the two issues covered there.

</details>

<details>
<summary><strong>Flash Attention &amp; CUDA Setup</strong></summary>

`flash-attn` is a compiled CUDA extension, not a pure-Python wheel — `uv sync`
needs a working `nvcc` on `PATH` to build/link against, and the extension is
compiled for a *specific* GPU compute capability. This is the part of the setup
most likely to break silently on a machine that isn't the one it was first set
up on, so read this section in full rather than trial-and-error-ing flags.

### Path 1: a system CUDA toolkit already exists

On a shared cluster, a CUDA toolkit is often already installed system-wide at
a fixed path (e.g. `/usr/local/cuda-12.2`), just not on `PATH` by default. If
`which nvcc` fails but `ls /usr/local/cuda*` shows something, add it yourself
(e.g. in `~/.bashrc`):

```bash
export CUDA_HOME=/usr/local/cuda-12.2   # match the version actually installed
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

This is a one-time shell configuration change on your machine, not something
this repository can do for you — there's no way to discover "the" system CUDA
path portably across clusters.

### Path 2: no root access, no system CUDA toolkit

Some clusters expose a GPU driver but no toolkit at all, and you can't install
one system-wide. `scripts/setup/install_local_cuda.sh` handles this case: it
downloads the NVIDIA CUDA runfile installer, installs a toolkit under
`~/.local/cuda-<version>` (no `sudo` required), and patches `~/.bashrc` between
managed markers so it's sourced automatically on every new shell.

```bash
bash scripts/setup/install_local_cuda.sh
source ~/.bashrc
uv sync
source .venv/bin/activate
```

For a non-default install location or CUDA version:

```bash
bash scripts/setup/install_local_cuda.sh --help
bash scripts/setup/install_local_cuda.sh \
  --install-dir "$HOME/scratch/cuda-12.2" \
  --arch-list "8.0"
```

Non-default CUDA versions require the exact NVIDIA runfile URL (there's no
generic download pattern to derive it from a version number alone):

```bash
bash scripts/setup/install_local_cuda.sh \
  --cuda-version 12.6.0 \
  --runfile-url "<NVIDIA CUDA runfile URL>"
```

### The `TORCH_CUDA_ARCH_LIST` gotcha (read this if you switch GPUs)

`TORCH_CUDA_ARCH_LIST` tells the build which GPU compute capability to compile
CUDA extensions for. It is **not** auto-detected from the GPU present at
install time in every code path, and a value copied from a different machine
will silently produce a binary that either fails to build or builds fine but
targets the wrong architecture. Match it to the GPU you're actually building
on:

| GPU  | Compute capability | `TORCH_CUDA_ARCH_LIST` |
| ---- | ------------------- | ---------------------- |
| A100 | 8.0                 | `"8.0"`                |
| H100 | 9.0                 | `"9.0"`                |

`install_local_cuda.sh --arch-list` defaults to `"8.0"` (A100) — **if you're
building on an H100, pass `--arch-list "9.0"` explicitly**, or export
`TORCH_CUDA_ARCH_LIST="9.0"` yourself before `uv sync` if you're on Path 1
above. This project has been run on both A100 and H100 nodes; the value simply
has to match whichever node you're building the environment on.

### If the pinned wheel stops matching your stack

The `flash-attn==2.8.3` source in `pyproject.toml` points at a specific
prebuilt wheel for `torch2.9` + `cu12` + `cp312` (see `[tool.uv.sources]`). If
you change the pinned `torch`, CUDA, or Python version, that wheel URL will
404 — you need the matching wheel from the
[flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases)
for your exact combination, or build from source.

### Common Setup Checks

If installation fails, check these first:

1. `python --version` reports Python 3.12.
2. `nvcc --version` works in the same shell where you run `uv sync`.
3. `TORCH_CUDA_ARCH_LIST` matches the GPU you're building on (see above).
4. `.env` points `PYTHONPATH` and Hugging Face caches to valid paths.
5. `pyproject.toml` still contains the pinned `flash-attn` wheel source.

</details>

## Repository Structure

```text
scripts/                    Python experiment and analysis code
  eval/                     Benchmark, compositionality, COCOQA, logit-lens, and
                            layer-ablation (text + multimodal) evaluation
  representations_interpretability/
                            Representation extraction and geometry metrics
  embeddings_extraction/    Multi-model activation extraction (hook-based) +
                            information imbalance / intrinsic dimension
  retrieval/                Cross-modal retrieval and representation-overlap evaluation
  probing/                  SentEval-style linguistic probing
  models_utils/             Model loading, porting, and download helpers
  create_datasets/          One-off dataset construction/patching
  captioning/               Caption generation and scoring
  setup/                    Environment setup helpers
shell/                      Local and SLURM entrypoints for experiments (mirrors
                            the scripts/ subdirectory layout 1:1)
src/                        Core tracing pipeline
utils/                      Model, dataset, and metric utilities
  vlm_layer_skipping/       Layer-skipping ablation library (generic primitive +
                            per-model patch/lm-eval-harness wrappers)
  embeddings_extraction/    Library backing scripts/embeddings_extraction/
  probing/                  Library backing scripts/probing/
plotting/                   Figure and analysis scripts
third_party/                External model code
```

## Reproducing the Experiments

Every experiment family below is one lens on the same question (what changes
inside the model after visual instruction tuning). Each row is a distinct,
independently-runnable entrypoint; open the target script and review the
configuration block at the top before submitting (model lists, dataset lists,
SLURM resources, and output directories are set there, not on the command
line).

| Experiment | What it measures | Entry point |
| ---------- | ----------------- | ----------- |
| Text-image representation similarity | CKA / neighborhood overlap / information imbalance between image and text residual streams, per layer | `shell/representations_interpretability/multimodality_similarity.sh` |
| Representation classification | Per-layer label-overlap / kNN classification on MCQ benchmarks | `shell/representations_interpretability/representations_classification.sh` |
| Llama-Vision multimodality similarity | Same similarity analysis, specialized for the 90B Llama-Vision model | `shell/representations_interpretability/llama_vision_similarity.sh` |
| Multi-model embeddings extraction | Hook-based activation extraction across all 6 models (distinct engine from the residual-stream tracer above) | `shell/embeddings_extraction/launch_extract_embeddings.sh` |
| Information imbalance / intrinsic dimension | Pretrained-vs-finetuned representation geometry, pre-extracted embeddings | `shell/embeddings_extraction/launch_information_imbalance.sh` |
| Cross-modal retrieval | Recall@k / NDCG / mrr between image and text embeddings | `shell/retrieval/launch_retrieval.sh` |
| Representation overlap | kNN agreement between image and text neighborhoods | `shell/retrieval/overlap.sh` |
| Linguistic probing | SentEval-style semantic probing on VLM hidden states | `shell/probing/launch_probing.sh` |
| Text-only layer-ablation benchmarking | Causal intervention: skip decoder layers, measure lm-eval-harness task degradation | `shell/eval/launch_layer_ablation_text.sh` |
| Multimodal layer-ablation benchmarking | Same causal intervention, masking image attention specifically, across lmms-eval benchmarks | `shell/eval/launch_layer_ablation_multimodal.sh` |
| LMMS-style benchmark evaluation | Standard VLM benchmark accuracy (no ablation) | `shell/eval/run_local_evaluation.sh` |
| COCOQA evaluation | VQA accuracy under visual / human-caption / synthetic-caption modes | `shell/eval/run_local_cocoqa_evaluation.sh` |
| Compositionality evaluation | ARO / BiVLC / SugarCrepe constrained-logit MCQ scoring | `shell/eval/launch_compositionality.sh` |
| Logit-lens evaluation | Per-layer logit-lens prediction accuracy | `shell/eval/launch_logitlens.sh` |

Local scripts run directly after setup:

```bash
bash shell/eval/run_local_evaluation.sh "lbasile/llava_7b"
```

SLURM launchers should be configured (open the file, check the resource and
parameter block) then submitted:

```bash
bash shell/eval/launch_compositionality.sh
```

Logs are written under `logs/` by the launchers that submit SLURM jobs.

### Constructing the pretrained/finetuned reference checkpoints

Several similarity and imbalance experiments above compare a finetuned model
against a "pretrained" reference checkpoint (same LLM backbone and vision
encoder, before visual instruction tuning). These references were built once
with a handful of standalone scripts, not part of any recurring pipeline —
run manually only if you need to reconstruct a reference checkpoint from
scratch:

- `scripts/build_cambrian_pretrained.py` — assembles a from-scratch Cambrian
  pretrained checkpoint (LLM + vision encoders + pretrain-stage projector).
- `scripts/download_cambrian_projector.py` — downloads and inspects the
  Cambrian pretrain-stage projector weights.
- `scripts/models_utils/create_pretrained_models.py` — swaps a base LLM +
  pretrained projector into a LLaVA checkpoint to synthesize its pretrained
  counterpart.
- `scripts/compare_model_weights.py` — diffs state-dict keys/values between
  two checkpoints, useful for sanity-checking the above.

### Known limitations

Some `shell/eval/run_local_*.sh` scripts still hardcode an author-local
`PROJECT_ROOT` (e.g. `/u/dssc/lbasile/scratch/plm`); others assume the
repository lives at `$HOME/multimodal_finetuned_representations`. Check the
top of the script before running it on a machine other than the one it was
last edited on.

## Configuration Notes

- Do not commit real credentials in `.env`.
- Keep large model, dataset, representation, and evaluation caches outside the repository.
- Use `uv sync` to install from the lockfile. Do not install project dependencies with raw `pip`.
