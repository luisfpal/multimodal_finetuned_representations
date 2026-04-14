# AGENTS.md — Codex Project Instructions

Project-specific context for the `multimodal_finetuned_representations` ML research repo.
Global rules (Six Core Laws, memory protocol, skill auto-invocation) apply from `~/.codex/AGENTS.md`.

## Project

Geometric interpretability study of VLMs (LLaVA 1.5/1.6, InternVL2/3.5).
Extracts residual stream representations from multimodal model backbones; computes similarity
measures (CKA, neighborhood overlap, information imbalance, distance correlation, intrinsic
dimension) to understand how visual instruction finetuning reshapes internal representations.

## Environment

- Python 3.12 (Flash Attention 2.8.3 wheel requirement — no other version)
- `uv` exclusively. NEVER pip. Activate: `source .venv/bin/activate`
- `.env` at project root: PYTHONPATH, HF cache, ORFEO API credentials
- SLURM cluster (ORFEO HPC), results cached in `$HOME/scratch`

## Experiment Workflow (Two-Layer Pattern)

```
Launcher (shell/*/launch_*.sh)
  → Worker (shell/*/[task_name].sh)
    → Python script (scripts/*/[task].py)
```

Always check for an existing `launch_*.sh` before creating new run scripts.
Monitor jobs: `tail -f logs/slurm-<JOB_ID>.out` / `squeue -u $USER`

Key experiment scripts:
| Python Script | Purpose |
|---------------|---------|
| `scripts/representations_interpretability/multimodality_similarity.py` | Image vs text residual stream similarity |
| `scripts/representations_interpretability/intra_model_similarity.py` | Layer-to-layer similarity within model |
| `scripts/representations_interpretability/intrinsic_dimension.py` | Intrinsic dimensionality |
| `scripts/representations_interpretability/models_similarity.py` | Cross-model comparison |
| `scripts/eval/lmms_evaluation.py` | Benchmark evaluation (lmms-eval) |
| `scripts/eval/logitlens_evaluation.py` | Logit lens analysis |
| `scripts/eval/ablation_layers_evaluation.py` | Layer ablation |

## Core Architecture

Central extraction engine: `src/residual_stream_tracer.py`

```
ResidualStreamTracer (ABC)
  ├── ResidualStreamOutputLayerTracer       # output_hidden_states=True
  └── HookBasedResidualStreamTracer (ABC)
        ├── ResidualStreamPostMLPTracer     # hooks on MLP output
        └── ResidualStreamHeadsProjectionTracer  # hooks on o_proj
```

Entry point: `residual_stream_tracer()` — handles caching (safetensors), tracer selection,
batch iteration, memory management.

Pooling methods: `mean`, `last`, `none`, `sample`, `mean_image`
Output shapes: `(num_layers, num_samples, hidden_size)` for pooled; dict for `none`.

## Key Utils

- `utils/model_utils.py` — `load_hf_model_and_processor_or_tokenizer()`, `replace_multimodal_lm()`, `replace_multimodal_projector()`
- `utils/dataset.py` — `MultipleChoiceDatasetBenchmark`, `OpenVQADataset`, `ImageCaptioningDataset`, `get_dataloader()`
- `utils/metrics_utils.py` — `linear_cka`, `neighborhood_overlap`, `information_imbalance`, `distance_correlation`, intrinsic dimension
- `utils/internvl_utils.py` — dynamic image tiling, chat template, vision-text embedding
- `utils/layers_weights_patching.py` — layer-level weight patching between pretrained/finetuned
- `utils/constants.py` — chat templates, system prompts, answer choice mappings

## ML Conventions

- `model.eval()` + `torch.no_grad()` for all inference
- Default dtype: `torch.bfloat16`
- Left-padding for batches: `processor.padding_side = "left"`
- Device: always `model.device` — never hardcode `"cuda:0"`
- Cleanup after large batch ops: `torch.cuda.empty_cache()`, `gc.collect()`
- New similarity measures: signature `(data1, data2) -> float`, add to `metrics_utils.py`
- New tracers: inherit from `ResidualStreamTracer` hierarchy

## Supported Models

- LLaVA 1.5: `lbasile/llava_7b`, `lbasile/llava_13b` (finetuned); pretrained variants
- LLaVA v1.6: `lbasile/llava-v1.6-*` variants
- InternVL2 8B: `OpenGVLab/InternVL2-8B` (finetuned), `OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain`
- InternVL3.5: standard `AutoModel` loading

## Validation (Opt-in Only)

Do NOT run pytest by default. Only when explicitly requested.
Pre-commit: ruff (format + lint + imports) + prettier (YAML, JSON) auto-run on `git commit`.
Fix order on failure: formatting → type errors → linting.
