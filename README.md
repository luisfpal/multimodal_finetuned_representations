# Multimodal Finetuned Representations

Interpreting multimodal models through their finetuned representations.

## Setup

Use `uv` with `uv.lock`. Tested on A100 40GB VRAM.

## Evaluations

Completed benchmarks (ready for plotting):

- scienceqa_img
- mme
- textvqa_val
- pope
- gqa
- mmbench_en_dev
- llava_in_the_wild

Example: `evals/llava-1.5-7b/standalones_runs/MME/mme_plots.ipynb`

## Files

Working:

- `scripts/eval/lmms_evaluation.py`
- `shell/eval/lmms_evaluation.sh`
- `shell/launch_lmms.sh`

Others need updates. `.env` shared, safe for ORFEO.
