#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=dssc
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --job-name=extract_embeddings
#SBATCH --gres=gpu:H100:1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#
# Multi-model activation extraction (hook-based, distinct from
# src/residual_stream_tracer.py). NOTE: model_path must be set via
# `sbatch --export=ALL,model_path=...` (see launch_extract_embeddings.sh) —
# it is not defined in this file. Supported values: LLaVA-OneVision-1.5-4B/8B,
# InternVL2-8B, LLaVA 1.5 7B/13B, Cambrian-8B.
#
# ---------- Dataset options (--dataset {coco,imagenet,coco_karpathy,winoground}) ----------
# COCO (LLaVA detail_23k.json over MSCOCO train2014 images):
#   --dataset coco --image_dir <coco-train2014-dir> --detail_json ./detail_23k.json
#     image + question text (default) -> {out_dir}/{model}/coco_image_question/
#     --image_only (no question text) -> {out_dir}/{model}/coco_image_only/
#     --text_only  (gpt description, no image, --image_dir ignored, mutex with --image_only)
#                                      -> {out_dir}/{model}/coco_text_only/
#
# COCO Karpathy (yerevann/coco-karpathy test split = 5000 images from val2014):
#   --dataset coco_karpathy [--karpathy_split test] --image_dir <coco-val2014-dir>
#     --image_only -> {out_dir}/{model}/coco_karpathy_test_image_only/
#     --text_only  -> {out_dir}/{model}/coco_karpathy_test_text_only/
#
# Winoground (facebook/winoground; 400 entries x (image_0/1, caption_0/1) = 800 rows):
#   --dataset winoground [--winoground_split test]  (--image_dir unused; PNGs cached
#   under $HF_HOME/winoground_images, override with --winoground_cache_dir)
#     --image_only -> {out_dir}/{model}/winoground_test_image_only/
#     --text_only  -> {out_dir}/{model}/winoground_test_text_only/
#     Layout: rows ordered (id=0,side=0),(id=0,side=1),(id=1,side=0),...;
#             reshape saved [800,D] tensors to [400,2,D] for per-entry pairs.
#
# ImageNet (one starter caption per image, round-robin over caption_starters.py):
#   --dataset imagenet --image_dir <imagenet-root>
#   --class_subset animals_10        # named list in an animal_lists module (not included in this release), OR
#   --num_classes N --seed K         # random N-class sample
#   [--num_images_per_class M]       # cap per class (deterministic, sorted)
#     -> {out_dir}/{model}/imagenet_<subset|captions>/
#
# ---------- Representation / extraction options ----------
# --target_layer {norm1,norm2,res2}  which decoder sub-layer to hook (default norm1).
#                                     norm1=input_layernorm, norm2=post_attn, res2=block-output residual.
# --avg_token                        mean-pool over non-pad positions instead of last-token.
# --extract_all                      also save per-layer extras: _imgavg/_eoi/_imgeoi/_last/
#                                     _textavg/_textspec/_userlast (see embeddings_vlms.py docstring).
# --every N                          stride over decoder layers (default 2); 1 = save all layers.
# --odd_layers                       shift the sweep to start at 1 (fills in layers a default
#                                     --every 2 run skipped).
# --batch_size                       must remain 1 (per-image variable tile/expansion sizes).

set -euo pipefail

PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi
source "$PROJECT_ROOT/.venv/bin/activate"

: "${model_path:?model_path must be set, e.g. via sbatch --export=ALL,model_path=...}"

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes[0]}
GPU_NUMBER=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' ' | wc -w)

export PYTHONUNBUFFERED=1

torchrun \
    --nnodes="$SLURM_NTASKS" \
    --node-rank="$SLURM_NODEID" \
    --nproc-per-node="$GPU_NUMBER" \
    --rdzv-id="$SLURM_PROCID" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$head_node:0" \
    scripts/embeddings_extraction/embeddings_vlms.py \
    --model_name_or_path "$model_path" \
    --dataset winoground \
    --target_layer res2 \
    --out_dir "${REPRESENTATIONS_DIR:-$HOME/scratch/representations}/embedding_similarity" \
    --extract_all \
    --image_only
