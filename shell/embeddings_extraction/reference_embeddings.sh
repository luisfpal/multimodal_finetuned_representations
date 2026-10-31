#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=dssc
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --job-name=reference_embeddings
#SBATCH --gres=gpu:H100:1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#
# Unimodal reference embedders (SigLIP2 for images, Qwen3-Embedding for text)
# used as pseudolabel/anchor encoders for the retrieval and overlap experiments.
# Submit directly: sbatch shell/embeddings_extraction/reference_embeddings.sh

set -euo pipefail

PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi
source "$PROJECT_ROOT/.venv/bin/activate"

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
    scripts/embeddings_extraction/siglip_embedding.py \
    --model_name_or_path google/siglip2-so400m-patch14-384 \
    --image_dir "${IMAGENET_DIR:?set IMAGENET_DIR in .env}" \
    --dataset imagenet \
    --class_subset animals_10 \
    --num_images_per_class 100

# Qwen3 text-embedding counterpart (run separately, CPU/GPU text-only):
#   python3 scripts/embeddings_extraction/qwen_embedding.py \
#       --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
#       --dataset_name llava_descriptions
