#!/bin/bash
#SBATCH --partition=GENOA
#SBATCH --account=dssc
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --job-name=overlap
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#
# Representation-overlap (kNN agreement between image/text neighborhoods),
# CPU-only. Submit directly: sbatch shell/retrieval/overlap.sh
#
# Available models under $REPRESENTATIONS_DIR/embedding_similarity:
#   onevision_8b_hf onevision_4b_hf llava_7b llava_13b internvl_8b_hf
# Available datasets:
#   imagenet_animals_10 coco_image_question coco_image_only coco_text_only

set -euo pipefail

PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi
source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1

python3 scripts/retrieval/overlap_helper.py \
    --model_name onevision_8b_hf \
    --dataset coco_image_question
