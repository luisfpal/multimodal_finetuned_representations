#!/bin/bash
# Download ARO order subtask images (COCO val2014 + Flickr30k).
# Run once before enabling coco_order / flickr30k_order subtasks.
#
# Usage:
#   bash shell/data/download_aro_images.sh           # download both
#   bash shell/data/download_aro_images.sh coco       # COCO only
#   bash shell/data/download_aro_images.sh flickr30k  # Flickr30k only
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$PROJECT_ROOT/.venv/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

python "$PROJECT_ROOT/scripts/data/download_aro_images.py" \
    --aro-data-root "$HOME/scratch/datasets/vision-language-models-are-bows" \
    --dataset "${1:-all}"
