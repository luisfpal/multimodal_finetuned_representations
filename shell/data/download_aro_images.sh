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

# ARO source code lives in third_party/vision-language-models-are-bows/ (tracked in git).
# This script only downloads the image data to scratch.

python "$PROJECT_ROOT/scripts/data/download_aro_images.py" \
    --aro-data-root "$HOME/scratch/datasets/vision-language-models-are-bows" \
    --dataset "${1:-all}"