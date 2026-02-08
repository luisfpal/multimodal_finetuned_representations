#!/bin/bash
# This script runs consecutive layers similarity analysis

# --- Configuration Section ---
DATASET_PATH_OR_NAME="l11p/VQAv2-COCO-restval-ICQ"

# Processing parameters
BATCH_SIZE="25"
SEED="42"
POOLING_METHODS="last mean_image"
SIMILARITY_MEASURES="neighborhood_overlap linear_cka id_correlation information_imbalance information_imbalance_reversed information_imbalance_mean"

# Dataset configuration (loaded from environment if available)
GUIDE_TEXT="${GUIDE_TEXT:-Answer the question using a single word or phrase.\n}"
DOWNSAMPLE_SIZE="${DOWNSAMPLE_SIZE:-10000}"

# Representations storage configuration (loaded from environment if available)
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$HOME/scratch}"
SAVE_REPRESENTATIONS="${SAVE_REPRESENTATIONS:-true}"

# Directory configuration
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"

# Script configuration
FILENAME="consecutive_layers_similarity.py"
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/$FILENAME"

# Models to analyze
MODELS="llava-hf/llava-1.5-7b-hf l11p/llava-v1.5-7b-pretrained-liuhaotian"

# --- Environment Setup ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

# --- Run Analysis ---
echo "====== Consecutive Layers Similarity Analysis ======"
echo "Dataset: $DATASET_PATH_OR_NAME"
echo "Results directory: $RESULTS_DIR"
echo "Models: $MODELS"
echo "Pooling Methods: $POOLING_METHODS"
echo "Similarity Measures: $SIMILARITY_MEASURES"
echo "Guide Text: $GUIDE_TEXT"
echo "Downsample Size: $DOWNSAMPLE_SIZE"

# Build command with conditional --save-representations flag
CMD_ARGS=(
    --results-dir "$RESULTS_DIR"
    --representations-dir "$REPRESENTATIONS_DIR"
    --dataset-path-or_name "$DATASET_PATH_OR_NAME"
    --models $MODELS
    --batch-size "$BATCH_SIZE"
    --seed "$SEED"
    --guide-text "$GUIDE_TEXT"
    --downsample-size "$DOWNSAMPLE_SIZE"
    --pooling-methods $POOLING_METHODS
    --similarity-measures $SIMILARITY_MEASURES
)

if [ "$SAVE_REPRESENTATIONS" = "true" ]; then
    CMD_ARGS+=(--save-representations)
fi

python "$FILEPATH" "${CMD_ARGS[@]}"

echo -e "\n====== All Analyses Complete ======"
