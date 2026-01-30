#!/bin/bash
# =============================================================================
# Helper script to launch intrinsic_dimension.py
# =============================================================================

set -e

# --- Configuration Section ---
# DATASETS is an array of dataset paths
DATASETS=(
    "l11p/COCOQA-restval-ICQA"
    "l11p/VQAv2-COCO-restval-ICQ"
)

# DATASET_SPLITS determines which subsets to process
DATASET_SPLITS=(
    "images_qa"
    "texts_qa"
)

# Models to evaluate
MODELS=(
    "lbasile/llava_7b"
    "l11p/llava-v1.5-7b-pretrained-lbasile"
)

# Analysis Parameters
POOLING="last"
RESIDUAL_STREAM_TYPES="output_layer"

# Processing parameters
BATCH_SIZE="25"
SEED="42"

# Dataset configuration (loaded from environment if available)
GUIDE_TEXT="${GUIDE_TEXT:-Answer the question using a single word or phrase.\n}"
DOWNSAMPLE_SIZE="${DOWNSAMPLE_SIZE:-100}"

# Representations storage configuration (loaded from environment if available)
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$HOME/scratch}"
SAVE_REPRESENTATIONS="${SAVE_REPRESENTATIONS:-true}"

# Directory configuration
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"

# Script configuration
FILENAME="intrinsic_dimension.py"
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/$FILENAME"

# --- Execution ---
echo "🚀 Starting Intrinsic Dimension Analysis..."
echo "  Results Dir: $RESULTS_DIR"
echo "  Models: ${MODELS[*]}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Splits: ${DATASET_SPLITS[*]}"

for DATASET in "${DATASETS[@]}"; do
    echo -e "\n----------------------------------------------------------------"
    echo "📂 Processing Dataset: $DATASET"
    echo "----------------------------------------------------------------"

    CMD_ARGS=(
        --results-dir "$RESULTS_DIR"
        --representations-dir "$REPRESENTATIONS_DIR"
        --dataset-path-or-name "$DATASET"
        --models "${MODELS[@]}"
        --dataset-splits "${DATASET_SPLITS[@]}"
        --pooling-methods "$POOLING"
        --residual-stream-types "$RESIDUAL_STREAM_TYPES"
        --guide-text "$GUIDE_TEXT"
        --downsample-size "$DOWNSAMPLE_SIZE"
        --batch-size "$BATCH_SIZE"
    )

    if [ "$SAVE_REPRESENTATIONS" = "true" ]; then
        CMD_ARGS+=(--save-representations)
    fi

    python3 "$FILEPATH" "${CMD_ARGS[@]}"
done

echo "✅ Analysis complete!"
