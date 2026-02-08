#!/bin/bash
# =============================================================================
# Helper script to launch intra_model_similarity.py
# =============================================================================

set -e

# --- Configuration Section ---
# DATASETS is an array of dataset paths
DATASETS=(
    "l11p/COCOQA-restval-ICQA"
    # "l11p/VQAv2-COCO-restval-ICQ"
)

# DATASET_SPLITS determines which subsets to process
DATASET_SPLITS=(
    "images_qa"
    # "texts_qa"
)

# Models to evaluate
MODELS=(
    "lbasile/llava_7b"
    "l11p/llava-v1.5-7b-pretrained-lbasile"
)

# Analysis configuration
SIMILARITY_MEASURES="neighborhood_overlap linear_cka information_imbalance information_imbalance_reversed information_imbalance_mean"
# SIMILARITY_MEASURES="linear_cka information_imbalance_mean"

# Analysis Parameters
POOLING="mean_image"
LOWER_TRIANGULAR=true

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
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/intra_model_similarity.py"

# Environment setup for Hugging Face caches
HF_HOME="$HOME/scratch/huggingface"
MODEL_CACHE_DIR="$HF_HOME/hub"
DATASETS_CACHE_DIR="$HF_HOME/datasets"

export HF_HOME
export HF_HUB_CACHE="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="$DATASETS_CACHE_DIR"

# Activate virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

# --- Execution ---
echo "🚀 Starting Intra-Model Similarity Analysis..."
echo "  Results Dir: $RESULTS_DIR"
echo "  Models: ${MODELS[*]}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Splits: ${DATASET_SPLITS[*]}"
echo "  Lower Triangular: $LOWER_TRIANGULAR"
echo "  Hugging Face Home: $HF_HOME"
echo "  HF Model Cache: $MODEL_CACHE_DIR"
echo "  HF Datasets Cache: $DATASETS_CACHE_DIR"

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
        --similarity-measures $SIMILARITY_MEASURES
        --guide-text "$GUIDE_TEXT"
        --downsample-size "$DOWNSAMPLE_SIZE"
        --batch-size "$BATCH_SIZE"
    )

    if [ "$SAVE_REPRESENTATIONS" = "true" ]; then
        CMD_ARGS+=(--save-representations)
    fi

    if [ "$LOWER_TRIANGULAR" = true ]; then
        CMD_ARGS+=(--lower-triangular-only)
    fi

    python3 "$FILEPATH" "${CMD_ARGS[@]}"
done

echo "✅ Analysis complete!"
