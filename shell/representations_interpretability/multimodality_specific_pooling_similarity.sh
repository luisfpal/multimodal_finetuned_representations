#!/bin/bash
# This script runs multimodality similarity analysis (Text vs Image) with specific pooling methods per split

# --- Configuration Section ---
# Datasets to process
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
    "llava-hf/llava-1.5-7b-hf"
    "l11p/llava-v1.5-7b-pretrained-liuhaotian"
)

# Analysis configuration
# Note: Pooling methods are hardcoded per split in the python script (text=mean, image=mean_image)
SIMILARITY_MEASURES="neighborhood_overlap linear_cka information_imbalance information_imbalance_reversed information_imbalance_mean"

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
FILENAME="multimodality_specific_pooling_similarity.py"
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/$FILENAME"

# --- Environment Setup ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Run Analysis
echo "====== Multimodality Specific Pooling Similarity Analysis ======"
echo "Results directory: $RESULTS_DIR"

for DATASET_PATH_OR_NAME in "${DATASETS[@]}"; do
    echo -e "\n================================================================"
    echo "🌍 Processing Dataset: $DATASET_PATH_OR_NAME"
    echo "================================================================"

    for MODEL_NAME in "${MODELS[@]}"; do
        echo -e "\n----------------------------------------------------------------"
        echo "🤖 Processing Model: $MODEL_NAME"
        echo "----------------------------------------------------------------"

        # Build command with conditional --save-representations flag
        CMD_ARGS=(
            --results-dir "$RESULTS_DIR"
            --representations-dir "$REPRESENTATIONS_DIR"
            --dataset-path-or-name "$DATASET_PATH_OR_NAME"
            --model-name-or-path "$MODEL_NAME"
            --batch-size "$BATCH_SIZE"
            --seed "$SEED"
            --guide-text "$GUIDE_TEXT"
            --downsample-size "$DOWNSAMPLE_SIZE"
            --similarity-measures ${SIMILARITY_MEASURES}
            --dataset-splits "${DATASET_SPLITS[@]}"
        )

        if [ "$SAVE_REPRESENTATIONS" = "true" ]; then
            CMD_ARGS+=(--save-representations)
        fi

        python3 "$FILEPATH" "${CMD_ARGS[@]}"

        echo "✅ Completed: $MODEL_NAME"
    done
done

echo -e "\n====== All Analyses Complete ======"
