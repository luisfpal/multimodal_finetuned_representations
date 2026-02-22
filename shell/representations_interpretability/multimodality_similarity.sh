#!/bin/bash
# This script runs multimodality similarity analysis (Text vs Image)

# --- Configuration Section ---
# Datasets to process
DATASETS=(
    # "l11p/COCOQA-restval-ICQA"
    # "l11p/VQAv2-COCO-restval-ICQ"
    "l11p/GQA-val-balanced-noyesno-captioned"
    # "l11p/MME-test-captioned"
    # "l11p/MMBench_EN-dev-captioned"
)

# Caption source: "human" for COCOQA / VQAv2-ICQ (which have human captions),
# "synthetic" for everything else. Always passed explicitly so path is deterministic.
CAPTION_SOURCE="${CAPTION_SOURCE:-synthetic}"

# Analysis configuration
TOKENS_POOLING_METHODS="last"
SIMILARITY_MEASURES="neighborhood_overlap linear_cka information_imbalance information_imbalance_reversed information_imbalance_mean"

# Processing parameters
BATCH_SIZE="1" # 1 for MMBench
SEED="42"

# Dataset configuration (loaded from environment if available)
GUIDE_TEXT="${GUIDE_TEXT:-Answer the question using a single word or phrase.}"
DOWNSAMPLE_SIZE="${DOWNSAMPLE_SIZE:-100}"

# Representations storage configuration (loaded from environment if available)
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$HOME/scratch}"
SAVE_REPRESENTATIONS="${SAVE_REPRESENTATIONS:-true}"
ENABLE_SEMANTIC_FILTER="${ENABLE_SEMANTIC_FILTER:-false}"

# Force recompute: overwrite existing representations and similarity results.
# (unlike dry-run, results ARE saved when force is true)
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-false}"

# Models to evaluate
MODELS=(
    # "lbasile/llava_7b"
    # "l11p/llava-v1.5-7b-pretrained-lbasile"
    # "lbasile/llava_13b"
    # "l11p/llava-v1.5-13b-pretrained-lbasile"
    "OpenGVLab/InternVL2-8B"  # Finetuned
    "OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"  # Pretrained
    # "lbasile/llava_7b_6_17"
    # "lbasile/llava_7b_6_17_complementary"
    # "OpenGVLab/InternVL3_5-8B-Pretrained"
    # "OpenGVLab/InternVL3_5-8B-Instruct"
    # "OpenGVLab/InternVL3_5-8B-MPO"
    # "OpenGVLab/InternVL3_5-8B"
)

# Directory configuration
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"

# Script configuration
FILENAME="multimodality_similarity.py"
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/$FILENAME"

# --- Environment Setup ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Run Analysis ---
echo "====== Multimodality Similarity Analysis ======"
echo "Results directory: $RESULTS_DIR"

for DATASET_PATH_OR_NAME in "${DATASETS[@]}"; do
    echo -e "\n================================================================"
    echo "🌍 Processing Dataset: $DATASET_PATH_OR_NAME"
    echo "================================================================"

    for MODEL_NAME in "${MODELS[@]}"; do
        echo -e "\n----------------------------------------------------------------"
        echo "🤖 Processing Model: $MODEL_NAME"
        echo "----------------------------------------------------------------"

        # Activate model-specific venv
        VENV="$PROJECT_DIR/.venv"
        source "$VENV/bin/activate"

        # Build command with conditional flags
        CMD_ARGS=(
            --results-dir "$RESULTS_DIR"
            --representations-dir "$REPRESENTATIONS_DIR"
            --dataset-path-or-name "$DATASET_PATH_OR_NAME"
            --model-name-or-path "$MODEL_NAME"
            --batch-size "$BATCH_SIZE"
            --seed "$SEED"
            --guide-text "$GUIDE_TEXT"
            --downsample-size "$DOWNSAMPLE_SIZE"
            --caption-source "$CAPTION_SOURCE"
            --pooling-methods ${TOKENS_POOLING_METHODS}
            --similarity-measures ${SIMILARITY_MEASURES}
        )

        if [ "$SAVE_REPRESENTATIONS" = "true" ]; then
            CMD_ARGS+=(--save-representations)
        fi

        if [ "$ENABLE_SEMANTIC_FILTER" = "true" ]; then
            CMD_ARGS+=(--enable-semantic-filter)
        fi

        if [ "$FORCE_RECOMPUTE" = "true" ]; then
            CMD_ARGS+=(--force)
        fi

        python3 "$FILEPATH" "${CMD_ARGS[@]}"

        echo "✅ Completed: $MODEL_NAME"
    done
done

echo ""
echo "====== All Analyses Complete ======"
