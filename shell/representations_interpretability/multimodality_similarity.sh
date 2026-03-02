#!/bin/bash
# This script runs multimodality similarity analysis (Text vs Image)

# --- Configuration Section ---
# Datasets to process
DATASETS=(
    "l11p/MME-test-captioned"
    "l11p/MMBench_EN-dev-captioned"
    "l11p/POPE-test-captioned"
    "l11p/SEED-Bench-images-captioned"
    # "l11p/GQA-val-balanced-noyesno-captioned"
    # "l11p/COCOQA-restval-ICQA"
    # "l11p/VQAv2-COCO-restval-ICQ"
)

# Caption source: "human" for COCOQA / VQAv2-ICQ (which have human captions),
# "synthetic" for everything else. Always passed explicitly so path is deterministic.
CAPTION_SOURCE="${CAPTION_SOURCE:-synthetic}"

# Analysis configuration
TOKENS_POOLING_METHODS=(
    "last"
    # "mean_context"
)
SIMILARITY_MEASURES="neighborhood_overlap linear_cka information_imbalance information_imbalance_reversed information_imbalance_mean"

# Processing parameters
BATCH_SIZE="1" # 1 for MMBench
SEED="42"

# Guide text per dataset (auto-selected in the loop below)
GUIDE_TEXT_DEFAULT="Answer the question using a single word or phrase."
GUIDE_TEXT_MULTICHOICE="Answer with the option's letter from the given choices directly."

# Configuration from environment (set by launch_job.sh)
DOWNSAMPLE_SIZE="${DOWNSAMPLE_SIZE:-}"
DATASET_SPLIT="${DATASET_SPLIT:-}"
REPEATED_IMAGES="${REPEATED_IMAGES:-false}"
ENABLE_SEMANTIC_FILTER="${ENABLE_SEMANTIC_FILTER:-false}"
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$HOME/scratch}"
SAVE_REPRESENTATIONS="${SAVE_REPRESENTATIONS:-true}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-false}"

# Models to evaluate
MODELS=(
    "lbasile/llava_7b"
    "l11p/llava-v1.5-7b-pretrained-lbasile"
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

    # Auto-select guide text based on dataset
    if [[ "$DATASET_PATH_OR_NAME" == *"MMBench"* ]] || [[ "$DATASET_PATH_OR_NAME" == *"SEED"* ]]; then
        GUIDE_TEXT="$GUIDE_TEXT_MULTICHOICE"
    else
        GUIDE_TEXT="$GUIDE_TEXT_DEFAULT"
    fi
    echo "📝 Guide text: $GUIDE_TEXT"

    # Detect dataset type for conditional flag application
    IS_MC_DATASET=false
    IS_OPENVQA_DATASET=false
    if [[ "$DATASET_PATH_OR_NAME" == *"MME"* ]] || \
       [[ "$DATASET_PATH_OR_NAME" == *"MMBench"* ]] || \
       [[ "$DATASET_PATH_OR_NAME" == *"POPE"* ]] || \
       [[ "$DATASET_PATH_OR_NAME" == *"SEED"* ]]; then
        IS_MC_DATASET=true
    elif [[ "$DATASET_PATH_OR_NAME" == *"COCOQA"* ]] || \
         [[ "$DATASET_PATH_OR_NAME" == *"VQAv2"* ]]; then
        IS_OPENVQA_DATASET=true
    fi

    for MODEL_NAME in "${MODELS[@]}"; do
        echo -e "\n----------------------------------------------------------------"
        echo "🤖 Processing Model: $MODEL_NAME"
        echo "----------------------------------------------------------------"

        # Activate model-specific venv
        VENV="$PROJECT_DIR/.venv"
        source "$VENV/bin/activate"

        for POOLING_METHOD in "${TOKENS_POOLING_METHODS[@]}"; do
            echo -e "\n  ── Pooling: $POOLING_METHOD"

            # Build command with conditional flags
            CMD_ARGS=(
                --results-dir "$RESULTS_DIR"
                --representations-dir "$REPRESENTATIONS_DIR"
                --dataset-path-or-name "$DATASET_PATH_OR_NAME"
                --model-name-or-path "$MODEL_NAME"
                --batch-size "$BATCH_SIZE"
                --seed "$SEED"
                --guide-text "$GUIDE_TEXT"
                --caption-source "$CAPTION_SOURCE"
                --pooling-methods "$POOLING_METHOD"
                --similarity-measures ${SIMILARITY_MEASURES}
            )

            # Only pass --downsample-size when explicitly set (empty = no downsampling)
            if [ -n "$DOWNSAMPLE_SIZE" ]; then
                CMD_ARGS+=(--downsample-size "$DOWNSAMPLE_SIZE")
            fi

            if [ "$SAVE_REPRESENTATIONS" = "true" ]; then
                CMD_ARGS+=(--save-representations)
            fi

            if [ "$ENABLE_SEMANTIC_FILTER" = "true" ]; then
                CMD_ARGS+=(--enable-semantic-filter)
            fi

            if [ "$FORCE_RECOMPUTE" = "true" ]; then
                CMD_ARGS+=(--force)
            fi

            # Apply --repeated-images only to MC datasets
            if [ "$REPEATED_IMAGES" = "true" ] && [ "$IS_MC_DATASET" = "true" ]; then
                CMD_ARGS+=(--repeated-images)
            fi

            # Apply --dataset-split only to OpenVQA datasets
            if [ -n "$DATASET_SPLIT" ] && [ "$IS_OPENVQA_DATASET" = "true" ]; then
                CMD_ARGS+=(--dataset-split "$DATASET_SPLIT")
            fi

            python3 "$FILEPATH" "${CMD_ARGS[@]}"

            echo "  ✅ Completed: $MODEL_NAME | $POOLING_METHOD"
        done
    done
done

echo ""
echo "====== All Analyses Complete ======"
