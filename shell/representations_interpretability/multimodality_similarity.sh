#!/bin/bash
# This script runs multimodality similarity analysis (Text vs Image)

# --- Configuration Section ---
# Datasets to process
DATASETS=(
    # "l11p/MME-test-captioned"
    # "l11p/POPE-test-captioned"
    # "l11p/MMBench_EN-dev-captioned"
    # "l11p/SEED-Bench-images-captioned"
    # "l11p/GQA-val-balanced-noyesno-captioned"
    "l11p/COCOQA-restval-ICQA"
    "l11p/VQAv2-COCO-restval-ICQ"
)

# Caption source: "human" or "synthetic" for COCOQA / VQAv2-ICQ (which have human captions),
# "synthetic" for everything else. Always passed explicitly so path is deterministic.
CAPTION_SOURCE="human"

# Analysis configuration
TOKENS_POOLING_METHODS=(
    "last"
    "mean_context"
    "mean_image_last_caption"
    # "last_image_last_caption"
)
SIMILARITY_MEASURES="neighborhood_overlap linear_cka information_imbalance information_imbalance_reversed information_imbalance_mean"
SIMILARITY_MEASURES="${SIMILARITY_MEASURES} linear_regression_r2 linear_regression_r2_reversed"

# Processing parameters
BATCH_SIZE="1" # 1 for MMBench
SEED="42"
REGRESSION_TYPE="ols" # Valid: ols | ridge | lasso

# Guide text per dataset (auto-selected in the loop below)
GUIDE_TEXT_OPENENDED="Answer the question using a single word or phrase."
GUIDE_TEXT_MULTICHOICE="Answer with the option's letter from the given choices directly."

# Models to evaluate
MODELS=(
    "lbasile/llava_7b"
    # "l11p/llava-v1.5-7b-pretrained-lbasile"
    # "OpenGVLab/InternVL2-8B"  # Finetuned
    # "OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"  # Pretrained
    # "lbasile/llava_13b"
    # "l11p/llava-v1.5-13b-pretrained-lbasile"
    # "lbasile/LLaVA-OneVision-1.5-8B-stage1"
    # "lbasile/LLaVA-OneVision-1.5-8B-stage2-2600"
    # "lmms-lab/LLaVA-OneVision-1.5-8B-Base"
    # "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    # "lbasile/llava_7b_6_17"
    # "lbasile/llava_7b_6_17_complementary"
    # "OpenGVLab/InternVL3_5-8B-Pretrained"
    # "OpenGVLab/InternVL3_5-8B-Instruct"
    # "OpenGVLab/InternVL3_5-8B-MPO"
    # "OpenGVLab/InternVL3_5-8B"

)

# Configuration from environment (set by launch_job.sh)
DOWNSAMPLE_SIZE="${DOWNSAMPLE_SIZE:-}"
DATASET_SPLIT="${DATASET_SPLIT:-}"
# REPEATED_IMAGES=true  → use raw benchmark with image repetitions (repeated_images/)
# REPEATED_IMAGES=false → use deduplicated variant (unique_images/) — DEFAULT
REPEATED_IMAGES="${REPEATED_IMAGES:-false}"
OPEN_ENDED="${OPEN_ENDED:-false}"
REPRESENTATIONS_ONLY="${REPRESENTATIONS_ONLY:-false}"
ENABLE_SEMANTIC_FILTER="${ENABLE_SEMANTIC_FILTER:-false}"
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$HOME/scratch}"
SAVE_REPRESENTATIONS="${SAVE_REPRESENTATIONS:-true}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-false}"

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

for POOLING_METHOD in "${TOKENS_POOLING_METHODS[@]}"; do
    echo -e "\n════════════════════════════════════════════════════════════════"
    echo "⚙️  Pooling Method: $POOLING_METHOD"
    echo "════════════════════════════════════════════════════════════════"

    for MODEL_NAME in "${MODELS[@]}"; do
        echo -e "\n────────────────────────────────────────────────────────────────"
        echo "🤖 Processing Model: $MODEL_NAME"
        echo "────────────────────────────────────────────────────────────────"

        # Activate model-specific venv
        VENV="$PROJECT_DIR/.venv"
        source "$VENV/bin/activate"

        for DATASET_PATH_OR_NAME in "${DATASETS[@]}"; do
            echo -e "\n  🌍 Dataset: $DATASET_PATH_OR_NAME"

            # Detect dataset type for conditional flag application
            #
            # Three mutually exclusive categories:
            #   IS_OPENVQA_DATASET      — COCOQA, VQAv2, GQA: naturally open-ended, --dataset-split applies
            #   IS_MC_OPENENDED_CAPABLE — MMBench, SEED: MC by default, can be run open-ended via OPEN_ENDED=true
            #   (else)                  — MME, POPE: yes/no MC, OPEN_ENDED has no effect
            #
            # IS_MC_DATASET = IS_MC_OPENENDED_CAPABLE OR yes/no MC (used by --repeated-images / --open-ended gates)
            IS_MC_DATASET=false
            IS_MC_OPENENDED_CAPABLE=false
            IS_OPENVQA_DATASET=false
            IS_OPENVQA_SPLIT_CAPABLE=false
            if [[ "$DATASET_PATH_OR_NAME" == *"MMBench"* ]] || \
               [[ "$DATASET_PATH_OR_NAME" == *"SEED"* ]]; then
                IS_MC_DATASET=true
                IS_MC_OPENENDED_CAPABLE=true
            elif [[ "$DATASET_PATH_OR_NAME" == *"MME"* ]] || \
                 [[ "$DATASET_PATH_OR_NAME" == *"POPE"* ]]; then
                IS_MC_DATASET=true
            elif [[ "$DATASET_PATH_OR_NAME" == *"COCOQA"* ]] || \
                 [[ "$DATASET_PATH_OR_NAME" == *"VQAv2"* ]] || \
                 [[ "$DATASET_PATH_OR_NAME" == *"GQA"* ]]; then
                IS_OPENVQA_DATASET=true
            fi
            # Only COCOQA and VQAv2 support --dataset-split. GQA does not have splits.
            if [[ "$DATASET_PATH_OR_NAME" == *"COCOQA"* ]] || \
               [[ "$DATASET_PATH_OR_NAME" == *"VQAv2"* ]]; then
                IS_OPENVQA_SPLIT_CAPABLE=true
            fi

            # Guide text: derived entirely from the flags above — no string matching needed
            # - OpenVQA (COCOQA, VQAv2, GQA)  → always open-ended
            # - MC open-ended-capable (MMBench, SEED) → open-ended if OPEN_ENDED=true, else multichoice
            # - MC yes/no (MME, POPE)           → always multichoice
            if [ "$IS_OPENVQA_DATASET" = "true" ]; then
                GUIDE_TEXT="$GUIDE_TEXT_OPENENDED"
            elif [ "$IS_MC_OPENENDED_CAPABLE" = "true" ]; then
                if [ "$OPEN_ENDED" = "true" ]; then
                    GUIDE_TEXT="$GUIDE_TEXT_OPENENDED"
                else
                    GUIDE_TEXT="$GUIDE_TEXT_MULTICHOICE"
                fi
            else
                GUIDE_TEXT="$GUIDE_TEXT_MULTICHOICE"
            fi
            echo "    📝 Guide: $GUIDE_TEXT"

            # Determine effective caption source: only COCOQA and VQAv2 support human captions
            # MMBench, SEED, GQA, MME, POPE only support synthetic (force override to prevent errors)
            EFFECTIVE_CAPTION_SOURCE="$CAPTION_SOURCE"
            if [[ "$DATASET_PATH_OR_NAME" != *"COCOQA"* ]] && [[ "$DATASET_PATH_OR_NAME" != *"VQAv2"* ]]; then
                EFFECTIVE_CAPTION_SOURCE="synthetic"
            fi

            # Build command with conditional flags
            CMD_ARGS=(
                --results-dir "$RESULTS_DIR"
                --representations-dir "$REPRESENTATIONS_DIR"
                --dataset-path-or-name "$DATASET_PATH_OR_NAME"
                --model-name-or-path "$MODEL_NAME"
                --batch-size "$BATCH_SIZE"
                --seed "$SEED"
                --guide-text "$GUIDE_TEXT"
                --caption-source "$EFFECTIVE_CAPTION_SOURCE"
                --pooling-methods "$POOLING_METHOD"
                --similarity-measures ${SIMILARITY_MEASURES}
                --regression-type "$REGRESSION_TYPE"
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

            # Apply --open-ended only to MC datasets
            if [ "$OPEN_ENDED" = "true" ] && [ "$IS_MC_OPENENDED_CAPABLE" = "true" ]; then
                CMD_ARGS+=(--open-ended)
            fi

            if [ "$REPRESENTATIONS_ONLY" = "true" ]; then
                CMD_ARGS+=(--representations-only)
            fi

            # Apply --dataset-split only to OpenVQA datasets that support splits (COCOQA, VQAv2)
            if [ -n "$DATASET_SPLIT" ] && [ "$IS_OPENVQA_SPLIT_CAPABLE" = "true" ]; then
                CMD_ARGS+=(--dataset-split "$DATASET_SPLIT")
            fi

            python3 "$FILEPATH" "${CMD_ARGS[@]}"

            echo "    ✅ Done: $MODEL_NAME | $POOLING_METHOD | $DATASET_PATH_OR_NAME"
        done
    done
done

echo ""
echo "====== All Analyses Complete ======"
