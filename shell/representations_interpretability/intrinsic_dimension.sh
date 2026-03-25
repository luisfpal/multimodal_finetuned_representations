#!/bin/bash
# This script runs intrinsic-dimension analysis on paired text/image representations.

set -e

# --- Configuration Section ---
DATASETS=(
    # "l11p/MME-test-captioned"
    # "l11p/POPE-test-captioned"
    "l11p/MMBench_EN-dev-captioned"
    "l11p/SEED-Bench-images-captioned"
    # "l11p/GQA-val-balanced-noyesno-captioned"
    # "l11p/COCOQA-restval-ICQA"
    # "l11p/VQAv2-COCO-restval-ICQ"
)

# Human captions only apply to COCOQA/VQAv2; overridden per dataset below.
CAPTION_SOURCE="synthetic"

TOKENS_POOLING_METHODS=(
    "last"
    "mean_context"
    "mean_image_last_caption"
    # "last_image_last_caption"
)

ID_ALGORITHM="scaling_gride"  # scaling_gride | MLE | twoNN
ID_NN_RANK="16"
ID_NN_RANGE_MAX="100"
ID_FRACTION="0.9"

# Processing parameters
BATCH_SIZE="1"
SEED="42"

GUIDE_TEXT_OPENENDED="Answer the question using a single word or phrase."
GUIDE_TEXT_MULTICHOICE="Answer with the option's letter from the given choices directly."

MODELS=(
    "lbasile/llava_7b"
    "l11p/llava-v1.5-7b-pretrained-lbasile"
    # "OpenGVLab/InternVL2-8B"
    # "OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"
    # "lbasile/llava_13b"
    # "l11p/llava-v1.5-13b-pretrained-lbasile"
)

# Configuration from environment (set by launch_job.sh)
DOWNSAMPLE_SIZE="${DOWNSAMPLE_SIZE:-}"
DATASET_SPLIT="${DATASET_SPLIT:-}"
REPEATED_IMAGES="${REPEATED_IMAGES:-false}"
OPEN_ENDED="${OPEN_ENDED:-false}"
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$HOME/scratch}"
SAVE_REPRESENTATIONS="${SAVE_REPRESENTATIONS:-true}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-false}"
DRY_RUN="${DRY_RUN:-false}"

PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"

FILENAME="intrinsic_dimension.py"
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/$FILENAME"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')

echo "====== Intrinsic Dimension Analysis ======"
echo "Results directory: $RESULTS_DIR"
echo "Started at: $START_HUMAN"

for POOLING_METHOD in "${TOKENS_POOLING_METHODS[@]}"; do
    echo -e "\n════════════════════════════════════════════════════════════════"
    echo "⚙️  Pooling Method: $POOLING_METHOD"
    echo "════════════════════════════════════════════════════════════════"

    for MODEL_NAME in "${MODELS[@]}"; do
        echo -e "\n────────────────────────────────────────────────────────────────"
        echo "🤖 Processing Model: $MODEL_NAME"
        echo "────────────────────────────────────────────────────────────────"

        VENV="$PROJECT_DIR/.venv"
        source "$VENV/bin/activate"

        for DATASET_PATH_OR_NAME in "${DATASETS[@]}"; do
            echo -e "\n  🌍 Dataset: $DATASET_PATH_OR_NAME"

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

            if [[ "$DATASET_PATH_OR_NAME" == *"COCOQA"* ]] || \
               [[ "$DATASET_PATH_OR_NAME" == *"VQAv2"* ]]; then
                IS_OPENVQA_SPLIT_CAPABLE=true
            fi

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

            EFFECTIVE_CAPTION_SOURCE="$CAPTION_SOURCE"
            if [[ "$DATASET_PATH_OR_NAME" != *"COCOQA"* ]] && [[ "$DATASET_PATH_OR_NAME" != *"VQAv2"* ]]; then
                EFFECTIVE_CAPTION_SOURCE="synthetic"
            fi

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
                --id-algorithm "$ID_ALGORITHM"
                --id-nn-rank "$ID_NN_RANK"
                --id-nn-range-max "$ID_NN_RANGE_MAX"
                --id-fraction "$ID_FRACTION"
            )

            if [ -n "$DOWNSAMPLE_SIZE" ]; then
                CMD_ARGS+=(--downsample-size "$DOWNSAMPLE_SIZE")
            fi

            if [ "$SAVE_REPRESENTATIONS" = "true" ]; then
                CMD_ARGS+=(--save-representations)
            fi

            if [ "$FORCE_RECOMPUTE" = "true" ]; then
                CMD_ARGS+=(--force)
            fi

            if [ "$DRY_RUN" = "true" ]; then
                CMD_ARGS+=(--dry-run)
            fi

            if [ "$REPEATED_IMAGES" = "true" ] && [ "$IS_MC_DATASET" = "true" ]; then
                CMD_ARGS+=(--repeated-images)
            fi

            if [ "$OPEN_ENDED" = "true" ] && [ "$IS_MC_OPENENDED_CAPABLE" = "true" ]; then
                CMD_ARGS+=(--open-ended)
            fi

            if [ -n "$DATASET_SPLIT" ] && [ "$IS_OPENVQA_SPLIT_CAPABLE" = "true" ]; then
                CMD_ARGS+=(--dataset-split "$DATASET_SPLIT")
            fi

            python3 "$FILEPATH" "${CMD_ARGS[@]}"

            echo "    ✅ Done: $MODEL_NAME | $POOLING_METHOD | $DATASET_PATH_OR_NAME"
        done
    done
done

END_EPOCH=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
ELAPSED_SECONDS=$((END_EPOCH - START_EPOCH))
ELAPSED_HMS=$(printf '%02d:%02d:%02d' \
    $((ELAPSED_SECONDS / 3600)) \
    $(((ELAPSED_SECONDS % 3600) / 60)) \
    $((ELAPSED_SECONDS % 60)))

echo ""
echo "====== Intrinsic Dimension Analysis Complete ======"
echo "Finished at: $END_HUMAN"
echo "Elapsed: $ELAPSED_HMS"
