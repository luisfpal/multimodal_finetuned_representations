#!/bin/bash
# Runs representations classification metrics on LLaVA residual stream representations.
set -euo pipefail

MODELS=(
    "lbasile/llava_7b"
    # "l11p/llava-v1.5-7b-pretrained-lbasile"
    # "lbasile/llava_7b_frozen"
    # "lbasile/llava_7b_0_16"
    # "lbasile/llava_7b_16_32"
    # "lbasile/llava_7b_6_16"
    # "lbasile/llava_7b_6_17_complementary"

    # "lbasile/llava_13b"
    # "l11p/llava-v1.5-13b-pretrained-lbasile"
    # "lbasile/llava_13b_6_21"
    # "lbasile/llava_13b_6_21_complementary"

    # "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    # "lmms-lab/LLaVA-OneVision-1.5-4B-Base"

    # "OpenGVLab/InternVL2-8B"
    # "OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"

    # "nyu-visionx/cambrian-8b"
    # "lbasile/cambrian-8b-pretrained"
)

DATASETS=(
    # "derek-thomas/ScienceQA"           # split: test
    # "lmms-lab/MMMU"                    # split: test
    "timm/mini-imagenet"                  # split: validation (10k, 100 cls × 100 imgs)
)

TOKENS_POOLING_METHODS=(
    "last"
    # "mean_image"
)

# Select any subset, e.g. ("logistic_regression" "knn")
METRICS=(
    "logistic_regression"
    "label_overlap"
    "knn"
    "homogeneity"
)

# Set split to match the dataset: "test" for ScienceQA/MMMU, "validation" for mini-imagenet
SPLIT="validation"
MAX_SAMPLES=""
# For mini-imagenet: guide text is auto-cleared in the Python script (captioning, not MCQ)
# For ScienceQA/MMMU: use "Answer with the option's letter from the given choices directly."
GUIDE_TEXT="Answer with the option's letter from the given choices directly."
SAVE_REPRESENTATIONS="true"
FORCE_RECOMPUTE="true"
SAVE_PROMPTS="false"
# MMMU label granularity: "intermediate" (subjects, ~30 classes) or "detailed" (subfields >= mean count)
MMMU_LABEL_GRANULARITY="detailed"
# In-context learning shots from MMMU dev split (0 = zero-shot)
N_SHOTS=0

PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$HOME/scratch}"
PY_SCRIPT="$PROJECT_DIR/scripts/representations_interpretability/representations_classification.py"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')

echo "====== Representations Classification ======"
echo "Results directory: $RESULTS_DIR"
echo "Representations directory: $REPRESENTATIONS_DIR"
echo "Metrics: ${METRICS[*]}"
echo "Pooling methods: ${TOKENS_POOLING_METHODS[*]}"
echo "MMMU label granularity: $MMMU_LABEL_GRANULARITY"
echo "ICL shots: $N_SHOTS"
echo "Save prompts: $SAVE_PROMPTS"
echo "Started at: $START_HUMAN"

for MODEL_NAME in "${MODELS[@]}"; do
    echo -e "\n────────────────────────────────────────────────────────────────"
    echo "Processing Model: $MODEL_NAME"
    echo "────────────────────────────────────────────────────────────────"

    for DATASET_NAME in "${DATASETS[@]}"; do
        echo -e "\n  Dataset: $DATASET_NAME"
        for TOKENS_POOLING_METHOD in "${TOKENS_POOLING_METHODS[@]}"; do
            echo "    Pooling: $TOKENS_POOLING_METHOD"

            CMD_ARGS=(
                --model-name "$MODEL_NAME"
                --dataset-name "$DATASET_NAME"
                --split "$SPLIT"
                --guide-text "$GUIDE_TEXT"
                --results-dir "$RESULTS_DIR"
                --representations-dir "$REPRESENTATIONS_DIR"
                --tokens-pooling-method "$TOKENS_POOLING_METHOD"
                --metrics "${METRICS[@]}"
                --mmmu-label-granularity "$MMMU_LABEL_GRANULARITY"
                --n-shots "$N_SHOTS"
            )

            if [ "$SAVE_REPRESENTATIONS" = "true" ]; then
                CMD_ARGS+=(--save-representations)
            else
                CMD_ARGS+=(--no-save-representations)
            fi

            if [ "$FORCE_RECOMPUTE" = "true" ]; then
                CMD_ARGS+=(--force-recompute-representations)
            fi

            if [ -n "$MAX_SAMPLES" ]; then
                CMD_ARGS+=(--max-samples "$MAX_SAMPLES")
            fi

            if [ "$SAVE_PROMPTS" = "true" ]; then
                CMD_ARGS+=(--save-prompts)
            else
                CMD_ARGS+=(--no-save-prompts)
            fi

            "$PYTHON_BIN" "$PY_SCRIPT" "${CMD_ARGS[@]}"

            echo "    Done: $MODEL_NAME | $DATASET_NAME | $TOKENS_POOLING_METHOD"
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
echo "====== All Analyses Complete ======"
echo "Finished at: $END_HUMAN"
echo "Elapsed: $ELAPSED_HMS"
