#!/bin/bash
# Runs representations classification metrics on LLaVA residual stream representations.

MODELS=(
    "lbasile/llava_7b"
    "lbasile/llava_7b_frozen"
    # "lbasile/llava_7b_0_16"
    # "lbasile/llava_7b_16_32"
    # "lbasile/llava_7b_6_16"
    # "lbasile/llava_7b_6_17_complementary"
    # "lbasile/llava_13b"
    # "lbasile/llava_13b_6_21"
    # "lbasile/llava_13b_6_21_complementary"
    # "l11p/llava-v1.5-7b-pretrained-lbasile"
    # "l11p/llava-v1.5-13b-pretrained-lbasile"
)

DATASETS=(
    "derek-thomas/ScienceQA"
)

# Select any subset, e.g. ("logistic_regression" "knn")
METRICS=(
    # "logistic_regression"
    # "label_overlap"
    "knn"
)

SPLIT="test"
MAX_SAMPLES=""
BACKEND="sklearn"  # Used only for logistic_regression
GUIDE_TEXT="Answer with the option's letter from the given choices directly."

PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"
PY_SCRIPT="$PROJECT_DIR/scripts/representations_interpretability/representations_classification.py"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')

echo "====== Representations Classification ======"
echo "Results directory: $RESULTS_DIR"
echo "Metrics: ${METRICS[*]}"
echo "Started at: $START_HUMAN"

for MODEL_NAME in "${MODELS[@]}"; do
    echo -e "\n────────────────────────────────────────────────────────────────"
    echo "Processing Model: $MODEL_NAME"
    echo "────────────────────────────────────────────────────────────────"

    for DATASET_NAME in "${DATASETS[@]}"; do
        echo -e "\n  Dataset: $DATASET_NAME"

        CMD_ARGS=(
            --model-name "$MODEL_NAME"
            --dataset-name "$DATASET_NAME"
            --split "$SPLIT"
            --guide-text "$GUIDE_TEXT"
            --results-dir "$RESULTS_DIR"
            --backend "$BACKEND"
            --metrics "${METRICS[@]}"
        )

        if [ -n "$MAX_SAMPLES" ]; then
            CMD_ARGS+=(--max-samples "$MAX_SAMPLES")
        fi

        "$PYTHON_BIN" "$PY_SCRIPT" "${CMD_ARGS[@]}"

        echo "    Done: $MODEL_NAME | $DATASET_NAME"
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
