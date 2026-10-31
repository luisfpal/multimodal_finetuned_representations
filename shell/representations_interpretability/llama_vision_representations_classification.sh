#!/bin/bash
# LLaMA Vision 3.2 label-overlap experiment.
#
# Loads the model ONCE and runs all 7 image-classification datasets.
# Each dataset is associated with the correct MCQ filter prompt mode.
#
# Submit via launch_job.sh:
#   Set SCRIPT_NAME="llama_vision_representations_classification.sh", then:
#   bash shell/representations_interpretability/launch_job.sh

set -euo pipefail

# Hardcoded path: edit this path for your setup.
MODEL_PATH="/orfeo/cephfs/scratch/area/ddoimo/models/hub/Meta-Llama-3-90B-instruct"

# Each entry: "dataset_name|split|prompt_mode"
# Prompt mode is hardcoded per dataset:
#   mini-imagenet   → mcq_nocat_filter      (category-free MCQ, filtered to n classes)
#   food101, sun397, Places365, dtd → generic_mcq_filter       (random n classes)
#   caltech101, oxford_flowers102   → generic_mcq_top_filter   (top-n by count; imbalanced datasets)
DATASET_RUNS=(
    "timm/mini-imagenet|validation|mcq_nocat_filter"
    "ethz/food101|validation|generic_mcq_filter"
    "tanganke/sun397|test|generic_mcq_filter"
    "flwrlabs/caltech101|train|generic_mcq_top_filter"
    "dpdl-benchmark/oxford_flowers102|test|generic_mcq_top_filter"
    "torch-uncertainty/Places365|train|generic_mcq_filter"
    "tanganke/dtd|train|generic_mcq_filter"
)

# ── Experiment settings ───────────────────────────────────────────────────────
N_CHOICES=10
CLASS_FRACTION=0.3

# ── Environment ───────────────────────────────────────────────────────────────
# Hardcoded path: edit this path for your setup.
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$HOME/scratch}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-false}"
DRY_RUN="${DRY_RUN:-false}"

PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/llama_vision_representations_classification.py"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$PROJECT_DIR/.venv/bin/activate"

START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')

echo "====== LLaMA Vision Representations Classification ======"
echo "Model:       $MODEL_PATH"
echo "Datasets:    ${#DATASET_RUNS[@]}"
echo "n_choices:   $N_CHOICES"
echo "cf:          $CLASS_FRACTION"
echo "Reprs dir:   $REPRESENTATIONS_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Started at:  $START_HUMAN"
echo ""

CMD_ARGS=(
    --model-path "$MODEL_PATH"
    --dataset-runs "${DATASET_RUNS[@]}"
    --n-choices "$N_CHOICES"
    --class-fraction "$CLASS_FRACTION"
    --representations-dir "$REPRESENTATIONS_DIR"
    --results-dir "$RESULTS_DIR"
    --save-representations
)

[ "$FORCE_RECOMPUTE" = "true" ] && CMD_ARGS+=(--force-recompute-representations --force-recompute-metrics)

if [ "$DRY_RUN" = "true" ]; then
    echo "DRY RUN:"
    printf '%q ' "$PYTHON_BIN" "$FILEPATH" "${CMD_ARGS[@]}"
    printf '\n'
else
    "$PYTHON_BIN" "$FILEPATH" "${CMD_ARGS[@]}"
fi

END_EPOCH=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
ELAPSED_SECONDS=$((END_EPOCH - START_EPOCH))
ELAPSED_HMS=$(printf '%02d:%02d:%02d' \
    $((ELAPSED_SECONDS / 3600)) \
    $(((ELAPSED_SECONDS % 3600) / 60)) \
    $((ELAPSED_SECONDS % 60)))

echo ""
echo "====== LLaMA Vision Complete ======"
echo "Finished at: $END_HUMAN"
echo "Elapsed:     $ELAPSED_HMS"
