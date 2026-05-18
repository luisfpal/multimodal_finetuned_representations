#!/bin/bash
# Llama 3.2 Vision 90B — text-vs-image similarity analysis
#
# Calls the Python script ONCE with all dataset specs so the 90B model
# is loaded a single time (~15 min) rather than once per dataset.
#
# SLURM launch: update launch_job.sh with:
#   ACCOUNT="lade" | PARTITION="H100" | GPU_TYPE="H100" | NUM_GPUS=3
#   MEM="256GB" | CPUS=16 | TIME="08:00:00"
#   SCRIPT_NAME="llama_vision_similarity.sh"
# Then: bash shell/representations_interpretability/launch_job.sh
set -e

# =============================================================================
# Configuration
# =============================================================================

MODEL_PATHS=(
    # "/orfeo/cephfs/scratch/area/ddoimo/models/hub/Meta-Llama-3-90B-instruct"
    "/orfeo/cephfs/scratch/area/ddoimo/models/hub/Meta-Llama-3-90B"
)

# Dataset run specs passed directly to the Python script.
# Format: "dataset_path|caption_source|dataset_split|downsample_size|repeated_images|open_ended"
DATASET_RUNS=(
    # "l11p/MMBench_EN-dev-captioned|synthetic||2000|false|true"
    # "l11p/SEED-Bench-images-captioned|synthetic||2000|false|true"
    # "l11p/GQA-val-balanced-noyesno-captioned|synthetic||2000|false|true"
    # "l11p/COCOQA-restval-ICQA|human||2000|false|true"
    # "l11p/VQAv2-COCO-restval-ICQ|human||2000|false|true"
    # "l11p/MMBench_EN-dev-captioned|synthetic|||false|true"
    # "l11p/SEED-Bench-images-captioned|synthetic|||false|true"
    # "l11p/GQA-val-balanced-noyesno-captioned|synthetic|||false|true"
    "l11p/COCOQA-restval-ICQA|human|split01||false|true"
    "l11p/VQAv2-COCO-restval-ICQ|human|split01||false|true"
)

# SIMILARITY_MEASURES="linear_cka neighborhood_overlap information_imbalance information_imbalance_reversed"
SIMILARITY_MEASURES="linear_cka information_imbalance information_imbalance_reversed"
GUIDE_TEXT="Answer the question using a single word or phrase."

BATCH_SIZE="1"
SEED="42"

# Run control flags
SAVE_REPRESENTATIONS="true"
FORCE_RECOMPUTE="false"
FORCE_METRICS_ONLY="false"
REPRESENTATIONS_ONLY="true"

if [ "$FORCE_RECOMPUTE" = "true" ] && [ "$FORCE_METRICS_ONLY" = "true" ]; then
    echo "❌ FORCE_RECOMPUTE and FORCE_METRICS_ONLY cannot both be true."
    exit 1
fi

# Directory configuration
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results/multimodality_comparison"
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$HOME/scratch/representations}"

FILENAME="llama_vision_similarity.py"
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/$FILENAME"

# =============================================================================
# Environment Setup
# =============================================================================

export PYTORCH_ALLOC_CONF=expandable_segments:True

VENV="$PROJECT_DIR/.venv"
source "$VENV/bin/activate"

# =============================================================================
# Run Analysis — single Python call, model loaded once for all datasets
# =============================================================================

START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')

echo "====== Llama Vision 90B — Multimodality Similarity Analysis ======"
echo "Models:   ${#MODEL_PATHS[@]}"
echo "Results:  $RESULTS_DIR"
echo "Datasets: ${#DATASET_RUNS[@]}"
echo "Started:  $START_HUMAN"

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_START_EPOCH=$(date +%s)
    MODEL_START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')

    echo ""
    echo "----------------------------------------------------------------------"
    echo "Model:   $MODEL_PATH"
    echo "Started: $MODEL_START_HUMAN"
    echo "----------------------------------------------------------------------"

    CMD_ARGS=(
        --results-dir "$RESULTS_DIR"
        --representations-dir "$REPRESENTATIONS_DIR"
        --model-path "$MODEL_PATH"
        --batch-size "$BATCH_SIZE"
        --seed "$SEED"
        --guide-text "$GUIDE_TEXT"
        --similarity-measures ${SIMILARITY_MEASURES}
        --dataset-runs "${DATASET_RUNS[@]}"
    )

    [ "$SAVE_REPRESENTATIONS" = "true" ]  && CMD_ARGS+=(--save-representations)
    [ "$FORCE_RECOMPUTE" = "true" ]       && CMD_ARGS+=(--force)
    [ "$FORCE_METRICS_ONLY" = "true" ]    && CMD_ARGS+=(--force-metrics-only)
    [ "$REPRESENTATIONS_ONLY" = "true" ]  && CMD_ARGS+=(--representations-only)

    # One Python call per model: model loaded once, then reused across all datasets.
    python3 "$FILEPATH" "${CMD_ARGS[@]}"

    MODEL_END_EPOCH=$(date +%s)
    MODEL_ELAPSED_SECONDS=$((MODEL_END_EPOCH - MODEL_START_EPOCH))
    MODEL_ELAPSED_HMS=$(printf '%02d:%02d:%02d' \
        $((MODEL_ELAPSED_SECONDS / 3600)) \
        $(((MODEL_ELAPSED_SECONDS % 3600) / 60)) \
        $((MODEL_ELAPSED_SECONDS % 60)))

    echo ""
    echo "Model complete: $MODEL_PATH"
    echo "Elapsed:        $MODEL_ELAPSED_HMS"
done

END_EPOCH=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
ELAPSED_SECONDS=$((END_EPOCH - START_EPOCH))
ELAPSED_HMS=$(printf '%02d:%02d:%02d' \
    $((ELAPSED_SECONDS / 3600)) \
    $(((ELAPSED_SECONDS % 3600) / 60)) \
    $((ELAPSED_SECONDS % 60)))

echo ""
echo "====== Analysis Complete ======"
echo "Finished: $END_HUMAN"
echo "Elapsed:  $ELAPSED_HMS"
