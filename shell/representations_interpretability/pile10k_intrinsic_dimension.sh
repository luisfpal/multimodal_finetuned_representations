#!/bin/bash
# Compute intrinsic dimension for pile-10k text-only representations.
# Representations must be pre-extracted via pile10k_representations.py.

set -e

# ── Models (same 12 as intrinsic_dimension.sh) ───────────────────────────────
MODELS=(
    "lbasile/llava_7b"
    # "l11p/llava-v1.5-7b-pretrained-lbasile"
    # "OpenGVLab/InternVL2-8B"
    # "OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"
    # "lbasile/llava_13b"
    # "l11p/llava-v1.5-13b-pretrained-lbasile"
    # "nyu-visionx/cambrian-8b"
    # "lbasile/cambrian-8b-pretrained"
    # "lmms-lab/LLaVA-OneVision-1.5-4B-Base"
    # "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    # "lmms-lab/LLaVA-OneVision-1.5-8B-Base"
    # "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
)

# ── ID settings (mirror intrinsic_dimension.sh defaults) ─────────────────────
POOLING_METHODS="last"
MAX_TOKENS=128
MIN_TOKENS=64
ID_ALGORITHM="scaling_gride"
ID_NN_RANK="8"
ID_NN_RANGE_MAX="100"
ID_FRACTION="0.9"
PROMPT_FORMAT="plain"
REMOVE_ASSISTANT_TOKENS="true"
RESIDUAL_STREAM_TYPE="output_layer"

# ── Environment ───────────────────────────────────────────────────────────────
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$HOME/scratch}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-true}"
DRY_RUN="${DRY_RUN:-false}"

PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/pile10k_intrinsic_dimension.py"

source "$PROJECT_DIR/.venv/bin/activate"

START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')

echo "====== pile-10k Intrinsic Dimension ======"
echo "Representations dir: $REPRESENTATIONS_DIR"
echo "Results dir:         $RESULTS_DIR"
echo "Pooling methods:     $POOLING_METHODS"
echo "Token range:         tokens_${MIN_TOKENS}_${MAX_TOKENS}"
echo "Algorithm:           $ID_ALGORITHM (k=$ID_NN_RANK, range_max=$ID_NN_RANGE_MAX, fraction=$ID_FRACTION)"
echo "Prompt format:       $PROMPT_FORMAT"
echo "Remove asst tokens:  $REMOVE_ASSISTANT_TOKENS"
echo "Residual stream:     $RESIDUAL_STREAM_TYPE"
echo "Started at:          $START_HUMAN"

for MODEL_NAME in "${MODELS[@]}"; do
    echo -e "\n════════════════════════════════════════════════════════════════"
    echo "🤖 Model: $MODEL_NAME"
    echo "════════════════════════════════════════════════════════════════"

    CMD_ARGS=(
        --representations-dir   "$REPRESENTATIONS_DIR"
        --results-dir           "$RESULTS_DIR"
        --model-name-or-path    "$MODEL_NAME"
        --pooling-methods       $POOLING_METHODS
        --max-tokens            "$MAX_TOKENS"
        --min-tokens            "$MIN_TOKENS"
        --id-algorithm          "$ID_ALGORITHM"
        --id-nn-rank            "$ID_NN_RANK"
        --id-nn-range-max       "$ID_NN_RANGE_MAX"
        --id-fraction           "$ID_FRACTION"
        --prompt-format         "$PROMPT_FORMAT"
        --residual-stream-type  "$RESIDUAL_STREAM_TYPE"
    )

    [ "$FORCE_RECOMPUTE" = "true" ]         && CMD_ARGS+=(--force)
    [ "$DRY_RUN" = "true" ]                 && CMD_ARGS+=(--dry-run)
    [ "$REMOVE_ASSISTANT_TOKENS" = "true" ] && CMD_ARGS+=(--remove-assistant-tokens)

    python3 "$FILEPATH" "${CMD_ARGS[@]}"

    echo "    ✅ Done: $MODEL_NAME"
done

END_EPOCH=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
ELAPSED_SECONDS=$((END_EPOCH - START_EPOCH))
ELAPSED_HMS=$(printf '%02d:%02d:%02d' \
    $((ELAPSED_SECONDS / 3600)) \
    $(((ELAPSED_SECONDS % 3600) / 60)) \
    $((ELAPSED_SECONDS % 60)))

echo ""
echo "====== pile-10k Intrinsic Dimension Complete ======"
echo "Finished at: $END_HUMAN"
echo "Elapsed:     $ELAPSED_HMS"
