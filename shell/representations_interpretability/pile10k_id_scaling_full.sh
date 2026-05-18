#!/bin/bash
# Compute per-layer ID-scaling profiles (all GRIDE scales) for pile-10k.
#
# Output: results/intrinsic_dimension/pile10k_train_txt/.../txt_scaling_gride_full.safetensors
#         shape (num_layers, n_scales) — does NOT overwrite txt_scaling_gride.safetensors
#
# Submit via launch_job.sh:
#   Set SCRIPT_NAME="pile10k_id_scaling_full.sh" in launch_job.sh, then:
#   bash shell/representations_interpretability/launch_job.sh

set -e

# ── Models ────────────────────────────────────────────────────────────────────
MODELS=(
    # "lbasile/llava_7b"
    # "l11p/llava-v1.5-7b-pretrained-lbasile"
    # "lbasile/llava_13b"
    # "l11p/llava-v1.5-13b-pretrained-lbasile"
    # "OpenGVLab/InternVL2-8B"
    # "OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"
    # "nyu-visionx/cambrian-8b"
    # "lbasile/cambrian-8b-pretrained"
    # "lmms-lab/LLaVA-OneVision-1.5-4B-Base"
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Base"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
)

# ── Settings — must match the representation extraction job ───────────────────
POOLING_METHODS="last"
MAX_TOKENS=64
MIN_TOKENS=64
PROMPT_FORMAT="raw"
REMOVE_ASSISTANT_TOKENS="false"
RESIDUAL_STREAM_TYPE="output_layer"

# ── GRIDE hyperparameter ──────────────────────────────────────────────────────
RANGE_MAX=550

# ── Environment ───────────────────────────────────────────────────────────────
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$HOME/scratch}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-false}"
DRY_RUN="${DRY_RUN:-false}"

PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/pile10k_id_scaling_full.py"

source "$PROJECT_DIR/.venv/bin/activate"

START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')

echo "====== pile-10k ID Scaling Full ======"
echo "Representations dir: $REPRESENTATIONS_DIR"
echo "Results dir:         $RESULTS_DIR"
echo "Pooling methods:     $POOLING_METHODS"
echo "Token range:         tokens_${MIN_TOKENS}_${MAX_TOKENS}"
echo "Prompt format:       $PROMPT_FORMAT"
echo "Remove asst tokens:  $REMOVE_ASSISTANT_TOKENS"
echo "Residual stream:     $RESIDUAL_STREAM_TYPE"
echo "GRIDE range_max:     $RANGE_MAX"
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
        --prompt-format         "$PROMPT_FORMAT"
        --residual-stream-type  "$RESIDUAL_STREAM_TYPE"
        --range-max             "$RANGE_MAX"
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
echo "====== pile-10k ID Scaling Full Complete ======"
echo "Finished at: $END_HUMAN"
echo "Elapsed:     $ELAPSED_HMS"
