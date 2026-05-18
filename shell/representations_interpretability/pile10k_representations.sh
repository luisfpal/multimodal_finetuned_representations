#!/bin/bash
# Extract residual-stream representations for NeelNanda/pile-10k (text-only).
#
# Two pooling methods per model:
#   last          — last token of the full prompt sequence
#   last_user_text — last token of {text_context} in the structured prompt
#
# Representations are cached in $REPRESENTATIONS_DIR and reused if they exist.
# Use FORCE_RECOMPUTE=true to bypass the cache.
#
# Submit via launch_job.sh:
#   Set SCRIPT_NAME="pile10k_representations.sh" in launch_job.sh, then:
#   bash shell/representations_interpretability/launch_job.sh

set -e

# ── Models (same 12 as intrinsic_dimension.sh) ───────────────────────────────
MODELS=(
    "lbasile/llava_7b"
    "l11p/llava-v1.5-7b-pretrained-lbasile"
    "lbasile/llava_13b"
    "l11p/llava-v1.5-13b-pretrained-lbasile"
    "OpenGVLab/InternVL2-8B"
    "OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"

    "nyu-visionx/cambrian-8b"
    "lbasile/cambrian-8b-pretrained"
    "lmms-lab/LLaVA-OneVision-1.5-4B-Base"
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Base"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
)

# ── Extraction settings ───────────────────────────────────────────────────────
POOLING_METHODS="last"
MAX_TOKENS=64
MIN_TOKENS=64
BATCH_SIZE=1
PROMPT_FORMAT="raw"
RESIDUAL_STREAM_TYPE="output_layer"
REMOVE_ASSISTANT_TOKENS="false"
# RESIDUAL_STREAM_TYPE="post_input_layernorm"

# ── Environment (override via env vars or set by launch_job.sh) ──────────────
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$HOME/scratch}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-false}"
DRY_RUN="${DRY_RUN:-false}"

PROJECT_DIR="$HOME/multimodal_finetuned_representations"
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/pile10k_representations.py"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$PROJECT_DIR/.venv/bin/activate"

START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')

echo "====== pile-10k Representation Extraction ======"
echo "Representations dir: $REPRESENTATIONS_DIR"
echo "Pooling methods:     $POOLING_METHODS"
echo "Max tokens:          $MAX_TOKENS"
echo "Min tokens:          $MIN_TOKENS"
echo "Batch size:          $BATCH_SIZE"
echo "Prompt format:       $PROMPT_FORMAT"
echo "Remove asst tokens:  $REMOVE_ASSISTANT_TOKENS"
echo "Residual stream:     $RESIDUAL_STREAM_TYPE"
echo "Started at:          $START_HUMAN"

for MODEL_NAME in "${MODELS[@]}"; do
    echo -e "\n════════════════════════════════════════════════════════════════"
    echo "🤖 Model: $MODEL_NAME"
    echo "════════════════════════════════════════════════════════════════"

    CMD_ARGS=(
        --model-name-or-path    "$MODEL_NAME"
        --representations-dir   "$REPRESENTATIONS_DIR"
        --pooling-methods       $POOLING_METHODS
        --max-tokens            "$MAX_TOKENS"
        --min-tokens            "$MIN_TOKENS"
        --batch-size            "$BATCH_SIZE"
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
echo "====== pile-10k Extraction Complete ======"
echo "Finished at: $END_HUMAN"
echo "Elapsed:     $ELAPSED_HMS"
