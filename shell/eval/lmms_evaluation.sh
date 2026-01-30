#!/bin/bash
set -e  # Exit on error

# ⚠️ ! WARNING: This script not intended for direct execution. ⚠️
# It is launched by 'shell/eval/launch.sh' with proper environment variables.
# You can also use `shell/eval/run_local_evaluation.sh``

# =============================================================================
# TASK SELECTION
# =============================================================================
# Uncomment ONE task to evaluate:
#
# Safe tasks (no external API):
#   scienceqa_img, mme, pope, gqa, textvqa_val, seedbench, vizwiz_vqa_val, vqav2_val
#
# Submission-only (no LLM judge):
#   vizwiz_vqa_test, vqav2_test
#
# LLM judge required (API key loaded from .env):
#   mmvet, mmbench_en_dev, mmbench_cn_dev, llava_in_the_wild
# =============================================================================

# =============================================================================
# EVALUATION TIMING REFERENCE (1 A100 GPU, 40GB VRAM)
# =============================================================================
# ┌─────────────────┬─────────┬─────────────┬─────────────┬─────────────────────┬─────────────┐
# │ Benchmark       │ Samples │ Finetuned   │ Pretrained  │ Collection (~ per)  │ Notes       │
# ├─────────────────┼─────────┼─────────────┼─────────────┼─────────────────────┼─────────────┤
# │ scienceqa_img   │ 2018    │ ~4'         │ ~30'        │ -                   │ Safe        │
# │ mme            │ 2374    │ ~7'         │ ~41'        │ ~4h25m              │ Safe        │
# │ textvqa_val    │ 5000    │ ~14'        │ ~2h11'      │ ~55m                │ Safe        │
# │ pope           │ 9000    │ ~16'        │ ~2h10'      │ ~1h30m              │ Safe        │
# │ gqa            │ 12578   │ ~31'        │ ~4h4'       │ ~1h30m              │ Safe        │
# │ seedbench      │ 17990   │ ~5h3'       │ ~15h        │ multi-GPU rec'd     │ Safe        │
# │ mmbench_en_dev │ 4330    │ ~8'         │ ~2h24'      │ ~10m                │ API req'd   │
# │ mmbench_cn_dev │ 4330    │ ~11'        │ ~4h         │ -                   │ API req'd   │
# │ llava_in_the_wild│ 60     │ ~7'        │ ~1h4'       │ ~20m                │ API req'd   │
# │ mmvet          │ 218       │ ~2h41'      │ ~45'        │ -                   │ API req'd   │
# │ vizwiz_vqa_val │ 4320    │ -          │ -          │ -                   │ Safe        │
# │ vqav2_val      │ 214000  │ -          │ -          │ -                   │ Safe        │
# └─────────────────┴─────────┴─────────────┴─────────────┴─────────────────────┴─────────────┘
#
# Collection models: Llava1.5-7B, Llava1.5-7B-0-16, Llava1.5-7B-frozen, Llava1.5-7B-16-32
# API tasks: Use openai/gpt-oss-120b for reliable scoring (Orfeo Qwen may fail)

# Available tasks (uncomment one):
# TASK_NAME="scienceqa_img" # Done
# TASK_NAME="mme" # Done
# TASK_NAME="textvqa_val" # Done
# TASK_NAME="pope" # Done
# TASK_NAME="gqa" # Done
# TASK_NAME="seedbench" # Running
# TASK_NAME="mmbench_en_dev" # Done
# TASK_NAME="llava_in_the_wild" # Done
# TASK_NAME="vizwiz_vqa_val" # To do
# TASK_NAME="vqav2_val" # To do
# TASK_NAME="mmbench_cn_dev" # To finish

# This script expects TASK_NAME to be set via environment variable.
# Fallback to specified task name if not set.
TASK_NAME="${TASK_NAME:-vizwiz_vqa_val}"

# =============================================================================
# EVALUATION PARAMETERS
# =============================================================================

BATCH_SIZE=1                # Limitation of lmms-eval for LLaVA models (must be 1)
NUM_FEWSHOT=""              # Empty = default, or set number (e.g., "5")
# Default sample limits are now defined in the Python script.
# Set LIMIT here to override for a specific run, e.g., LIMIT="50"
LIMIT=""                    # Empty = use Python script's default, or set number
MAX_NEW_TOKENS=""           # Empty = auto (128/1024 based on task), or set number

# Multi-GPU: Inherited from launch.sh, defaults to 1 if not set
NUM_GPUS="${NUM_GPUS:-1}"

# =============================================================================
# PATHS & MODEL CONFIGURATION
# =============================================================================
MODEL="llava-1.5-7b"  # ⚠️Used for output directory naming⚠️
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
OUTPUT_DIR="$PROJECT_ROOT/evals/$MODEL/lmms_eval/$TASK_NAME"

# Model names
FINE_TUNED_MODEL_NAME="lbasile/llava_7b"
PRETRAINED_MODEL_NAME="l11p/llava-v1.5-7b-pretrained-lbasile"
#⚠️ADJUST FOR OTHER MODELS, see documentation⚠️
LMMS_EVAL_MODEL="${LMMS_EVAL_MODEL:-llava_hf}"

# Handle No Image flag
NO_IMAGE="${NO_IMAGE:-false}"
if [ "$NO_IMAGE" = "true" ]; then
    MODEL="${MODEL}_noimage"
fi

# Cache directories
HF_HOME="$HOME/scratch/huggingface"
MODEL_CACHE_DIR="$HF_HOME/hub"
DATASETS_CACHE_DIR="$HF_HOME/datasets"

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

# Hugging Face cache
export HF_HOME
export HF_HUB_CACHE="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="$DATASETS_CACHE_DIR"

# Response caching (reload cached model outputs for faster re-runs)
export LMMS_EVAL_USE_CACHE="True"
export LMMS_EVAL_HOME="$HOME/scratch/lmms_eval_cache"

# Multi-GPU: Prevent CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# =============================================================================
# EXECUTION
# ==============================================================================

PYTHON_SCRIPT="scripts/eval/lmms_evaluation.py"

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     LLaVA EVALUATION                         ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║ Task: %-53s ║\n" "$TASK_NAME"
printf "║ Batch Size: %-47s ║\n" "$BATCH_SIZE"
if [ "$NUM_GPUS" -gt 1 ]; then
    printf "║ Mode: %-52s ║\n" "Accelerate ($NUM_GPUS GPUs)"
else
    printf "║ Mode: %-52s ║\n" "Plain Python (device_map=auto)"
fi
if [ -n "$NUM_FEWSHOT" ]; then
    printf "║ Few-shot: %-49s ║\n" "$NUM_FEWSHOT"
else
    printf "║ Few-shot: %-49s ║\n" "default"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Build base arguments
base_args=(
    --model-cache-dir "$MODEL_CACHE_DIR"
    --output-dir "$OUTPUT_DIR"
    --batch-size "$BATCH_SIZE"
    --task-name "$TASK_NAME"
    --lmms-eval-model "$LMMS_EVAL_MODEL"
)

# Add optional arguments
[ -n "$NUM_FEWSHOT" ] && base_args+=(--num-fewshot "$NUM_FEWSHOT")
[ -n "$LIMIT" ] && base_args+=(--limit "$LIMIT")
[ -n "$MAX_NEW_TOKENS" ] && base_args+=(--max-new-tokens "$MAX_NEW_TOKENS")
[ "$NO_IMAGE" = "true" ] && base_args+=(--no-image)

# Run evaluation function
run_evaluation() {
    export NUM_GPUS

    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "[Accelerate distributed: $NUM_GPUS GPUs]"
        accelerate launch --num_processes="$NUM_GPUS" \
            "$PROJECT_ROOT/$PYTHON_SCRIPT" "${base_args[@]}" "$@"
    else
        echo "[Plain Python: device_map=auto]"
        python "$PROJECT_ROOT/$PYTHON_SCRIPT" "${base_args[@]}" "$@"
    fi
}

# =============================================================================
# EVALUATION MODES
# =============================================================================

if [ -n "$SINGLE_MODEL" ]; then
    echo "[Single Model]"
    run_evaluation --model-name-or-path "$SINGLE_MODEL"

elif [ "$FINETUNED_PRETRAINED" = "true" ]; then
    echo "[Finetuned + Pretrained]"
    run_evaluation \
        --fine-tuned-model-name "$FINE_TUNED_MODEL_NAME" \
        --pretrained-model-name "$PRETRAINED_MODEL_NAME"

elif [ -n "$COLLECTION_NAME" ]; then
    echo "[Collection]"
    run_evaluation --collection-name "$COLLECTION_NAME"

else
    echo "Error: No evaluation mode specified."
    exit 1
fi

echo ""
echo "✓ Evaluation complete"
echo ""
