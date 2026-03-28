#!/bin/bash
set -e

# Worker script called by launch_lmms_specific.sh (or directly with env vars).
# Expects: SINGLE_MODEL, TASK_NAME, OUTPUT_MODEL_DIR (from launcher)

TASK_NAME="${TASK_NAME:-scienceqa_img}"
BATCH_SIZE=1
NUM_GPUS="${NUM_GPUS:-1}"
NO_IMAGE="${NO_IMAGE:-false}"

# Paths
OUTPUT_MODEL_DIR="${OUTPUT_MODEL_DIR:-$(basename "$SINGLE_MODEL")}"
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
OUTPUT_DIR="$PROJECT_ROOT/evals/$OUTPUT_MODEL_DIR/lmms_eval/$TASK_NAME"

if [ "$NO_IMAGE" = "true" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_noimage"
fi

# Cache
HF_HOME="$HOME/scratch/huggingface"
MODEL_CACHE_DIR="$HF_HOME/hub"
export HF_HOME
export HF_HUB_CACHE="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export LMMS_EVAL_USE_CACHE="True"
export LMMS_EVAL_HOME="$HOME/scratch/lmms_eval_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_SCRIPT="scripts/eval/lmms_evaluation.py"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
printf "║ Task: %-49s ║\n" "$TASK_NAME"
printf "║ Model: %-48s ║\n" "${SINGLE_MODEL:-none}"
printf "║ GPUs: %-49s ║\n" "$NUM_GPUS"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Build args
base_args=(
    --model-cache-dir "$MODEL_CACHE_DIR"
    --output-dir "$OUTPUT_DIR"
    --batch-size "$BATCH_SIZE"
    --task-name "$TASK_NAME"
)

[ "$NO_IMAGE" = "true" ] && base_args+=(--no-image)

run_evaluation() {
    export NUM_GPUS
    if [ "$NUM_GPUS" -gt 1 ]; then
        accelerate launch --num_processes="$NUM_GPUS" \
            "$PROJECT_ROOT/$PYTHON_SCRIPT" "${base_args[@]}" "$@"
    else
        python "$PROJECT_ROOT/$PYTHON_SCRIPT" "${base_args[@]}" "$@"
    fi
}

if [ -n "$SINGLE_MODEL" ]; then
    run_evaluation --model-name-or-path "$SINGLE_MODEL"
else
    echo "Error: SINGLE_MODEL not set."
    exit 1
fi

echo ""
echo "✓ Evaluation complete"
