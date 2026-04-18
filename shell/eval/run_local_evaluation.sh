#!/bin/bash
# =============================================================================
# Run evaluation for a single model locally (no SLURM).
#
# Usage:
#   bash shell/eval/run_local_evaluation.sh                              # default model
#   bash shell/eval/run_local_evaluation.sh "OpenGVLab/InternVL2-8B"     # specify model
#   bash shell/eval/run_local_evaluation.sh "lbasile/llava_7b" "--no-image"  # extra args
#
# Backend is auto-inferred from model name. No manual LMMS_EVAL_MODEL needed.
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default model (override via first argument)
# DEFAULT_MODEL_NAME="lbasile/cambrian-8b-pretrained"
# DEFAULT_MODEL_NAME="nyu-visionx/cambrian-8b"
# DEFAULT_MODEL_NAME="lbasile/llava_7b"
# DEFAULT_MODEL_NAME="lbasile/llava_13b"
# DEFAULT_MODEL_NAME="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
# DEFAULT_MODEL_NAME="OpenGVLab/InternVL2-8B"
# DEFAULT_MODEL_NAME="lmms-lab/LLaVA-OneVision-1.5-4B-Base"
# DEFAULT_MODEL_NAME="l11p/llava-v1.5-7b-pretrained-lbasile"
# DEFAULT_MODEL_NAME="l11p/llava-v1.5-13b-pretrained-lbasile"
DEFAULT_MODEL_NAME="lbasile/llava_7b_6_16"


# Tasks to evaluate (uncomment to enable)
TASKS=(
    "scienceqa_img"
    # "mme"
    # "textvqa_val"
    # "pope"
    # "gqa"
    # "seedbench"
    # "mmbench_en_dev"
    # "llava_in_the_wild"
    # "vizwiz_vqa_val"
    # "vqav2_val"
    # "mmbench_cn_dev"
    # "docvqa_val"
    # "ai2d"
    # "mmmu_val"
    # "coco2017_cap_val"
    # "flickr30k"
)

# =============================================================================
# ARGUMENTS
# =============================================================================

MODEL_TO_EVAL="${1:-$DEFAULT_MODEL_NAME}"
EXTRA_ARGS="${2:-}"
# EXTRA_ARGS="${2:---no-image}"

echo "Model: $MODEL_TO_EVAL"

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT="/u/dssc/lpalaciosflores/multimodal_finetuned_representations"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/eval/lmms_evaluation.py"
VENV="$PROJECT_ROOT/.venv"

# =============================================================================
# ENVIRONMENT
# =============================================================================

# Virtual environment
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi

# Load .env (HF cache, API keys, etc.)
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export TOKENIZERS_PARALLELISM="false"
export LMMS_EVAL_USE_CACHE="True"
export LMMS_EVAL_HOME="$HOME/scratch/lmms_eval_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# CUDA setup
setup_cuda() {
    if command -v module &> /dev/null; then
        module load cuda/12.1 2>/dev/null || true
    fi

    if ! command -v nvcc &> /dev/null || [ ! -x "$(command -v nvcc)" ]; then
        for path in /usr/local/cuda-12.1 /usr/local/cuda-12.2 /usr/local/cuda-12 /usr/local/cuda; do
            if [ -x "$path/bin/nvcc" ]; then
                export CUDA_HOME="$path"
                export PATH="$path/bin:$PATH"
                export LD_LIBRARY_PATH="$path/lib64:$LD_LIBRARY_PATH"
                break
            fi
        done
    fi

    if command -v nvcc &> /dev/null; then
        local cuda_root
        cuda_root=$(dirname "$(dirname "$(command -v nvcc)")")
        export CUDA_HOME="$cuda_root"
        export PATH="$cuda_root/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_root/lib64:$LD_LIBRARY_PATH"
    fi
}
setup_cuda

# Orfeo LLM JWT (optional, for judge tasks)
if [ -n "${ORFEO_AUTH_URL:-}" ] && [ -n "${ORFEO_CLIENT_ID:-}" ]; then
    JWT=$(curl -s -X POST "${ORFEO_AUTH_URL}" \
      -H "Content-Type: application/x-www-form-urlencoded" \
      -d "grant_type=password" \
      -d "client_id=${ORFEO_CLIENT_ID}" \
      -d "username=${ORFEO_USERNAME}" \
      -d "password=${ORFEO_CLIENT_SECRET}" \
      -d "scope=profile" | jq -r '.access_token')

    if [ -n "$JWT" ] && [ "$JWT" != "null" ]; then
        export OPENAI_API_KEY="$JWT"
        export OPENAI_API_URL="${ORFEO_LLM_URL}"
        export MODEL_VERSION="${ORFEO_MODEL}"
    fi
fi

# =============================================================================
# RUN
# =============================================================================

overall_start=$SECONDS

for TASK_NAME in "${TASKS[@]}"; do
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  TASK: $TASK_NAME"
    echo "═══════════════════════════════════════════════════════"

    # Output directory: evals/<model_basename>/lmms_eval/<task>
    MODEL_SUBDIR=$(basename "$MODEL_TO_EVAL")
    if [[ "$EXTRA_ARGS" == *"--no-image"* ]]; then
        MODEL_SUBDIR="${MODEL_SUBDIR}_noimage"
    fi
    OUTPUT_DIR="$PROJECT_ROOT/evals/$MODEL_SUBDIR/lmms_eval/$TASK_NAME"
    mkdir -p "$OUTPUT_DIR"

    python "$PYTHON_SCRIPT" \
        --model-name-or-path "$MODEL_TO_EVAL" \
        --model-cache-dir "${HF_HUB_CACHE:-$HOME/scratch/huggingface/hub}" \
        --output-dir "$OUTPUT_DIR" \
        --task-name "$TASK_NAME" \
        --batch-size 1 \
        $EXTRA_ARGS
done

echo ""
echo "All tasks complete for $MODEL_TO_EVAL in $(( SECONDS - overall_start ))s."

# =============================================================================
# COMPOSITIONALITY BENCHMARKS (ARO + BiVLC + SugarCrepe)
# Results saved under evals/<model>/compositionality/ (not lmms_eval/)
# =============================================================================
bash "$PROJECT_ROOT/shell/eval/run_local_compositionality_evaluation.sh" "$MODEL_TO_EVAL"
