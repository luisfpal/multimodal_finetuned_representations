#!/bin/bash
# =============================================================================
# HOW TO USE
# =============================================================================
# Run captioned + visual evaluation for a single model locally.
# Supports both captioned tasks (images → text captions) and original tasks.
# The model backend (llava_hf, internvl2) is auto-inferred
# from the model name — no manual configuration needed.
#
# Usage:
#   1. Default model:
#      bash shell/eval/run_local_captioned_evaluation.sh
#
#   2. Specify model:
#      bash shell/eval/run_local_captioned_evaluation.sh "OpenGVLab/InternVL2-8B"
#      bash shell/eval/run_local_captioned_evaluation.sh "lbasile/llava_7b"
#
#   3. Pass extra args (e.g., --limit):
#      bash shell/eval/run_local_captioned_evaluation.sh "lbasile/llava_7b" "--limit 10"
# =============================================================================

set -e

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Default model (override with first argument)
DEFAULT_MODEL_NAME="lbasile/llava_7b"
# DEFAULT_MODEL_NAME="OpenGVLab/InternVL2-8B"

# =============================================================================
# SCRIPT SETUP & ARGUMENT PARSING
# =============================================================================

MODEL_TO_EVAL="${1:-$DEFAULT_MODEL_NAME}"
EXTRA_ARGS="${2:-}"

if [ -z "$MODEL_TO_EVAL" ]; then
    echo "❌ Error: No model specified."
    echo "   Usage: bash $0 [HF_MODEL_NAME] [EXTRA_ARGS]"
    exit 1
fi

echo "▶️  Model to evaluate: $MODEL_TO_EVAL"

PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/eval/lmms_evaluation_captioned.py"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# =============================================================================
# TASK CONFIGURATION
# =============================================================================

TASKS=(
    # Captioned variants (images replaced by text captions)
    "mme_captioned"
    "mmbench_en_dev_captioned"
    "vqav2_val_captioned_synthetic"
    "vqav2_val_captioned_human"
    "gqa_val_balanced_captioned"

    # GQA val-balanced with images (visual baseline for new dataset)
    "gqa_val_balanced"

    # Original tasks also work with this script:
    # "mme"
    # "mmbench_en_dev"
    # "vqav2_val"
    # "gqa"
)

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo "🔧 Setting up environment..."

# --- Python Virtual Environment ---
VENV="$PROJECT_ROOT/.venv"
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
    echo "✓ Virtual environment activated ($VENV)."
else
    echo "⚠️ Warning: Virtual environment not found at $VENV"
fi

# --- Hugging Face Cache ---
export HF_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
echo "✓ HF_HUB_CACHE set to: $HF_HUB_CACHE"
echo "✓ HF_DATASETS_CACHE set to: $HF_DATASETS_CACHE"

# --- Other Environment Variables ---
export TOKENIZERS_PARALLELISM="false"
export LMMS_EVAL_USE_CACHE="True"
export LMMS_EVAL_HOME="$HOME/scratch/lmms_eval_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- CUDA Setup ---
setup_cuda() {
    echo ""
    if command -v module &> /dev/null; then
        module load cuda/12.1 2>/dev/null || true
    fi
    if ! command -v nvcc &> /dev/null || [ ! -x "$(command -v nvcc)" ]; then
        local candidates=("/usr/local/cuda-12.1" "/usr/local/cuda-12.2" "/usr/local/cuda-12" "/usr/local/cuda")
        for path in "${candidates[@]}"; do
            if [ -x "$path/bin/nvcc" ]; then
                echo "🔧 Configuring CUDA from: $path"
                export CUDA_HOME="$path"
                export PATH="$path/bin:$PATH"
                export LD_LIBRARY_PATH="$path/lib64:$LD_LIBRARY_PATH"
                break
            fi
        done
    fi
    if command -v nvcc &> /dev/null; then
        local nvcc_path=$(command -v nvcc)
        local cuda_root=$(dirname $(dirname "$nvcc_path"))
        echo "✓ CUDA active: $nvcc_path"
        echo "  → Enforcing CUDA_HOME=$cuda_root"
        export CUDA_HOME="$cuda_root"
        export PATH="$cuda_root/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_root/lib64:$LD_LIBRARY_PATH"
    else
        echo "⚠️  WARNING: nvcc not found!"
    fi
    echo ""
}
setup_cuda

# --- Orfeo LLM Judge Authentication ---
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    if [ -n "${ORFEO_AUTH_URL}" ] && [ -n "${ORFEO_CLIENT_ID}" ]; then
        echo "🔐 Generating JWT token for Orfeo LLM..."
        JWT=$(curl -s -X POST "${ORFEO_AUTH_URL}" \
          -H "Content-Type: application/x-www-form-urlencoded" \
          -d "grant_type=password" \
          -d "client_id=${ORFEO_CLIENT_ID}" \
          -d "username=${ORFEO_USERNAME}" \
          -d "password=${ORFEO_CLIENT_SECRET}" \
          -d "scope=profile" | jq -r '.access_token')

        if [ -z "$JWT" ] || [ "$JWT" = "null" ]; then
            echo "❌ Failed to get JWT token from Orfeo!"
        else
            export OPENAI_API_KEY="$JWT"
            export OPENAI_API_URL="${ORFEO_LLM_URL}"
            export MODEL_VERSION="${ORFEO_MODEL}"
            echo "✓ Orfeo JWT token acquired."
        fi
    fi
fi

# =============================================================================
# EXECUTION LOOP
# =============================================================================

overall_start_time=$SECONDS

for TASK_NAME in "${TASKS[@]}"; do
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  TASK: $TASK_NAME | MODEL: $MODEL_TO_EVAL"
    echo "══════════════════════════════════════════════════════════════"

    task_start_time=$SECONDS

    MODEL_SUBDIR=$(basename "$MODEL_TO_EVAL" | sed 's/\//_/g')
    OUTPUT_DIR="$PROJECT_ROOT/evals/$MODEL_SUBDIR/lmms_eval/$TASK_NAME"
    mkdir -p "$OUTPUT_DIR"

    base_args=(
        --model-name-or-path "$MODEL_TO_EVAL"
        --model-cache-dir "$HF_HUB_CACHE"
        --output-dir "$OUTPUT_DIR"
        --task-name "$TASK_NAME"
        --batch-size 1
    )

    echo "[Plain Python: device_map=auto on 1 GPU]"
    python "$PYTHON_SCRIPT" "${base_args[@]}" $EXTRA_ARGS

    task_elapsed=$(( SECONDS - task_start_time ))
    echo "✓ Task '$TASK_NAME' completed in ${task_elapsed}s."
done

overall_elapsed=$(( SECONDS - overall_start_time ))
echo ""
echo "🎉 All tasks complete for $MODEL_TO_EVAL in ${overall_elapsed}s."
