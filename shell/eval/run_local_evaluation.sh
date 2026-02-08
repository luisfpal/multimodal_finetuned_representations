#!/bin/bash
# =============================================================================
# HOW TO USE
# =============================================================================
# This script is for running a single model evaluation against all benchmarks
# directly in your current shell (i.e., without submitting a SLURM job).
# It is ideal for interactive sessions and debugging.
#
# Usage:
# 1. To run with the default model specified inside this script:
#    bash shell/eval/run_local_evaluation.sh
#
# 2. To run with a different model, pass its name as an argument:
#    bash shell/eval/run_local_evaluation.sh "your-hf-org/your-model-name"
#
# 3. Configure the `DEFAULT_MODEL_NAME` and cache paths below as needed.

# =============================================================================

set -e # Exit immediately if a command exits with a non-zero status.

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# 1. Default model to evaluate if no command-line argument is provided.
# DEFAULT_MODEL_NAME="OpenGVLab/InternVL2-8B"
# DEFAULT_MODEL_NAME="OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"
# DEFAULT_MODEL_NAME="OpenGVLab/InternVL3_5-8B-Pretrained"
# DEFAULT_MODEL_NAME="OpenGVLab/InternVL3_5-8B-Instruct"
# DEFAULT_MODEL_NAME="OpenGVLab/InternVL3_5-8B-MPO"
DEFAULT_MODEL_NAME="OpenGVLab/InternVL3_5-8B"
# DEFAULT_MODEL_NAME="lbasile/llava_13b"

# 2. Model class in the lmms-eval library
#⚠️ADJUST FOR OTHER MODELS, see documentation⚠️
LMMS_EVAL_MODEL="${LMMS_EVAL_MODEL:-internvl2}"
# LMMS_EVAL_MODEL="${LMMS_EVAL_MODEL:-llava_hf}"

# =============================================================================
# SCRIPT SETUP & ARGUMENT PARSING
# =============================================================================

# If a command-line argument is provided, it overrides the default model name.
MODEL_TO_EVAL="${1:-$DEFAULT_MODEL_NAME}"
EXTRA_ARGS="${2:-}"

if [ -z "$MODEL_TO_EVAL" ]; then
    echo "❌ Error: No model specified."
    echo "   Usage: bash $0 [HF_MODEL_NAME]"
    echo "   Or, set DEFAULT_MODEL_NAME inside the script."
    exit 1
fi

echo "▶️  Model to evaluate: $MODEL_TO_EVAL"

# Get the absolute path of the project root (assuming the script is in a subdirectory)
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/eval/lmms_evaluation.py"

# Load environment variables from .env (if needed)
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

# List of benchmarks to evaluate against.
TASKS=(
    # "scienceqa_img"
    "mme"
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
)

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo "🔧 Setting up environment..."

# --- Python Virtual Environment ---
# Activate the model-specific virtual environment.
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
export LMMS_EVAL_USE_CACHE="True" # Use lmms-eval's response caching
export LMMS_EVAL_HOME="$HOME/scratch/lmms_eval_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- CUDA Setup ---
# Robust CUDA configuration for both H100 (module) and A100 (manual)
setup_cuda() {
    echo ""
    # 1. Try module load (standard on H100)
    if command -v module &> /dev/null; then
        module load cuda/12.1 2>/dev/null || true
    fi

    # 2. If nvcc is missing/broken, try to find a valid existing installation
    #    This often happens on A100 nodes where modules might differ
    if ! command -v nvcc &> /dev/null || [ ! -x "$(command -v nvcc)" ]; then
        # List of potential CUDA paths to check (prefer 12.1, then 12.x, then default)
        local candidates=("/usr/local/cuda-12.1" "/usr/local/cuda-12.2" "/usr/local/cuda-12" "/usr/local/cuda")

        for path in "${candidates[@]}"; do
            if [ -x "$path/bin/nvcc" ]; then
                echo "🔧 Manually configuring CUDA from: $path"
                export CUDA_HOME="$path"
                export PATH="$path/bin:$PATH"
                export LD_LIBRARY_PATH="$path/lib64:$LD_LIBRARY_PATH"
                break
            fi
        done
    fi

    # Verification and Enforcement
    if command -v nvcc &> /dev/null; then
        local nvcc_path=$(command -v nvcc)
        # Resolve symlinks if possible, or assume standard structure
        local cuda_root=$(dirname $(dirname "$nvcc_path"))

        echo "✓ CUDA active: $nvcc_path"
        echo "  → Enforcing CUDA_HOME=$cuda_root"

        export CUDA_HOME="$cuda_root"
        export PATH="$cuda_root/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_root/lib64:$LD_LIBRARY_PATH"
    else
        echo "⚠️  WARNING: nvcc not found! Compilation may fail."
    fi
    echo ""
}
setup_cuda

# --- Orfeo LLM Judge Authentication (Optional) ---
# If you use a private LLM judge, this will acquire a token.
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

# Record total time
overall_start_time=$SECONDS

for TASK_NAME in "${TASKS[@]}"; do
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  TASK: $TASK_NAME"
    echo "══════════════════════════════════════════════════════════════"

    task_start_time=$SECONDS

    # Define a unique output directory for this model and task
    MODEL_SUBDIR=$(basename "$MODEL_TO_EVAL" | sed 's/\//_/g')
    if [[ "$EXTRA_ARGS" == *"--no-image"* ]]; then
        MODEL_SUBDIR="${MODEL_SUBDIR}_noimage"
    fi
    OUTPUT_DIR="$PROJECT_ROOT/evals/$MODEL_SUBDIR/lmms_eval/$TASK_NAME"
    mkdir -p "$OUTPUT_DIR"

    # Build the arguments for the Python script
    base_args=(
        --model-name-or-path "$MODEL_TO_EVAL"
        --model-cache-dir "$HF_HUB_CACHE"
        --output-dir "$OUTPUT_DIR"
        --task-name "$TASK_NAME"
        --lmms-eval-model "$LMMS_EVAL_MODEL"
        --batch-size 1 # LLaVA models in lmms-eval often require batch size 1
    )

    echo "[Plain Python: device_map=auto on 1 GPU]"
    python "$PYTHON_SCRIPT" "${base_args[@]}" $EXTRA_ARGS

    task_elapsed=$(( SECONDS - task_start_time ))
    echo "✓ Task '$TASK_NAME' completed in ${task_elapsed}s."
done

overall_elapsed=$(( SECONDS - overall_start_time ))
echo ""
echo "🎉 All benchmarks complete for $MODEL_TO_EVAL in ${overall_elapsed}s."
