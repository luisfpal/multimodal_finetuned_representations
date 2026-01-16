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
DEFAULT_MODEL_NAME="llava-hf/llava-1.5-7b-hf"

# 2. Shared Hugging Face cache directory paths.
#    Set these to your personal scratch space to avoid re-downloading.
#    We are setting these explicitly and leaving HF_HOME unset to avoid potential
#    permission issues on the cluster's filesystem.
HF_HUB_CACHE_PATH="/orfeo/cephfs/scratch/dssc/lpalaciosflores/huggingface/hub"
HF_DATASETS_CACHE_PATH="/orfeo/cephfs/scratch/dssc/lpalaciosflores/huggingface/datasets"

# 3. GPU configuration. This script is designed for single-GPU interactive use.
#    Using more than 1 GPU would require `accelerate launch`.
NUM_GPUS=1

# =============================================================================
# SCRIPT SETUP & ARGUMENT PARSING
# =============================================================================

# If a command-line argument is provided, it overrides the default model name.
MODEL_TO_EVAL="${1:-$DEFAULT_MODEL_NAME}"

if [ -z "$MODEL_TO_EVAL" ]; then
    echo "❌ Error: No model specified."
    echo "   Usage: bash $0 [HF_MODEL_NAME]"
    echo "   Or, set DEFAULT_MODEL_NAME inside the script."
    exit 1
fi

echo "▶️  Model to evaluate: $MODEL_TO_EVAL"

# Get the absolute path of the project root (assuming the script is in a subdirectory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/eval/lmms_evaluation.py"

# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

# List of benchmarks to evaluate against.
TASKS=(
    "scienceqa_img"
    "mme"
    "textvqa_val"
    "pope"
    "gqa"
    "seedbench"
    "mmbench_en_dev"
    "llava_in_the_wild"
    "vizwiz_vqa_val"
    "vqav2_val"
    "mmbench_cn_dev"
    # "mmvet"
)

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo "🔧 Setting up environment..."

# --- Python Virtual Environment ---
# Activate the virtual environment to ensure the correct Python interpreter and
# dependencies are used. This is crucial as `bash` starts a new shell.
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "✓ Virtual environment activated."
else
    echo "⚠️ Warning: Virtual environment not found at $PROJECT_ROOT/.venv/"
fi

# --- Hugging Face Cache ---
# Unset HF_HOME to avoid conflicts and use the specific cache paths below.
unset HF_HOME
export HF_HUB_CACHE="$HF_HUB_CACHE_PATH"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE_PATH"
echo "✓ HF_HUB_CACHE set to: $HF_HUB_CACHE"
echo "✓ HF_DATASETS_CACHE set to: $HF_DATASETS_CACHE"

# --- Other Environment Variables ---
export TOKENIZERS_PARALLELISM="false"
export LMMS_EVAL_USE_CACHE="True" # Use lmms-eval's response caching
export LMMS_EVAL_HOME="$HOME/scratch/lmms_eval_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- CUDA Setup ---
# Robustly find and set up CUDA environment.
setup_cuda() {
    if ! command -v nvcc &> /dev/null; then
        local candidates=("/usr/local/cuda-12.1" "/usr/local/cuda-12.2" "/usr/local/cuda-12" "/usr/local/cuda")
        for path in "${candidates[@]}"; do
            if [ -x "$path/bin/nvcc" ]; then
                export CUDA_HOME="$path"
                export PATH="$path/bin:$PATH"
                export LD_LIBRARY_PATH="$path/lib64:$LD_LIBRARY_PATH"
                break
            fi
        done
    fi

    if command -v nvcc &> /dev/null; then
        echo "✓ CUDA found: $(command -v nvcc)"
    else
        echo "⚠️ WARNING: nvcc not found. GPU operations may fail."
    fi
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
    OUTPUT_DIR="$PROJECT_ROOT/evals/$(basename "$MODEL_TO_EVAL" | sed 's/\//_/g')/lmms_eval/$TASK_NAME"
    mkdir -p "$OUTPUT_DIR"

    # Build the arguments for the Python script
    base_args=(
        --model-name-or-path "$MODEL_TO_EVAL"
        --model-cache-dir "$HF_HUB_CACHE"
        --output-dir "$OUTPUT_DIR"
        --task-name "$TASK_NAME"
        --lmms-eval-model "llava_hf" # This seems to be the default for your setup
        --batch-size 1 # LLaVA models in lmms-eval often require batch size 1
    )

    echo "[Plain Python: device_map=auto on 1 GPU]"
    python "$PYTHON_SCRIPT" "${base_args[@]}"

    task_elapsed=$(( SECONDS - task_start_time ))
    echo "✓ Task '$TASK_NAME' completed in ${task_elapsed}s."
done

overall_elapsed=$(( SECONDS - overall_start_time ))
echo ""
echo "🎉 All benchmarks complete for $MODEL_TO_EVAL in ${overall_elapsed}s."
