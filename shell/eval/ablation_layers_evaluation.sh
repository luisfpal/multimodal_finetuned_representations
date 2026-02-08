#!/bin/bash

# =============================================================================
# Environment Setup
# =============================================================================
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
OUTPUT_DIR="$PROJECT_ROOT/results/ablations"
HF_HOME="$HOME/scratch/huggingface"
MODEL_CACHE_DIR="$HF_HOME/hub"
DATASETS_CACHE_DIR="$HF_HOME/datasets"
FILEPATH="$PROJECT_ROOT/scripts/eval/ablation_layers_evaluation.py"

export HF_HOME
export HF_HUB_CACHE="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="$DATASETS_CACHE_DIR"

# =============================================================================
# Configuration
# =============================================================================

# Models to evaluate
MODELS=(
    # "llava-hf/llava-1.5-7b-hf" #! "belongs" to hf -> llava-hf
    # "l11p/llava-v1.5-7b-pretrained-liuhaotian"
    "lbasile/llava_7b" #!"belongs" to Lorenzo -> lbasile
    "l11p/llava-v1.5-7b-pretrained-lbasile" #! "belongs" to me -> l11p 
    "lbasile/llava_13b" #!"belongs" to Lorenzo -> lbasile
    "l11p/llava-v1.5-13b-pretrained-lbasile" #! "belongs" to me -> l11p
)

# Layers to skip: 0 to 32 with step 2
# LAYERS=($(seq 0 2 32))
# LAYERS=($(seq 2 2 8))
LAYERS=(0)
# LAYERS=($(seq 10 2 16))
# LAYERS=($(seq 18 2 24))
# LAYERS=($(seq 26 2 32))

# Tasks to evaluate
TASKS=(
    # "scienceqa_img"
    # "mme"
    "mmbench_en_dev"
    # "gqa"
    # "vqav2_val"
    # "vizwiz_vqa_val"
)

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

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

# Environment specific config
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

# Load environment variables from .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Set PYTHONPATH to project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM="false"

# =============================================================================
# ORFEO LLM JWT AUTHENTICATION
# =============================================================================
if [ -n "${ORFEO_AUTH_URL}" ] && [ -n "${ORFEO_CLIENT_ID}" ]; then
    echo ""
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
        exit 1
    fi
    # Export for Python/lmms-eval
    export OPENAI_API_KEY="$JWT"
    export OPENAI_API_URL="${ORFEO_LLM_URL}"
    # export MODEL_VERSION="${ORFEO_MODEL}"
    export MODEL_VERSION="openai/gpt-oss-120b"
    echo ""
fi

# =============================================================================
# Execution Loop
# =============================================================================

echo ""
echo "Starting Ablation Evaluation..."
echo "Output Directory: $OUTPUT_DIR"
echo ""

for model in "${MODELS[@]}"; do
    echo "#################################################################"
    echo "Model: $model"
    echo "#################################################################"

    # Activate model-specific venv
    VENV="$PROJECT_ROOT/.venv"
    source "$VENV/bin/activate"

    for task in "${TASKS[@]}"; do
        echo "------------------------------------------------"
        echo "Task: $task"
        echo "------------------------------------------------"

        for layer in "${LAYERS[@]}"; do
            echo ">> Running ablation: Skip $layer layers"

            # Determine model type
            MODEL_TYPE="llava_hf"
            if [[ "$model" == *"internvl"* ]] || [[ "$model" == *"InternVL"* ]]; then
                MODEL_TYPE="internvl2"
            fi

            python "$FILEPATH" \
                --model-name-or-path "$model" \
                --model-type "$MODEL_TYPE" \
                --task "$task" \
                --layers-to-skip "$layer" \
                --output-dir "$OUTPUT_DIR"

        done
    done
done

echo "All evaluations complete."