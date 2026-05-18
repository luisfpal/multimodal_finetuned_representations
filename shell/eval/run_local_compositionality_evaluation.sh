#!/bin/bash
# =============================================================================
# Run compositionality evaluation (ARO + BiVLC) locally without SLURM.
#
# Usage:
#   1. Default model:
#      bash shell/eval/run_local_compositionality_evaluation.sh
#
#   2. Specify model:
#      bash shell/eval/run_local_compositionality_evaluation.sh "OpenGVLab/InternVL2-8B"
#
#   3. Specify model + benchmark:
#      bash shell/eval/run_local_compositionality_evaluation.sh "lbasile/llava_7b" "bivlc"
#
#   4. Quick test (limited samples):
#      bash shell/eval/run_local_compositionality_evaluation.sh "lbasile/llava_7b" "bivlc" 10
# =============================================================================

set -e

# =============================================================================
# USER CONFIGURATION
# =============================================================================

PROJECT_ROOT="/u/dssc/lbasile/scratch/plm"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/eval/compositionality_evaluation.py"

# Default model (override with first argument)
DEFAULT_MODEL_NAME="lbasile/llava_7b"
# DEFAULT_MODEL_NAME="OpenGVLab/InternVL2-8B"
# =============================================================================
# DATA PATHS
# =============================================================================
ARO_DATA_ROOT="/orfeo/cephfs/scratch/dssc/lpalaciosflores/datasets/vision-language-models-are-bows"
SUGARCREPE_DATA_ROOT="/orfeo/cephfs/scratch/dssc/lpalaciosflores/datasets/sugar-crepe"
# =============================================================================
# ARGUMENT PARSING
# =============================================================================

MODEL_TO_EVAL="${1:-$DEFAULT_MODEL_NAME}"
BENCHMARK="all"
LIMIT=""
EXTRA_ARGS=""

shift || true
for arg in "$@"; do
    case "$arg" in
        --*)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
        *)
            if [ -z "$BENCHMARK_SET" ]; then
                BENCHMARK="$arg"
                BENCHMARK_SET=1
            elif [ -z "$LIMIT" ]; then
                LIMIT="$arg"
            fi
            ;;
    esac
done
EXTRA_ARGS="${EXTRA_ARGS# }"

if [ -z "$MODEL_TO_EVAL" ]; then
    echo "Error: No model specified."
    echo "Usage: bash $0 [MODEL] [BENCHMARK] [LIMIT] [--no-image]"
    exit 1
fi

echo "Model:     $MODEL_TO_EVAL"
echo "Benchmark: $BENCHMARK"
[ -n "$LIMIT" ] && echo "Limit:     $LIMIT samples"
[ -n "$EXTRA_ARGS" ] && echo "Extra:     $EXTRA_ARGS"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
echo ""
echo "Setting up environment..."

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# --- Python Virtual Environment ---
VENV="$PROJECT_ROOT/.venv"
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
    echo "Virtual environment activated ($VENV)."
else
    echo "Warning: Virtual environment not found at $VENV"
fi

# --- Hugging Face Cache ---
export HF_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"

# --- Other Environment Variables ---
export TOKENIZERS_PARALLELISM="false"
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
                echo "Configuring CUDA from: $path"
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
        echo "CUDA active: $nvcc_path"
        export CUDA_HOME="$cuda_root"
        export PATH="$cuda_root/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_root/lib64:$LD_LIBRARY_PATH"
    else
        echo "WARNING: nvcc not found!"
    fi
    echo ""
}
setup_cuda

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_SUBDIR=$(basename "$MODEL_TO_EVAL" | sed 's/\//_/g')
if [[ "$EXTRA_ARGS" == *"--no-image"* ]]; then
    MODEL_SUBDIR="${MODEL_SUBDIR}_noimage"
fi
OUTPUT_DIR="$PROJECT_ROOT/evals/$MODEL_SUBDIR/compositionality"

mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "  COMPOSITIONALITY EVALUATION (LOCAL)"
echo "  Model:  $MODEL_TO_EVAL"
echo "  Output: $OUTPUT_DIR"
echo "================================================================"
echo ""

start_time=$SECONDS

args=(
    --model-name-or-path "$MODEL_TO_EVAL"
    --output-dir "$OUTPUT_DIR"
    --benchmark "$BENCHMARK"
    --datasets-cache-dir "$HF_DATASETS_CACHE"
    --aro-data-root "$ARO_DATA_ROOT"
    --sugarcrepe-data-root "$SUGARCREPE_DATA_ROOT"
)
[ -n "$LIMIT" ] && args+=(--limit "$LIMIT")

python "$PYTHON_SCRIPT" "${args[@]}" $EXTRA_ARGS

elapsed=$(( SECONDS - start_time ))
echo ""
echo "All evaluations complete for $MODEL_TO_EVAL in ${elapsed}s."
