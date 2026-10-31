#!/bin/bash
# =============================================================================
# Run logit lens evaluation for a single model locally (no SLURM).
#
# Usage:
#   bash shell/eval/run_local_logitlens.sh                           # default model
#   bash shell/eval/run_local_logitlens.sh "OpenGVLab/InternVL2-8B"  # specify model
# =============================================================================

set -e

DEFAULT_MODEL_NAME="OpenGVLab/InternVL2-8B"
MODEL_TO_EVAL="${1:-$DEFAULT_MODEL_NAME}"

echo "Model: $MODEL_TO_EVAL"

# =============================================================================
# PATHS
# =============================================================================

# Hardcoded path: edit this path for your setup.
PROJECT_ROOT="/u/dssc/lbasile/scratch/plm"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/eval/logitlens_evaluation.py"
RESULTS_DIR="$PROJECT_ROOT/results"
VENV="$PROJECT_ROOT/.venv"

BATCH_SIZE=1
SEED=42

# =============================================================================
# ENVIRONMENT
# =============================================================================

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda3 | paste -sd ':')
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM="false"
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

# =============================================================================
# RUN
# =============================================================================

overall_start=$SECONDS

python "$PYTHON_SCRIPT" \
    --model-name-or-path "$MODEL_TO_EVAL" \
    --results-dir "$RESULTS_DIR" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED"

echo ""
echo "Logit lens eval complete for $MODEL_TO_EVAL in $(( SECONDS - overall_start ))s."
