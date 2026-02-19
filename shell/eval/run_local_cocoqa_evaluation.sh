#!/bin/bash
# =============================================================================
# HOW TO USE
# =============================================================================
# Run COCOQA evaluation for a single model locally.
# Evaluates l11p/COCOQA-restval-ICQA in 3 modes: visual, human_captions,
# synthetic_caption. Both QA selection strategies (all, single) are run.
#
# Usage:
#   1. Default model:
#      bash shell/eval/run_local_cocoqa_evaluation.sh
#
#   2. Specify model:
#      bash shell/eval/run_local_cocoqa_evaluation.sh "OpenGVLab/InternVL2-8B"
#      bash shell/eval/run_local_cocoqa_evaluation.sh "lbasile/llava_7b"
#
#   3. Pass extra args (e.g., --limit for quick testing):
#      bash shell/eval/run_local_cocoqa_evaluation.sh "lbasile/llava_7b" "--limit 10"
# =============================================================================

set -e

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Default model (override with first argument)
DEFAULT_MODEL_NAME="lbasile/llava_7b"
# DEFAULT_MODEL_NAME="OpenGVLab/InternVL2-8B"

# Modes and QA selections to evaluate
MODES=("visual" "human_captions" "synthetic_caption")
QA_SELECTIONS=("single")

BATCH_SIZE=4

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
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/eval/cocoqa_evaluation.py"

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
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/scratch/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/scratch/huggingface/datasets}"
echo "✓ HF_HUB_CACHE: $HF_HUB_CACHE"
echo "✓ HF_DATASETS_CACHE: $HF_DATASETS_CACHE"

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
        export CUDA_HOME="$cuda_root"
        export PATH="$cuda_root/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_root/lib64:$LD_LIBRARY_PATH"
    else
        echo "⚠️  WARNING: nvcc not found!"
    fi
    echo ""
}
setup_cuda

# =============================================================================
# EXECUTION LOOP
# =============================================================================

MODEL_SUBDIR=$(basename "$MODEL_TO_EVAL" | sed 's/\//_/g')
overall_start_time=$SECONDS

for MODE in "${MODES[@]}"; do
    for QA_SEL in "${QA_SELECTIONS[@]}"; do
        echo ""
        echo "══════════════════════════════════════════════════════════════"
        echo "  COCOQA | MODE: $MODE | QA: $QA_SEL | MODEL: $MODEL_TO_EVAL"
        echo "══════════════════════════════════════════════════════════════"

        task_start_time=$SECONDS

        OUTPUT_DIR="$PROJECT_ROOT/evals/$MODEL_SUBDIR/cocoqa/$MODE"
        mkdir -p "$OUTPUT_DIR"

        python "$PYTHON_SCRIPT" \
            --model-name-or-path "$MODEL_TO_EVAL" \
            --mode "$MODE" \
            --qa-selection "$QA_SEL" \
            --output-dir "$OUTPUT_DIR" \
            --batch-size "$BATCH_SIZE" \
            --model-cache-dir "$HF_HUB_CACHE" \
            $EXTRA_ARGS

        task_elapsed=$(( SECONDS - task_start_time ))
        echo "✓ $MODE/$QA_SEL completed in ${task_elapsed}s."
    done
done

overall_elapsed=$(( SECONDS - overall_start_time ))
echo ""
echo "🎉 All COCOQA evaluations complete for $MODEL_TO_EVAL in ${overall_elapsed}s."
