#!/bin/bash
# This script runs logit lens evaluation on specified models

# =============================================================================
# Configuration
# =============================================================================

# Project paths
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_ROOT/results"
SCRIPTS_DIR="$PROJECT_ROOT/scripts/eval"
PYTHON_SCRIPT="$SCRIPTS_DIR/logitlens_evaluation.py"

# Processing parameters
BATCH_SIZE="1"
SEED="42"

# Models to evaluate
MODELS=(
    # "lbasile/llava_7b"
    # "l11p/llava-v1.5-7b-pretrained-lbasile"
    "OpenGVLab/InternVL2-8B"  # Finetuned
)

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Environment specific config
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

# Load environment variables from .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Set PYTHONPATH to project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM="false"

# =============================================================================
# Run Analysis
# =============================================================================

echo "====== Logit Lens Evaluation ======"
echo "Results directory: $RESULTS_DIR"

CMD_ARGS=(
    --results-dir "$RESULTS_DIR"
    --batch-size "$BATCH_SIZE"
    --seed "$SEED"
)

for MODEL in "${MODELS[@]}"; do
    echo -e "\n----------------------------------------------------------------"
    echo "🤖 Processing Model: $MODEL"
    echo "----------------------------------------------------------------"

    python "$PYTHON_SCRIPT" \
        --model-name-or-path "$MODEL" \
        "${CMD_ARGS[@]}"
done

echo -e "\n====== All Evaluations Complete ======"
