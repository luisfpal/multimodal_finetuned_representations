#!/bin/bash
# This script runs multimodality similarity analysis (Text vs Image)

# --- Configuration Section ---
DATASET_PATH_OR_NAME=${COCOVQA_CACHE}
# Uncomment below for Lorenzo runs
# export PYTHONPATH=${HOME}/multimodal_finetuned_representations
# export HF_HUB_CACHE=/orfeo/cephfs/scratch/dssc/lpalaciosflores/huggingface/hub
# DATASET_PATH_OR_NAME=/orfeo/cephfs/scratch/dssc/lpalaciosflores/datasets/cocoqa_unified
## Activate virtual environment
# source "$PROJECT_DIR/.venv/bin/activate"

# Analysis configuration
# Uses defaults from python script:
# RESIDUAL_STREAM_TYPES="output_layer"
# TOKENS_POOLING_METHODS="last"
# SIMILARITY_MEASURES="neighborhood_overlap,linear_cka"

# Processing parameters
BATCH_SIZE="1"
SEED="42"

# Directory configuration
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"

# Script configuration
FILENAME="multimodality_similarity.py"
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/$FILENAME"

# Models to evaluate
MODELS=(
    "llava-hf/llava-1.5-7b-hf"
    "l11p/llava-v1.5-7b-pretrained-liuhaotian"
)

# --- Environment Setup ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Run Analysis
echo "====== Multimodality Similarity Analysis ======"
echo "Dataset: $DATASET_PATH_OR_NAME"
echo "Results directory: $RESULTS_DIR"

for MODEL_NAME in "${MODELS[@]}"; do
    echo -e "\n----------------------------------------------------------------"
    echo "🤖 Processing Model: $MODEL_NAME"
    echo "----------------------------------------------------------------"

    python3 "$FILEPATH" \
        --results-dir "$RESULTS_DIR" \
        --dataset-path-or-name "$DATASET_PATH_OR_NAME" \
        --model-name-or-path "$MODEL_NAME" \
        --batch-size "$BATCH_SIZE" \
        --seed "$SEED"

    echo "✅ Completed: $MODEL_NAME"
done


echo -e "\n====== All Analyses Complete ======"
