#!/bin/bash
# This script runs model vs model similarity analysis

# --- Configuration Section ---
DATASET_PATH_OR_NAME=${COCOVQA_CACHE}

# Processing parameters
BATCH_SIZE="25"
SEED="42"

# Directory configuration
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"

# Script configuration
FILENAME="models_similarity.py"
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/$FILENAME"

# Models to compare
# Model A (Rows)
MODEL_A="llava-hf/llava-1.5-7b-hf"
# Model B (Columns)
MODEL_B="l11p/llava-v1.5-7b-pretrained-liuhaotian"

# --- Environment Setup ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Run Analysis ---
echo "====== Models Similarity Analysis ======"
echo "Dataset: $DATASET_PATH_OR_NAME"
echo "Results directory: $RESULTS_DIR"
echo "Model A (Rows): $MODEL_A"
echo "Model B (Cols): $MODEL_B"

python "$FILEPATH" \
    --results-dir "$RESULTS_DIR" \
    --dataset-path-or-name "$DATASET_PATH_OR_NAME" \
    --model-a-name-or-path "$MODEL_A" \
    --model-b-name-or-path "$MODEL_B" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED"

echo -e "\n====== All Analyses Complete ======"
