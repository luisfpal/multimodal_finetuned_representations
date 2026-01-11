#!/bin/bash
# This script runs residual stream extraction and measures evaluation

# --- Configuration Section ---
DATASET_PATH_OR_NAME="~/scratch/datasets/cocoqa_unified"

# Analysis configuration
DATASET_TYPE="coco_captioning, cocoqa_img"
RESIDUAL_STREAM_TYPES="output_layer"
TOKENS_POOLING_METHODS="last"
SIMILARITY_MEASURES=""

# Similarity measure parameters
MAXK="30"
ACCEPT_RATE="0.95"

# Intrinsic dimension parameters
ID_NN_RANK="16"
ID_NN_RANGE_MAX="100"

# Processing parameters
BATCH_SIZE="25"
SEED="42"

# Directory configuration
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"
MODEL_CACHE_DIR="$HOME/scratch/huggingface/hub"

# Script configuration
FILENAME="residual_stream_trace_and_measures_evaluation.py"
FILEPATH="$PROJECT_DIR/scripts/$FILENAME"

# --- Argument Construction ---
SCRIPT_ARGS=()
SCRIPT_ARGS+=(--results-dir "$RESULTS_DIR")
SCRIPT_ARGS+=(--dataset-path-or-name "$DATASET_PATH_OR_NAME")
SCRIPT_ARGS+=(--model-cache-dir "$MODEL_CACHE_DIR")
SCRIPT_ARGS+=(--batch-size "$BATCH_SIZE")
SCRIPT_ARGS+=(--dataset-type "$DATASET_TYPE")
SCRIPT_ARGS+=(--residual-stream-types "$RESIDUAL_STREAM_TYPES")
SCRIPT_ARGS+=(--tokens-pooling-methods "$TOKENS_POOLING_METHODS")
SCRIPT_ARGS+=(--similarity-measures "$SIMILARITY_MEASURES")
SCRIPT_ARGS+=(--maxk "$MAXK")
SCRIPT_ARGS+=(--accept-rate "$ACCEPT_RATE")
SCRIPT_ARGS+=(--id-nn-rank "$ID_NN_RANK")
SCRIPT_ARGS+=(--id-nn-range-max "$ID_NN_RANGE_MAX")
SCRIPT_ARGS+=(--seed "$SEED")

# --- Debugger ---
DEBUGGER_ARGS=()
if [ "$1" = "--debug" ]; then
    DEBUGGER_ARGS=(-m debugpy --listen localhost:5678 --wait-for-client)
    echo "Debugger enabled - waiting for client connection on localhost:5678"
fi

# --- Create directories ---
mkdir -p "$RESULTS_DIR"

# Print configuration
echo "====== Residual Stream Analysis Configuration ======"
echo "Dataset: $DATASET_PATH_OR_NAME"
echo "Results directory: $RESULTS_DIR"
echo "Model cache directory: $MODEL_CACHE_DIR"
echo "Dataset type: $DATASET_TYPE"
echo "Residual stream types: $RESIDUAL_STREAM_TYPES"
echo "Token pooling methods: $TOKENS_POOLING_METHODS"
echo "Similarity measures: $SIMILARITY_MEASURES"
echo "Max k (neighborhood overlap): $MAXK"
echo "Accept rate (SVCCA): $ACCEPT_RATE"
echo "ID NN rank: $ID_NN_RANK"
echo "ID NN range max: $ID_NN_RANGE_MAX"
echo "Batch size: $BATCH_SIZE"
echo "Seed: $SEED"
echo "==============================="

# Run analysis
echo -e "\n====== Running Residual Stream Analysis ======"

# Set PyTorch memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the command
TOKENIZERS_PARALLELISM=false python "${DEBUGGER_ARGS[@]}" "${FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== Analysis Complete ======"
echo "Results saved to: $RESULTS_DIR"
