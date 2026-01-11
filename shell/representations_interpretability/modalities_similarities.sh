#!/bin/bash
# This script runs modalities similarities analysis

# --- Configuration Section ---
DATASET_PATH_OR_NAME="~/scratch/datasets/cocoqa_unified"

# Analysis configuration
RESIDUAL_STREAM_TYPES="output_layer"
DATASET_TYPE="coco_captioning"

# Processing parameters
BATCH_SIZE="16"
SEED="42"
BOOTSTRAP="True"
N_BOOT="1000"

# Homogeneity score parameters
MAXK="128"
RANGE_MAX="128"
K="16"
Z="1.65"

# Directory configuration
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"
MODEL_CACHE_DIR="$HOME/scratch/huggingface/hub"

# Script configuration
FILENAME="modalities_similarities.py"
FILEPATH="$PROJECT_DIR/scripts/$FILENAME"

# --- Argument Construction ---
SCRIPT_ARGS=()
SCRIPT_ARGS+=(--results-dir "$RESULTS_DIR")
SCRIPT_ARGS+=(--dataset-path-or-name "$DATASET_PATH_OR_NAME")
SCRIPT_ARGS+=(--model-cache-dir "$MODEL_CACHE_DIR")
SCRIPT_ARGS+=(--batch-size "$BATCH_SIZE")
SCRIPT_ARGS+=(--dataset-type "$DATASET_TYPE")
SCRIPT_ARGS+=(--residual-stream-types "$RESIDUAL_STREAM_TYPES")
SCRIPT_ARGS+=(--maxk "$MAXK")
SCRIPT_ARGS+=(--range-max "$RANGE_MAX")
SCRIPT_ARGS+=(--k "$K")
SCRIPT_ARGS+=(--Z "$Z")
SCRIPT_ARGS+=(--seed "$SEED")
SCRIPT_ARGS+=(--bootstrap "$BOOTSTRAP")
SCRIPT_ARGS+=(--n-boot "$N_BOOT")

# --- Debugger ---
DEBUGGER_ARGS=()
if [ "$1" = "--debug" ]; then
    DEBUGGER_ARGS=(-m debugpy --listen localhost:5678 --wait-for-client)
    echo "Debugger enabled - waiting for client connection on localhost:5678"
fi

# --- Create directories ---
mkdir -p "$RESULTS_DIR"

# Print configuration
echo "====== Modalities Similarities Analysis Configuration ======"
echo "Dataset: $DATASET_PATH_OR_NAME"
echo "Results directory: $RESULTS_DIR"
echo "Model cache directory: $MODEL_CACHE_DIR"
echo "Residual stream types: $RESIDUAL_STREAM_TYPES"
echo "Dataset type: $DATASET_TYPE"
echo "Batch size: $BATCH_SIZE"
echo "Homogeneity score parameters:"
echo "  - maxk: $MAXK"
echo "  - range_max: $RANGE_MAX"
echo "  - k: $K"
echo "  - Z: $Z"
echo "Seed: $SEED"
echo "Bootstrap: $BOOTSTRAP"
echo "Number of bootstrap samples: $N_BOOT"
echo "==============================="

# Run analysis
echo -e "\n====== Running Modalities Similarities Analysis ======"

# Set PyTorch memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the command
TOKENIZERS_PARALLELISM=false python "${DEBUGGER_ARGS[@]}" "${FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== Analysis Complete ======"
echo "Results saved to: $RESULTS_DIR" 