#!/bin/bash
# This script runs layer patching benchmarking for QA tasks

# --- Model Configuration ---
MULTIMODAL_MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf"
LANGUAGE_MODEL_NAME_OR_PATH="lmsys/vicuna-7b-v1.5"
PRETRAINED_PROJECTOR_NAME_OR_PATH="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5"

# --- Dataset Configuration ---
DATASET_PATH_OR_NAME="~/scratch/datasets/cocoqa_unified"

# --- Patching Configuration ---
# PATCHING_METHOD="sliding_window"
PATCHING_METHOD="cumulative"  # Can also use "two_parts" for backward compatibility
STRIDE="2"
WINDOW_SIZE="1"
REVERSE_CUMULATIVE_LAYERS="false"  # Set to "true" to patch layers cumulatively from end to start

# --- Benchmarking Parameters ---
BATCH_SIZE="10"
MAX_NEW_TOKENS="10"
SEED="42"
BENCHMARK_MODE="patching"  # Options: "baseline", "patching", "both"

# --- Output Configuration ---
SAVE_BASELINE_OUTPUTS="true"  # Set to "false" to disable saving baseline outputs

# --- Directory Configuration ---
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"
MODEL_CACHE_DIR="$HOME/scratch/huggingface/hub"

# --- Script Configuration ---
SCRIPT_FILENAME="layers_patching_qa_benchmarking.py"
SCRIPT_FILEPATH="$PROJECT_DIR/scripts/basic_benchmarking/$SCRIPT_FILENAME"

# Activate virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

# --- Argument Construction ---
SCRIPT_ARGS=()
SCRIPT_ARGS+=(--multimodal_model_name_or_path "$MULTIMODAL_MODEL_NAME_OR_PATH")
SCRIPT_ARGS+=(--model_cache_dir "$MODEL_CACHE_DIR")
SCRIPT_ARGS+=(--pretrained_projector_name_or_path "$PRETRAINED_PROJECTOR_NAME_OR_PATH")
SCRIPT_ARGS+=(--language_model_name_or_path "$LANGUAGE_MODEL_NAME_OR_PATH")
SCRIPT_ARGS+=(--patching_method "$PATCHING_METHOD")
SCRIPT_ARGS+=(--stride "$STRIDE")
SCRIPT_ARGS+=(--window_size "$WINDOW_SIZE")
SCRIPT_ARGS+=(--batch_size "$BATCH_SIZE")
SCRIPT_ARGS+=(--max_new_tokens "$MAX_NEW_TOKENS")
SCRIPT_ARGS+=(--results_dir "$RESULTS_DIR")
SCRIPT_ARGS+=(--dataset_path_or_name "$DATASET_PATH_OR_NAME")
SCRIPT_ARGS+=(--seed "$SEED")
SCRIPT_ARGS+=(--benchmark_mode "$BENCHMARK_MODE")

# Add save_baseline_outputs flag if enabled
if [ "$SAVE_BASELINE_OUTPUTS" = "true" ]; then
    SCRIPT_ARGS+=(--save_baseline_outputs)
fi

# Add reverse_cumulative_layers flag if enabled
if [ "$REVERSE_CUMULATIVE_LAYERS" = "true" ]; then
    SCRIPT_ARGS+=(--reverse_cumulative_layers)
fi

# --- Debug Configuration ---
DEBUGGER_ARGS=()
if [ "$1" = "--debug" ]; then
    DEBUGGER_ARGS=(-m debugpy --listen localhost:5678 --wait-for-client)
    echo "Debugger enabled - waiting for client connection on localhost:5678"
fi

# --- Create directories ---
mkdir -p "$RESULTS_DIR"

# --- Display Configuration ---
echo "====== Transplanting Layers QA Benchmarking Configuration ======"
echo "Multimodal model: $MULTIMODAL_MODEL_NAME_OR_PATH"
echo "Language model: $LANGUAGE_MODEL_NAME_OR_PATH"
echo "Pretrained projector: $PRETRAINED_PROJECTOR_NAME_OR_PATH"
echo "Dataset: $DATASET_PATH_OR_NAME"
echo "Transplantation method: $TRANSPLANTATION_METHOD"
echo "Stride: $STRIDE"
echo "Window size: $WINDOW_SIZE"
echo "Results directory: $RESULTS_DIR"
echo "Model cache directory: $MODEL_CACHE_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Seed: $SEED"
echo "Save baseline outputs: $SAVE_BASELINE_OUTPUTS"
echo "Reverse cumulative layers: $REVERSE_CUMULATIVE_LAYERS"
echo "Benchmark mode: $BENCHMARK_MODE"
echo "==============================="

# --- Run Benchmarking ---
echo -e "\n====== Running Transplanting Layers QA Benchmarking ======"

# Set PyTorch memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the command
TOKENIZERS_PARALLELISM=false python "${DEBUGGER_ARGS[@]}" "${SCRIPT_FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== QA Benchmarking Complete ======"
echo "Results saved to: $RESULTS_DIR"
