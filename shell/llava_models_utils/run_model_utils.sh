#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# =================================================================================
#  MODEL UTILITIES RUN SCRIPT
# =================================================================================
#
#  Usage:
#  1. Set the `MODE` variable to either "download" or "port".
#  2. Edit the variables in the corresponding configuration section below.
#  3. Run the script.
#
# =================================================================================


# =================================================================================
#  1. USER CONFIGURATION
# =================================================================================

# --- CHOOSE MODE ---
# Set the operation you want to perform. Available modes: "download", "port"
MODE="download"


# --- A) DOWNLOAD MODE CONFIGURATION ---

# List of individual models to download.
MODELS_TO_DOWNLOAD=(
  # "llava-hf/llava-1.5-7b-hf"
  # "openai/clip-vit-large-patch14-336"
  "l11p/llava-7b-lbasile"
)

# List of collections to download. Leave empty to ignore.
COLLECTIONS_TO_DOWNLOAD=(
  # "Llava1.5-7B"
  # "Llava1.5-7B-frozen"
)


# --- B) PORT MODE CONFIGURATION ---

# List of source models and their corresponding target repo IDs.
# The number of models in each list below must be equal.
SOURCE_MODELS_TO_PORT=(
  "lbasile/llava-7b"
  # "lbasile/llava-v1-7b"
)
TARGET_REPOS_FOR_PORT=(
  "l11p/llava-v1.5-7b-lbasile-original"
  # "l11p/llava-v1-7b-lbasile-original"
)

# List of collections to port. Leave empty to ignore.
COLLECTIONS_TO_PORT=(
  "lbasile-7b-raw"
)


# =================================================================================
#  2. EXECUTION LOGIC (DO NOT EDIT BELOW THIS LINE)
# =================================================================================

# --- Script Constants ---
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
DOWNLOAD_SCRIPT="$PROJECT_ROOT/scripts/llava_models_utils/download_models.py"
PORT_SCRIPT="$PROJECT_ROOT/scripts/llava_models_utils/port_model_weights.py"
CACHE_DIR="$HOME/scratch/huggingface/hub"


# --- Function Definitions ---

run_download() {
  echo "Running Download Script..."

  DOWNLOAD_ARGS=()
  if [ ${#MODELS_TO_DOWNLOAD[@]} -gt 0 ]; then
    DOWNLOAD_ARGS+=(--model-id "${MODELS_TO_DOWNLOAD[@]}")
  fi
  if [ ${#COLLECTIONS_TO_DOWNLOAD[@]} -gt 0 ]; then
    DOWNLOAD_ARGS+=(--collection "${COLLECTIONS_TO_DOWNLOAD[@]}")
  fi

  if [ ${#DOWNLOAD_ARGS[@]} -eq 0 ]; then
      echo "No models or collections specified for download. Exiting."
      exit 0
  fi

  python "$DOWNLOAD_SCRIPT" --cache-dir "$CACHE_DIR" "${DOWNLOAD_ARGS[@]}"
}

run_port() {
  echo "Running Porting Script..."

  PORT_ARGS=()
  if [ ${#SOURCE_MODELS_TO_PORT[@]} -gt 0 ]; then
    PORT_ARGS+=(--source-model-id "${SOURCE_MODELS_TO_PORT[@]}")
    PORT_ARGS+=(--target-repo-id "${TARGET_REPOS_FOR_PORT[@]}")
  fi
  if [ ${#COLLECTIONS_TO_PORT[@]} -gt 0 ]; then
    PORT_ARGS+=(--collection "${COLLECTIONS_TO_PORT[@]}")
  fi

  if [ ${#PORT_ARGS[@]} -eq 0 ]; then
      echo "No models or collections specified for porting. Exiting."
      exit 0
  fi

  # This operation requires a GPU and specific memory configuration
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  python "$PORT_SCRIPT" --cache-dir "$CACHE_DIR" "${PORT_ARGS[@]}"
}

# --- Main Execution ---

echo "================================================="
echo "        Llava Model Utilities"
echo "================================================="
echo "Mode: $MODE"
echo "-------------------------------------------------"

case "$MODE" in
  download)
    run_download
    ;;
  port)
    run_port
    ;;
  *)
    echo "Error: Invalid MODE selected. Please choose 'download' or 'port'."
    exit 1
    ;;
esac

echo "-------------------------------------------------"
echo "✅ Script finished."
echo "================================================="
