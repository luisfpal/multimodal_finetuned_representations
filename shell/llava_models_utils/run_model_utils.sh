#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# =================================================================================
#  MODEL UTILITIES RUN SCRIPT
# =================================================================================
#
#  How to Use:
#  1. Set the `MODE` variable below to "download" or "port".
#  2. Configure the variables in the relevant section (A, B, or C).
#  3. Run the script: `bash shell/llava_models_utils/run_model_utils.sh`
#
# =================================================================================
# 0. ⚠️⚠️⚠️ PROJECT ROOT ⚠️⚠️⚠️ Adjust this path to your local setup
# =================================================================================
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
# =================================================================================
#  1. CORE CONFIGURATION
# =================================================================================

# --- A) CHOOSE MODE ---
# Set the operation to perform. Available modes: "download", "port".
MODE="port"

# --- B) CACHE DIRECTORY ---
# All downloaded models will be stored here.
# It's recommended to use a directory on your scratch space.
CACHE_DIR="$HOME/scratch/huggingface/hub"


# =================================================================================
#  2. MODE-SPECIFIC CONFIGURATION
# =================================================================================

# --- C) DOWNLOAD MODE ---

# List of individual models to download.
MODELS_TO_DOWNLOAD=(
  "lbasile/llava_13b_6_21_complementary"
)

# List of collections to download. Leave empty to ignore.
COLLECTIONS_TO_DOWNLOAD=(
# ...
)


# --- D) PORT MODE ---

# This is the model whose architecture and processor config will be used as the
# 'template' for porting. The default contains the correct format.
REFERENCE_MODEL="llava-hf/llava-1.5-13b-hf"

# The Hugging Face organization to push the ported models to.
# This only applies when porting a 'collection' of checkpoints.
TARGET_ORG="lbasile"

# -- Option 1: Port a list of specified models --
# The source model and the target repository can be the same if you are just
# fixing a broken configuration. The number of models in each list must be equal.
SOURCE_MODELS_TO_PORT=("${1:-llava_7b}")
TARGET_REPOS_FOR_PORT=("${SOURCE_MODELS_TO_PORT[@]}")

# -- Option 2: Port a collection of checkpoints --
# These are predefined sets of models (e.g., all checkpoints for an experiment).
# Leave empty to ignore.
COLLECTIONS_TO_PORT=(
# ...
)


# =================================================================================
#  3. EXECUTION LOGIC (DO NOT EDIT BELOW THIS LINE)
# =================================================================================

# --- Script Constants ---
DOWNLOAD_SCRIPT="$PROJECT_ROOT/scripts/llava_models_utils/download_models.py"
PORT_SCRIPT="$PROJECT_ROOT/scripts/llava_models_utils/port_model_weights.py"

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
  # Add arguments for specified models if provided
  if [ ${#SOURCE_MODELS_TO_PORT[@]} -gt 0 ]; then
    PORT_ARGS+=(--source-model-id "${SOURCE_MODELS_TO_PORT[@]}")
    PORT_ARGS+=(--target-repo-id "${TARGET_REPOS_FOR_PORT[@]}")
  fi
  # Add arguments for collections if provided
  if [ ${#COLLECTIONS_TO_PORT[@]} -gt 0 ]; then
    PORT_ARGS+=(--collection "${COLLECTIONS_TO_PORT[@]}")
  fi

  if [ ${#PORT_ARGS[@]} -eq 0 ]; then
      echo "No models or collections specified for porting. Exiting."
      exit 0
  fi

  # This operation requires a GPU and specific memory configuration
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

  # Execute the python script with all arguments
  python "$PORT_SCRIPT" \
    --cache-dir "$CACHE_DIR" \
    --reference-model "$REFERENCE_MODEL" \
    --target-org "$TARGET_ORG" \
    "${PORT_ARGS[@]}"
}

# --- Main Execution ---

echo "================================================="
echo "        Llava Model Utilities"
echo "================================================="
echo "Mode: $MODE"
echo "Cache Dir: $CACHE_DIR"
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
