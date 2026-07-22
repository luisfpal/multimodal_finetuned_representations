#!/bin/bash
# Submits one extract_embeddings.sh SLURM job per model.
# Usage: bash shell/embeddings_extraction/launch_extract_embeddings.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    "OpenGVLab/InternVL2-8B"
    "lbasile/llava_13b"
    "lbasile/llava_7b"
    "nyu-visionx/cambrian-8b"
)

for model_path in "${MODELS[@]}"; do
    echo "Submitting extract_embeddings job: model_path=$model_path"
    sbatch --export=ALL,model_path="$model_path" "$SCRIPT_DIR/extract_embeddings.sh"
done

echo "All jobs submitted successfully"
