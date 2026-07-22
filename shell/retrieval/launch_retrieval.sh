#!/bin/bash
# Submits one retrieval.sh SLURM job per model.
# Usage: bash shell/retrieval/launch_retrieval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(internvl_8b_hf onevision_8b_hf onevision_4b_hf llava_13b llava_7b cambrian_8b)

for model in "${MODELS[@]}"; do
    echo "Submitting retrieval job: model=$model"
    sbatch --export=ALL,model="$model" "$SCRIPT_DIR/retrieval.sh"
done

echo "All jobs submitted successfully"
