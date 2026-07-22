#!/bin/bash
# Submits one information_imbalance.sh SLURM job per (model, dataset, ft_mode) combination.
# Usage: bash shell/embeddings_extraction/launch_information_imbalance.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(llava_13b llava_7b)
DATASETS=(mmbench vqav2 cocoqa gqa seed)
FT_MODES=(pre ft)

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for ft_mode in "${FT_MODES[@]}"; do
            echo "Submitting information_imbalance job: model=$model dataset=$dataset ft_mode=$ft_mode"
            sbatch --export=ALL,model="$model",dataset="$dataset",ft_mode="$ft_mode" \
                "$SCRIPT_DIR/information_imbalance.sh"
        done
    done
done

echo "All jobs submitted successfully"
