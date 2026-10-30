#!/bin/bash
# Sweeps the 6 VLMs across rep types (norm1, res2). Each iteration submits
# one SLURM job that runs linguistic_probing.py for a single
# (model, target_layer); the script always extracts both text_avg and
# last_text pools in one pass. Results land in:
#   $REPRESENTATIONS_DIR/probing_tasks/{model_name}/{target_layer}/{text_avg|last_text}/
# Usage: bash shell/probing/launch_probing.sh
# Paper (arXiv:2606.03871): linguistic probing, Fig. 20.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(
    llava-hf/llava-1.5-7b-hf
    llava-hf/llava-1.5-13b-hf
    lmms-lab/LLaVA-OneVision-1.5-4B-Instruct
    lmms-lab/LLaVA-OneVision-1.5-8B-Instruct
    OpenGVLab/InternVL2-8B
    nyu-visionx/cambrian-8b
)

for model_path in "${MODELS[@]}"; do
    for target_layer in norm1 res2; do
        echo "Submitting probing job: model_path=$model_path target_layer=$target_layer"
        sbatch --export=ALL,model_path="$model_path",target_layer="$target_layer" \
            "$SCRIPT_DIR/probing.sh"
    done
done

echo "All jobs submitted successfully"
