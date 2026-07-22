#!/bin/bash
# Submits one layer_ablation_text_evaluation.sh SLURM job per text-only benchmark task.
# Usage: bash shell/eval/launch_layer_ablation_text.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASKS=(mmlu hellaswag nq_open triviaqa)
WINDOW_SIZE=4

for task in "${TASKS[@]}"; do
    echo "Submitting layer_ablation_text_evaluation job: task=$task window_size=$WINDOW_SIZE"
    sbatch --export=ALL,task="$task",window_size="$WINDOW_SIZE" \
        "$SCRIPT_DIR/layer_ablation_text_evaluation.sh"
done

echo "All text jobs submitted successfully"
