#!/bin/bash
# Submits one layer_ablation_multimodal_evaluation.sh SLURM job per benchmark task.
# Usage: bash shell/eval/launch_layer_ablation_multimodal.sh
# Paper (arXiv:2606.03871): layer-skipping / cross-modal attention knockout, Figs. 2, 6, 8-11, 32.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="Meta-Llama-3-90B-instruct"
ABLATION_MODE="cross_modal"
WINDOW_SIZE=0
START_LAYER=0

TASKS=(
    gqa cv_bench seedbench mme vqav2_val mmbench_en_dev mmbench_cn_dev
    scienceqa_img countbench mmstar realworldqa mmmu_pro_standard pope
    mmmu_val chartqa
)

for task in "${TASKS[@]}"; do
    echo "Submitting layer_ablation_multimodal_evaluation job: model_path=$MODEL_PATH task=$task"
    sbatch --export=ALL,model_path="$MODEL_PATH",task="$task",window_size="$WINDOW_SIZE",start_layer="$START_LAYER",ablation_mode="$ABLATION_MODE" \
        "$SCRIPT_DIR/layer_ablation_multimodal_evaluation.sh"
done

echo "All jobs submitted successfully"
