#!/bin/bash

PROJECT_ROOT="/u/dssc/lbasile/scratch/plm"
source "$PROJECT_ROOT/.venv/bin/activate"

# EXTRA_ARGS="${ONEVISION_EXTRA_ARGS:-}"
EXTRA_ARGS="--no-image"
# EXTRA_ARGS=""

checkpoints=(
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
)

log_dir="$PROJECT_ROOT/logs/logs_eval"
mkdir -p "$log_dir"

for model_name in "${checkpoints[@]}"; do
    model_tag=$(basename "$model_name" | sed 's/[^a-zA-Z0-9._-]/_/g')
    echo "Submitting eval job for model: $model_name"

    sbatch \
        -n1 -N1 -c4 -p DGX --mem=100G --time=4:0:0 --gres=gpu:A100:1 \
        --export=ALL,LMMS_EVAL_MODEL=llava_hf \
        --job-name="eval_${model_tag}" \
        --output="${log_dir}/eval_${model_tag}_%j.out" \
        --error="${log_dir}/eval_${model_tag}_%j.err" \
        shell/eval/run_local_evaluation_onevision.sh "$model_name" "$EXTRA_ARGS"
done
