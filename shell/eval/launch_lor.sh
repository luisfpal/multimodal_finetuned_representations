#!/bin/bash
# =============================================================================
# Submit SLURM jobs to evaluate one or more models.
# Each model gets its own SLURM job running run_local_evaluation.sh.
#
# Usage: bash shell/eval/launch_lor.sh
#
# Configure models in the array below. Backend is auto-inferred.
# =============================================================================

# PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
PROJECT_ROOT="/u/dssc/lbasile/scratch/plm"

# EXTRA_ARGS=""
EXTRA_ARGS="--no-image"

# Models to evaluate (one SLURM job per model)
models=(
    "OpenGVLab/InternVL2-8B"
    # "lbasile/llava_7b"
    # "nyu-visionx/cambrian-8b"
    # "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    # "OpenGVLab/InternVL3_5-8B"
)

# SLURM settings
PARTITION="DGX"
GPU_TYPE="A100"
TIME="4:00:00"
MEM="100G"
CPUS=4

log_dir="$PROJECT_ROOT/logs/logs_eval"
mkdir -p "$log_dir"

for model_name in "${models[@]}"; do
    model_tag=$(basename "$model_name" | sed 's/[^a-zA-Z0-9._-]/_/g')
    echo "Submitting: $model_name"

    sbatch \
        -n1 -N1 -c"$CPUS" -p "$PARTITION" -A dssc \
        --mem="$MEM" --time="$TIME" --gres=gpu:"$GPU_TYPE":1 \
        --job-name="eval_${model_tag}" \
        --output="${log_dir}/eval_${model_tag}_%j.out" \
        --error="${log_dir}/eval_${model_tag}_%j.err" \
        "$PROJECT_ROOT/shell/eval/run_local_evaluation.sh" "$model_name" "$EXTRA_ARGS"
done
