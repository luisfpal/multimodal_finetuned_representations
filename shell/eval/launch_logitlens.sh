#!/bin/bash
# =============================================================================
# Submit SLURM jobs to run logit lens evaluation on one or more models.
# Each model gets its own SLURM job running run_local_logitlens.sh.
#
# Usage: bash shell/eval/launch_logitlens.sh
# Paper (arXiv:2606.03871): logit lens, Figs. 17-18.
# =============================================================================

# Hardcoded path: edit this path for your setup.
PROJECT_ROOT="/u/dssc/lbasile/scratch/plm"

# Models to evaluate (one SLURM job per model)
models=(
    "lbasile/llava_7b"
    "lbasile/llava_13b"
    # "l11p/llava-v1.5-7b-pretrained-lbasile"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    "nyu-visionx/cambrian-8b"
    # "OpenGVLab/InternVL2-8B"
)

# SLURM settings
PARTITION="DGX"
GPU_TYPE="A100"
TIME="12:00:00"
MEM="100G"
CPUS=4

log_dir="$PROJECT_ROOT/logs/logs_logitlens"
mkdir -p "$log_dir"

for model_name in "${models[@]}"; do
    model_tag=$(basename "$model_name" | sed 's/[^a-zA-Z0-9._-]/_/g')
    echo "Submitting: $model_name"

    sbatch \
        -n1 -N1 -c"$CPUS" -p "$PARTITION" \
        --mem="$MEM" --time="$TIME" --gres=gpu:"$GPU_TYPE":1 \
        --job-name="logitlens_${model_tag}" \
        --output="${log_dir}/logitlens_${model_tag}_%j.out" \
        --error="${log_dir}/logitlens_${model_tag}_%j.err" \
        "$PROJECT_ROOT/shell/eval/run_local_logitlens.sh" "$model_name"
done
