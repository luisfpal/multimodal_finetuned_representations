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
# Hardcoded path: edit this path for your setup.
PROJECT_ROOT="/u/dssc/lbasile/scratch/plm"

# EXTRA_ARGS=""
EXTRA_ARGS="--no-image"

# Models to evaluate (one SLURM job per model)
models=(
    # "mistral-community/pixtral-12b"
    # "lbasile/llava_13b"
    # "lbasile/llava_7b_frozen"
    # "lbasile/llava_7b_0_4"
    # "lbasile/llava_7b_0_6"
    # "lbasile/llava_7b_0_8"
    # "lbasile/llava_7b_0_12"
    # "lbasile/llava_7b_0_16"
    # "lbasile/llava_7b_0_24"
    # "lbasile/llava_7b_8_32"
    # "lbasile/llava_7b_12_32"
    # "lbasile/llava_7b_16_32"
    # "lbasile/llava_7b_20_32"
    # "lbasile/llava_7b_24_32"
    # "lbasile/llava_7b_26_32"
    # "lbasile/llava_7b_28_32"
    # "lbasile/llava_7b_30_32"
    # "lbasile/llava_7b_6_17"
    # "lbasile/llava_7b_6_17_complementary"
    # "lbasile/llava_13b_6_21"
    # "lbasile/llava_13b_6_21_complementary"
    # "nyu-visionx/cambrian-34b"
    # "nyu-visionx/cambrian-8b"
    # "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    # "lbasile/LLaVA-OneVision-1.5-4B"
    # "lbasile/LLaVA-OneVision-1.5-4B-0-18"
    # "lbasile/LLaVA-OneVision-1.5-4B-18-36"
    # "OpenGVLab/InternVL2-8B"
)

# SLURM settings
PARTITION="DGX"
GPU_TYPE="A100"
TIME="12:00:00"
MEM="100G"
CPUS=4

log_dir="$PROJECT_ROOT/logs/logs_eval"
mkdir -p "$log_dir"

for model_name in "${models[@]}"; do
    model_tag=$(basename "$model_name" | sed 's/[^a-zA-Z0-9._-]/_/g')
    echo "Submitting: $model_name"

    sbatch \
        -n1 -N1 -c"$CPUS" -p "$PARTITION"\
        --mem="$MEM" --time="$TIME" --gres=gpu:"$GPU_TYPE":1 \
        --job-name="eval_${model_tag}" \
        --output="${log_dir}/eval_${model_tag}_%j.out" \
        --error="${log_dir}/eval_${model_tag}_%j.err" \
        "$PROJECT_ROOT/shell/eval/run_local_evaluation.sh" "$model_name" "$EXTRA_ARGS"
done
