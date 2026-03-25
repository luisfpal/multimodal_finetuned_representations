#!/bin/bash

PROJECT_ROOT="/u/dssc/lbasile/scratch/plm"
source "$PROJECT_ROOT/.venv/bin/activate"

# EXTRA_ARGS=""
EXTRA_ARGS="--no-image"

checkpoints=(
#     "llava_7b_8_32"
    "internvl2"
# "llava_7b_frozen"
#     "llava_7b_0_2"
#     "llava_7b_12_32"
#     "llava_7b_0_6"
# "llava_7b_16spaced"
# "llava_7b_8spaced"
# "llava_7b"
# "llava_7b_0_16"
# "llava_7b_16_32"
# "llava_7b_6_17"
# "llava_7b_6_17_complementary"
# "llava_13b_6_21_complementary"
#     "llava_7b_0_12"
#     "llava_7b_0_16"
#     "llava_7b_0_24"
#     "llava_7b_16_32"
#     "llava_7b_24_32"
#     "llava_7b_30_32"
)
# checkpoints=(
#     "llava_7b_28_32"
#     "llava_7b_20_32"
#     "llava_7b_26_32"
#     "llava_7b"
# )

# Create a logs directory for clean output management
log_dir="$PROJECT_ROOT/logs/logs_eval"
mkdir -p "$log_dir"

for ckpt in "${checkpoints[@]}"; do
    echo "Submitting eval job for checkpoint: $ckpt"
    
    # Submit the job using sbatch with specified resources
    sbatch \
        -n1 -N1 -c4 -p H100 -A lade --mem=100G --time=4:0:0 --gres=gpu:H100:1 \
        --job-name="eval_${ckpt}" \
        --output="${log_dir}/eval_${ckpt}_%j.out" \
        --error="${log_dir}/eval_${ckpt}_%j.err" \
        shell/eval/run_local_evaluation.sh "OpenGVLab/InternVL2-8B" "$EXTRA_ARGS"
done
