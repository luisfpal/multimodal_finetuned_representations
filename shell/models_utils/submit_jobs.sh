#!/bin/bash

# List of checkpoints to process
# You can add or remove checkpoints from this list as needed
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"

checkpoints=(
    "llava_13b_6_21_complementary"
)

# Create a logs directory for clean output management
log_dir="logs_port"
mkdir -p "$log_dir"

for ckpt in "${checkpoints[@]}"; do
    echo "Submitting port job for checkpoint: $ckpt"
    
    # Submit the job using sbatch with specified resources
    sbatch \
        -n1 -N1 -c4 -p GENOA -A lade --mem=100G --time=2:0:0 \
        --job-name="port_${ckpt}" \
        --output="${log_dir}/port_${ckpt}_%j.out" \
        --error="${log_dir}/port_${ckpt}_%j.err" \
        shell/models_utils/run_model_utils.sh "lbasile/$ckpt"
done
