#!/bin/bash
# ====================== Configuration ====================== #
ACCOUNT="dssc"
JOB_NAME="experiment"
PARTITION="DGX"
GPUS=1
CPUS=16

TIME="00:20:00"
MEM="64GB"

# TIME="06:00:00"
# MEM="100GB"
# MEM="564GB"
# TIME="06:00:00"

PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
LOG_DIR="$PROJECT_ROOT/.experiments_logs"
mkdir -p "$LOG_DIR"

# SCRIPT="compute_id_correlation.sh"
# SCRIPT="modalities_similarities.sh"
# SCRIPT="transplanting_layers_qa_benchmarking.sh"
# SCRIPT="transplanting_layers_caption_benchmarking.sh"
SCRIPT="pretrained_and_finetuned_benchmarking.sh"

# ====================== SBATCH Script Creation ====================== #
SBATCH_SCRIPT=$(mktemp /tmp/sbatch_experiment.XXXXXX.sh)

cat <<EOT >"$SBATCH_SCRIPT"
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$GPUS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --error=$LOG_DIR/slurm-%j.out

# Activate environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Run script
bash "$PROJECT_ROOT/shell/$SCRIPT"
EOT

# ====================== Job Submission ====================== #
echo "🟢 Submitting SBATCH job for experiment: $SCRIPT"
echo "----------------------------------------------------"
cat "$SBATCH_SCRIPT"
echo "----------------------------------------------------"

JOB_ID=$(sbatch "$SBATCH_SCRIPT" | awk '{print $4}')
echo "✅ Job submitted with ID: $JOB_ID"
echo "📁 Unified output: $LOG_DIR/slurm-$JOB_ID.out"
