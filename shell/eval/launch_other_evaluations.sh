#!/bin/bash

# =============================================================================
# Configuration
# =============================================================================

# SLURM settings
ACCOUNT="dssc"
JOB_NAME="experiment"
PARTITION="DGX"
GPU_TYPE="A100"
NUM_GPUS=1
CPUS=16
MEM="32GB"
TIME="01:00:00"
NODELIST=""             # Empty = any available node
EXCLUDE=""              # Empty = no exclusions

# Project paths
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
SHELL_SCRIPTS_DIR="$PROJECT_ROOT/shell/eval"
LOG_DIR="$PROJECT_ROOT/logs"

# Script to launch
# SCRIPT_NAME="logitlens_evaluation.sh"
SCRIPT_NAME="ablation_layers_evaluation.sh"

# =============================================================================
# Generate SBATCH Script & Submit
# =============================================================================

mkdir -p "$LOG_DIR"
SBATCH_SCRIPT=$(mktemp /tmp/sbatch_other_evaluations.XXXXXX.sh)

# Build SBATCH directives with conditional nodelist/exclude
SBATCH_DIRECTIVES="#SBATCH --account=$ACCOUNT
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$GPU_TYPE:$NUM_GPUS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --error=$LOG_DIR/slurm-%j.out"

[[ -n "$NODELIST" ]] && SBATCH_DIRECTIVES+="
#SBATCH --nodelist=$NODELIST"
[[ -n "$EXCLUDE" ]] && SBATCH_DIRECTIVES+="
#SBATCH --exclude=$EXCLUDE"

cat <<EOT >"$SBATCH_SCRIPT"
#!/bin/bash
$SBATCH_DIRECTIVES

bash $SCRIPT_NAME
EOT

# Submit Job
echo "🟢 Submitting job: $JOB_NAME ($SCRIPT_NAME)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

JOB_OUTPUT=$(sbatch "$SBATCH_SCRIPT" 2>&1)

if [ $? -eq 0 ]; then
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
    echo "✅ Job ID: $JOB_ID"
    echo "📁 Output: $LOG_DIR/slurm-$JOB_ID.out"
    echo ""
    echo "Monitor: tail -f $LOG_DIR/slurm-$JOB_ID.out"
else
    echo "❌ Failed: $JOB_OUTPUT"
    exit 1
fi
