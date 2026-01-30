#!/bin/bash

# =============================================================================
# Configuration
# =============================================================================

# SLURM settings
ACCOUNT="lade"
JOB_NAME="experiment"
PARTITION="H100"
GPU_TYPE="H100"
NUM_GPUS=1              # Number of GPUs
CPUS=16
MEM="128GB"
TIME="05:00:00"
NODELIST=""             # Empty = any available node
EXCLUDE=""              # Empty = no exclusions

# Project paths
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
SCRIPTS_DIR="$PROJECT_ROOT/shell/representations_interpretability"
LOG_DIR="$PROJECT_ROOT/logs"

# Script selection
# SCRIPT_NAME="models_similarity.sh"  # For model vs model similarity analysis
SCRIPT_NAME="multimodality_similarity.sh"  # For multimodality similarity analysis
# SCRIPT_NAME="intrinsic_dimension.sh"  # For intrinsic dimension analysis
# SCRIPT_NAME="intra_model_similarity.sh"  # For intra-model similarity analysis
# SCRIPT_NAME="multimodality_specific_pooling_similarity.sh" # For multimodality with specific pooling analysis

# =============================================================================
# Centralized Configuration for All Analysis Scripts
# =============================================================================

# Dataset parameters
# For COCOQA, VQAv2, MME and GQA use:
export GUIDE_TEXT="Answer the question using a single word or phrase.\n"
# For MMBench use:
# export GUIDE_TEXT="Answer with the option's letter from the given choices directly."
export DOWNSAMPLE_SIZE="10000"

# Representations storage configuration
export REPRESENTATIONS_DIR="$HOME/scratch"
export SAVE_REPRESENTATIONS="true"  # Set to "false" to disable saving

# =============================================================================
# Job Execution Script
# =============================================================================

read -r -d '' JOB_COMMANDS << JOB_EOF || true
# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

TOKENIZERS_PARALLELISM="false"

# Load environment variables from .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Change to working directory
cd "$SCRIPTS_DIR"

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=\$TOKENIZERS_PARALLELISM

# =============================================================================
# RUN ANALYSIS
# =============================================================================

bash "$SCRIPT_NAME"
JOB_EOF

# =============================================================================
# Generate SBATCH Script
# =============================================================================

mkdir -p "$LOG_DIR"
SBATCH_SCRIPT=$(mktemp /tmp/sbatch_experiment.XXXXXX.sh)

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

$JOB_COMMANDS
EOT

# =============================================================================
# Submit Job
# =============================================================================

echo "🟢 Submitting job: $SCRIPT_NAME"
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
