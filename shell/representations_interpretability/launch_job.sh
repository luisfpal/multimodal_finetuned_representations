#!/bin/bash

# =============================================================================
# Configuration
# =============================================================================

# SLURM settings
ACCOUNT="dssc"
JOB_NAME="experiment"
NUM_GPUS=1              # Number of GPUs
CPUS=16
MEM="64GB"
TIME="03:00:00"
NODELIST=""             # Empty = any available node
EXCLUDE=""              # Empty = no exclusions

# Project paths
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
SCRIPTS_DIR="$PROJECT_ROOT/shell/representations_interpretability"
LOG_DIR="$PROJECT_ROOT/logs"

# Script selection
SCRIPT_NAME="models_similarity.sh"  # For model vs model similarity analysis
# SCRIPT_NAME="multimodality_similarity.sh"  # For multimodality similarity analysis

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
# Auto-detect GPU partition
# =============================================================================

GPU_INFO=$(sinfo -h -o "%P %G" 2>/dev/null | grep -iE "A100|H100" | head -1)
[[ -z "$GPU_INFO" ]] && GPU_INFO=$(sinfo -h -o "%P %G" 2>/dev/null | grep -i gpu | head -1)

if [[ -n "$GPU_INFO" ]]; then
    PARTITION=$(echo "$GPU_INFO" | awk '{gsub(/\*/, ""); print $1}')
    GPU_TYPE=$(echo "$GPU_INFO" | awk -F: '{print $2}')
    echo "🔍 Detected: PARTITION=$PARTITION, GPU=$GPU_TYPE"
else
    PARTITION="DGX"; GPU_TYPE="A100"
    echo "⚠️  Defaults: PARTITION=$PARTITION, GPU=$GPU_TYPE"
fi

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
