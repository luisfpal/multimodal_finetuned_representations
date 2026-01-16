#!/bin/bash

# =============================================================================
# Configuration
# =============================================================================

# SLURM settings
ACCOUNT="dssc"
JOB_NAME="experiments"
PARTITION="EPYC" # Or DGX/H100 if GPU is needed for porting
GPU_TYPE="" # e.g., A100. Leave empty for CPU jobs (downloading)
NUM_GPUS=0 # 1 for porting, 0 for downloading
CPUS=8
MEM="48GB"
TIME="00:20:00"
NODELIST=""             # Empty = any available node
EXCLUDE=""              # Empty = no exclusions

# Project paths
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
SCRIPTS_DIR="$PROJECT_ROOT/shell/llava_models_utils"
LOG_DIR="$PROJECT_ROOT/logs/model_utils"

# Script selection
SCRIPT_NAME="run_model_utils.sh"

# =============================================================================
# Job Execution Script
# =============================================================================

read -r -d '' JOB_COMMANDS << JOB_EOF || true
# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Load environment variables from .env (e.g., HF_TOKEN)
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Change to the script's directory
cd "$SCRIPTS_DIR"

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

# =============================================================================
# RUN SCRIPT
# =============================================================================

bash "$SCRIPT_NAME"
JOB_EOF

# =============================================================================
# Generate SBATCH Script
# =============================================================================

mkdir -p "$LOG_DIR"
SBATCH_SCRIPT=$(mktemp /tmp/sbatch_model_utils.XXXXXX.sh)

# Build SBATCH directives
SBATCH_DIRECTIVES="#SBATCH --account=$ACCOUNT
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --error=$LOG_DIR/slurm-%j.out"

# Add GPU directive only if NUM_GPUS > 0
if [ "$NUM_GPUS" -gt 0 ]; then
    SBATCH_DIRECTIVES+="
#SBATCH --gres=gpu:$GPU_TYPE:$NUM_GPUS"
fi

# Add node constraints if specified
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
echo "Logs will be saved to: $LOG_DIR"

JOB_OUTPUT=$(sbatch "$SBATCH_SCRIPT" 2>&1)

if [ $? -eq 0 ]; then
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
    echo "✅ Job ID: $JOB_ID"
    echo "📁 Output: $LOG_DIR/slurm-$JOB_ID.out"
    echo "👀 Monitor with: tail -f $LOG_DIR/slurm-$JOB_ID.out"
else
    echo "❌ Failed to submit job: $JOB_OUTPUT"
    # cat "$SBATCH_SCRIPT" # Uncomment for debugging the sbatch script
    exit 1
fi

# Clean up the temporary script
rm "$SBATCH_SCRIPT"
