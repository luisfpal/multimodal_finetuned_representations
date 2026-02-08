#!/bin/bash

# =============================================================================
# Configuration
# =============================================================================

# Default Settings (Evaluation Run)
ACCOUNT="dssc"
JOB_NAME="experiment"
PARTITION="DGX"
GPU_TYPE="A100"
NUM_GPUS=1
CPUS=16
MEM="64GB"
TIME="04:00:00"         # Estimated time, adjusted from 6h
NODELIST=""             # Empty = any available node
EXCLUDE=""              # Empty = no exclusions

# Project paths
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
LOG_DIR="$PROJECT_ROOT/logs"

# =============================================================================
# Generate SBATCH Script & Submit
# =============================================================================

mkdir -p "$LOG_DIR"

SBATCH_SCRIPT=$(mktemp /tmp/sbatch_caption_eval.XXXXXX.sh)

# Build SBATCH directives
SBATCH_DIRECTIVES="#SBATCH --account=$ACCOUNT
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --error=$LOG_DIR/slurm-%j.out"

# Add optional directives
[[ -n "$NODELIST" ]] && SBATCH_DIRECTIVES+="
#SBATCH --nodelist=$NODELIST"
[[ -n "$EXCLUDE" ]] && SBATCH_DIRECTIVES+="
#SBATCH --exclude=$EXCLUDE"
[[ $NUM_GPUS -gt 0 ]] && SBATCH_DIRECTIVES+="
#SBATCH --gres=gpu:$GPU_TYPE:$NUM_GPUS"

# Write the script
cat <<EOT >"$SBATCH_SCRIPT"
#!/bin/bash
$SBATCH_DIRECTIVES

# Activate environment
cd $PROJECT_ROOT
# Using the same environment as captioning, assuming it has the necessary libs
source $PROJECT_ROOT/.venv/bin/activate 

# Run evaluation
python scripts/captioning/evaluate_captions_semantically.py --model jina

# Run comparison and stats generation immediately after
if [ \$? -eq 0 ]; then
    echo "✅ Evaluation complete. Running statistical comparison..."
    python scripts/captioning/compare_captions_scores.py
else
    echo "❌ Evaluation failed. Skipping comparison."
    exit 1
fi

EOT

# Submit Job
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
JOB_OUTPUT=$(sbatch "$SBATCH_SCRIPT" 2>&1)

if [ $? -eq 0 ]; then
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
    echo "✅ Job ID: $JOB_ID"
    echo "📁 Output: $LOG_DIR/slurm-$JOB_ID.out"
    echo ""
    echo "To monitor: tail -f $LOG_DIR/slurm-$JOB_ID.out"
else
    echo "❌ Failed: $JOB_OUTPUT"
    exit 1
fi
