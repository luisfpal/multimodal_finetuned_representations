#!/bin/bash

# =============================================================================
# Configuration
# =============================================================================

# Default Settings (Standard Run)
ACCOUNT="dssc"
JOB_NAME="captioning"
PARTITION="DGX"
GPU_TYPE="A100"
NUM_GPUS=0
CPUS=16
MEM="64GB"
TIME="06:00:00"
NODELIST=""             # Empty = any available node
EXCLUDE=""              # Empty = no exclusions

# Project paths
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
LOG_DIR="$PROJECT_ROOT/logs"
OUTPUT_DIR="$PROJECT_ROOT/results/captions"

# Script Settings
NUM_WORKERS=32
SEED=42
PYTHON_ARGS=""

# Argument Parsing
IS_RETRY=false
SEED_SET=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --retry)
            IS_RETRY=true
            shift
            ;;
        --seed)
            SEED="$2"
            SEED_SET=true
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--retry] [--seed <int>]"
            exit 1
            ;;
    esac
done

# Configure Mode
if [ "$IS_RETRY" = true ]; then
    echo "🟠 CONFIGURING FOR RETRY MODE"
    JOB_NAME="captioning_retry"
    PARTITION="EPYC"
    CPUS=32
    MEM="48GB"
    TIME="02:00:00"
    PYTHON_ARGS="--retry-failed"
    
    # Default retry seed to 999 if not explicitly set
    if [ "$SEED_SET" = false ]; then
        SEED=999
    fi
else
    echo "🟢 CONFIGURING FOR STANDARD RUN"
fi

# =============================================================================
# Generate SBATCH Script & Submit
# =============================================================================

mkdir -p "$LOG_DIR"
SBATCH_SCRIPT=$(mktemp /tmp/sbatch_captioning.XXXXXX.sh)

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
source $PROJECT_ROOT/.venv/bin/activate

echo "🚀 Starting Captioning Job"
echo "   Mode: \$( [[ -n "$PYTHON_ARGS" ]] && echo "RETRY" || echo "STANDARD" )"
echo "   Seed: $SEED"
echo "   Output: $OUTPUT_DIR"

# Run captioning
python scripts/captioning/caption_dataset.py \
    $PYTHON_ARGS \
    --num-workers $NUM_WORKERS \
    --seed $SEED \
    --output-dir $OUTPUT_DIR

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