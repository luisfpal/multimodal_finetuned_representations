#!/bin/bash

# =============================================================================
# Usage Instructions
# =============================================================================
# This script submits a SLURM job to caption images from a specified dataset using
# a VLM. It handles environment setup, configuration, and job submission.
#
# Usage:
#   ./launch_captioning.sh [options]
#
# Options:
#   --dataset <name>   : Dataset to process (default: gqa).
#                        Choices: mme, gqa, mmbench_en_dev, mmbench_en_test,
#                                 vqav2_coco, cocoqa, custom
#   --seed <int>       : Random seed for reproducibility (default: 42).
#   --retry            : Retry mode - only processes images that previously failed.
#                        (Auto-configures for CPU partition and short duration).
#
# Examples:
#   # Standard run for GQA (default)
#   ./launch_captioning.sh
#
#   # Run for VQAv2 COCO dataset
#   ./launch_captioning.sh --dataset vqav2_coco
#
#   # Retry failed captions for COCOQA with a specific seed
#   ./launch_captioning.sh --dataset cocoqa --retry --seed 123
# =============================================================================

# =============================================================================
# 🟢 USER CONFIGURATION - EDIT THESE SETTINGS
# =============================================================================

# --- Compute Resources ---
ACCOUNT="dssc"
PARTITION="DGX"
GPU_TYPE="A100"
NUM_GPUS=0          # Set to 1 if GPU needed, else 0
CPUS=16
MEM="64GB"
TIME="03:00:00"
NODELIST=""         # Optional: specific node (e.g., "node01")
EXCLUDE=""          # Optional: exclude nodes (e.g., "node02")

# --- Project Paths ---
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
LOG_DIR="$PROJECT_ROOT/logs"
OUTPUT_DIR="$PROJECT_ROOT/results/captions"

# --- Default Job Parameters ---
DEFAULT_NUM_WORKERS=1
DEFAULT_SEED=42
DEFAULT_DATASET="gqa"

# =============================================================================
# 🔴 END OF USER CONFIGURATION - INTERNAL LOGIC BELOW
# =============================================================================

# Initialize variables with defaults
NUM_WORKERS=$DEFAULT_NUM_WORKERS
SEED=$DEFAULT_SEED
DATASET=$DEFAULT_DATASET
PYTHON_ARGS=""
IS_RETRY=false
SEED_SET=false

# Argument Parsing
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
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--retry] [--seed <int>] [--dataset <name>]"
            exit 1
            ;;
    esac
done

# Configure Mode (Retry vs Standard)
JOB_NAME="experiment"

if [ "$IS_RETRY" = true ]; then
    echo "🟠 CONFIGURING FOR RETRY MODE"
    JOB_NAME="experiment"
    # Override compute settings for retry (lighter weight)
    PARTITION="EPYC"
    CPUS=16
    MEM="48GB"
    TIME="00:20:00"
    PYTHON_ARGS="--retry-failed"

    # Default retry seed to 999 if not explicitly set to avoid deterministic repeat of failure
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
#SBATCH --output=$LOG_DIR/slurm-%x-%j.out
#SBATCH --error=$LOG_DIR/slurm-%x-%j.out"

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
echo "   Dataset: $DATASET"
echo "   Mode: \$( [[ -n "$PYTHON_ARGS" ]] && echo "RETRY" || echo "STANDARD" )"
echo "   Seed: $SEED"
echo "   Output: $OUTPUT_DIR"

# Run captioning
python scripts/captioning/caption_dataset.py \
    $PYTHON_ARGS \
    --dataset $DATASET \
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
    echo "📁 Output: $LOG_DIR/slurm-${JOB_NAME}-${JOB_ID}.out"
    echo ""
    echo "To monitor: tail -f $LOG_DIR/slurm-${JOB_NAME}-${JOB_ID}.out"
else
    echo "❌ Failed: $JOB_OUTPUT"
    exit 1
fi
