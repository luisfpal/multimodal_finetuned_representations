#!/bin/bash

# =============================================================================
# Configuration
# =============================================================================

# SLURM settings
ACCOUNT="dssc"
JOB_NAME="experiment"
NUM_GPUS=1              # Number of GPUs for data parallelism (1 = single GPU, more stable)
CPUS=16
MEM="48GB"
TIME="08:00:00"        
NODELIST="dgx001"             # Empty = any available node
EXCLUDE="dgx002"              # Empty = no exclusions

# Project paths (must be defined before JOB_COMMANDS heredoc)
PROJECT_ROOT="$HOME/multimodal_representations_analysis"
SCRIPTS_DIR="$PROJECT_ROOT/shell/eval"
LOG_DIR="$PROJECT_ROOT/logs"
SCRIPT_NAME="lmms_evaluation.sh"

# =============================================================================
# EVALUATION MODE SELECTION - Choose ONE approach
# =============================================================================

# EVAL_MODE options:
#   "COLLECTION_NAME=<name>" - Single collection
#   "FINETUNED_PRETRAINED=true" - Finetuned preset
#   "SINGLE_MODEL=<model_id>" - Specific model
#   "COLLECTIONS" - Loop all collections below
EVAL_MODE="FINETUNED_PRETRAINED=true"
# EVAL_MODE="COLLECTIONS"
# EVAL_MODE="SINGLE_MODEL=l11p/llava-v1.5-7b-lbasile-full-step-1000"
# EVAL_MODE="COLLECTION_NAME=Llava1.5-7B-16-32"

# Collections to evaluate (used when EVAL_MODE="COLLECTIONS")
COLLECTIONS=(
    "Llava1.5-7B"
    "Llava1.5-7B-frozen"
    "Llava1.5-7B-0-16"
    "Llava1.5-7B-16-32"
)

# =============================================================================
# Job Execution Script
# =============================================================================

read -r -d '' JOB_COMMANDS << JOB_EOF || true
# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

TOKENIZERS_PARALLELISM="false"

# Load environment variables from .env (API keys, PYTHONPATH, etc.)
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

# Export NUM_GPUS for data parallelism
export NUM_GPUS=$NUM_GPUS

# =============================================================================
# ORFEO LLM JWT AUTHENTICATION (if configured)
# =============================================================================
if [ -n "\${ORFEO_AUTH_URL}" ] && [ -n "\${ORFEO_CLIENT_ID}" ]; then
    echo "🔐 Generating JWT token for Orfeo LLM..."
    JWT=\$(curl -s -X POST "\${ORFEO_AUTH_URL}" \\
      -H "Content-Type: application/x-www-form-urlencoded" \\
      -d "grant_type=password" \\
      -d "client_id=\${ORFEO_CLIENT_ID}" \\
      -d "username=\${ORFEO_USERNAME}" \\
      -d "password=\${ORFEO_CLIENT_SECRET}" \\
      -d "scope=profile" | jq -r '.access_token')
    
    if [ -z "\$JWT" ] || [ "\$JWT" = "null" ]; then
        echo "❌ Failed to get JWT token from Orfeo!"
        exit 1
    fi
    
    echo "✓ JWT token acquired"
    
    # Export for Python/lmms-eval
    export OPENAI_API_KEY="\$JWT"
    export OPENAI_API_URL="\${ORFEO_LLM_URL}"
    export MODEL_VERSION="\${ORFEO_MODEL}"
fi

# =============================================================================
# RUN EVALUATION
# =============================================================================

if [ "$EVAL_MODE" = "COLLECTIONS" ]; then
    for COLLECTION in "\${COLLECTIONS[@]}"; do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "▶ STARTING EVALUATION: \$COLLECTION"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        COLLECTION_NAME="\$COLLECTION" "$SCRIPTS_DIR/$SCRIPT_NAME"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✓ COMPLETED: \$COLLECTION"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    done
else
    $EVAL_MODE "$SCRIPTS_DIR/$SCRIPT_NAME"
fi
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