#!/bin/bash
# =============================================================================
# HOW TO USE
# =============================================================================
# This script submits a SLURM job for flexible evaluation scenarios on a
# SINGLE benchmark. The benchmark to run is NOT set here; it is set inside
# the `lmms_evaluation.sh` script.
#
# This launcher is for:
# - Evaluating a whole collection of model checkpoints against one benchmark.
# - Evaluating a single model against one benchmark.
#
# Usage:
# 1. Configure the SLURM settings below.
# 2. Go to `shell/eval/lmms_evaluation.sh` and uncomment the single `TASK_NAME`
#    you want to run.
# 3. Come back to this file and set the `EVAL_MODE` variable to choose what
#    set of models to run against that benchmark.
# 4. Run the script:
#   👉 bash shell/eval/launch_lmms.sh

# The model backend is now auto-inferred from the model name.
# =============================================================================

# SLURM settings
ACCOUNT="dssc"
JOB_NAME="experiment"
PARTITION="DGX"
GPU_TYPE="A100"
NUM_GPUS=1              # Number of GPUs for data parallelism (1 = single GPU, more stable)
CPUS=12
MEM="64GB"
TIME="01:00:00"
NODELIST=""             # Empty = any available node
EXCLUDE=""              # Empty = no exclusions

# Project paths (must be defined before JOB_COMMANDS heredoc)
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
SCRIPTS_DIR="$PROJECT_ROOT/shell/eval"
LOG_DIR="$PROJECT_ROOT/logs"
SCRIPT_NAME="lmms_evaluation.sh"

# =============================================================================
# EVALUATION MODE SELECTION - Choose ONE approach
# =============================================================================

# EVAL_MODE options:
#   "COLLECTION_NAME=<name>" - Single collection
#   "SINGLE_MODEL=<model_id>" - Specific model
#   "COLLECTIONS" - Loop all collections below
# EVAL_MODE="COLLECTIONS"
EVAL_MODE="SINGLE_MODEL=lbasile/llava_7b"
# EVAL_MODE="COLLECTION_NAME=Llava1.5-7B-0-16"

# Configuration
NO_IMAGE="false"

# Collections to evaluate (used when EVAL_MODE="COLLECTIONS")
COLLECTIONS=(
    "Llava1.5-7B-0-16"
    "Llava1.5-7B-16-32"
)

# Resolve model-specific venv
VENV="$PROJECT_ROOT/.venv"

# =============================================================================
# Job Execution Script
# =============================================================================

read -r -d '' JOB_COMMANDS << JOB_EOF || true
# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Robust CUDA configuration for both H100 (module) and A100 (manual)
setup_cuda() {
    # 1. Try module load (standard on H100)
    if command -v module &> /dev/null; then
        module load cuda/12.1 2>/dev/null || true
    fi

    # 2. If nvcc is missing/broken, try to find a valid existing installation
    #    This often happens on A100 nodes where modules might differ
    if ! command -v nvcc &> /dev/null || [ ! -x "\$(command -v nvcc)" ]; then
        # List of potential CUDA paths to check (prefer 12.1, then 12.x, then default)
        local candidates=("/usr/local/cuda-12.1" "/usr/local/cuda-12.2" "/usr/local/cuda-12" "/usr/local/cuda")

        for path in "\${candidates[@]}"; do
            if [ -x "\$path/bin/nvcc" ]; then
                echo "🔧 Manually configuring CUDA from: \$path"
                export CUDA_HOME="\$path"
                export PATH="\$path/bin:\$PATH"
                export LD_LIBRARY_PATH="\$path/lib64:\$LD_LIBRARY_PATH"
                break
            fi
        done
    fi

    # Verification and Enforcement
    if command -v nvcc &> /dev/null; then
        local nvcc_path=\$(command -v nvcc)
        # Resolve symlinks if possible, or assume standard structure
        local cuda_root=\$(dirname \$(dirname "\$nvcc_path"))

        echo "✓ CUDA active: \$nvcc_path"
        echo "  → Enforcing CUDA_HOME=\$cuda_root"

        export CUDA_HOME="\$cuda_root"
        export PATH="\$cuda_root/bin:\$PATH"
        export LD_LIBRARY_PATH="\$cuda_root/lib64:\$LD_LIBRARY_PATH"
    else
        echo "⚠️  WARNING: nvcc not found! Compilation may fail."
    fi
}
setup_cuda

TOKENIZERS_PARALLELISM="false"

# Load environment variables from .env (API keys, PYTHONPATH, etc.)
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Activate virtual environment
source "$VENV/bin/activate"

# Change to working directory
cd "$SCRIPTS_DIR"

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=\$TOKENIZERS_PARALLELISM

# Export NUM_GPUS for data parallelism
export NUM_GPUS=$NUM_GPUS

# Export Configuration
export NO_IMAGE=$NO_IMAGE

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
