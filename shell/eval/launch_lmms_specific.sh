#!/bin/bash
# =============================================================================
# HOW TO USE
# =============================================================================
# This script submits a SLURM job to evaluate ONE specific model against ALL
# benchmarks defined in the `TASKS` array inside the script.
#
# It is designed for comprehensive, non-interactive evaluation runs on a
# specific model checkpoint.
#
# Usage:
# 1. Configure the SLURM settings below (e.g., NUM_GPUS, PARTITION).
# 2. Set the `MODEL_TO_EVAL` variable to the Hugging Face model ID you want
#    to evaluate.
# 3. Run the script:
#    bash shell/eval/launch_lmms_specific.sh
# =============================================================================

# SLURM settings
ACCOUNT="dssc"
JOB_NAME="experiment"
PARTITION="DGX"
GPU_TYPE="A100"
NUM_GPUS=3
CPUS=16
MEM="72GB"
TIME="02:00:00"
NODELIST="dgx002"
EXCLUDE=""

# Project paths
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
SCRIPTS_DIR="$PROJECT_ROOT/shell/eval"
LOG_DIR="$PROJECT_ROOT/logs"

# Use the new DYNAMIC runner script
SCRIPT_NAME="lmms_evaluation.sh"

# =============================================================================
# MODEL SELECTION
# =============================================================================

# Uncomment the model you want to evaluate:
# MODEL_TO_EVAL="l11p/llava-v1.5-7b-lbasile-layers-8-16-step-10000"
# MODEL_TO_EVAL="l11p/llava-v1.5-7b-lbasile-layers-0-8-step-10000"
# MODEL_TO_EVAL="l11p/llava-7b-lbasile"
# MODEL_TO_EVAL="l11p/llava-v1.5-7b-lbasile-full-step-10000"
MODEL_TO_EVAL="llava-hf/llava-1.5-7b-hf"


# =============================================================================
# Job Execution Script
# =============================================================================

read -r -d '' JOB_COMMANDS << JOB_EOF || true
# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

TASKS=(
    "scienceqa_img"
    "mme"
    "mmbench_en_dev"
    "textvqa_val"
    "llava_in_the_wild"
    "gqa"
    "vizwiz_vqa_val"
    "vqav2_val"
    "seedbench"
    "mmbench_cn_dev"
    "pope"
)

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

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$SCRIPTS_DIR"
chmod +x "$SCRIPT_NAME"

export TOKENIZERS_PARALLELISM=false
export NUM_GPUS=$NUM_GPUS

# =============================================================================
# ORFEO LLM JWT AUTHENTICATION
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
    # Export for Python/lmms-eval
    export OPENAI_API_KEY="\$JWT"
    export OPENAI_API_URL="\${ORFEO_LLM_URL}"
    export MODEL_VERSION="\${ORFEO_MODEL}"
fi

# =============================================================================
# RUN EVALUATION LOOP
# =============================================================================

# Export the single model to be picked up by the evaluation script
export SINGLE_MODEL="$MODEL_TO_EVAL"

for TASK in "\${TASKS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ STARTING EVALUATION: \$TASK"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Export task name for the dynamic script
    export TASK_NAME="\$TASK"

    # Run the dynamic evaluation script
    ./"\$SCRIPT_NAME"

    echo ""
    echo "✓ COMPLETED: \$TASK"
done
JOB_EOF

# =============================================================================
# Generate SBATCH Script & Submit
# =============================================================================
# (Simplified from original launch script)

mkdir -p "$LOG_DIR"
SBATCH_SCRIPT=$(mktemp /tmp/sbatch_experiment.XXXXXX.sh)

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

# Propagate variables from the launch script
export PROJECT_ROOT="$PROJECT_ROOT"
export SCRIPTS_DIR="$SCRIPTS_DIR"
export SCRIPT_NAME="$SCRIPT_NAME"
export MODEL_TO_EVAL="$MODEL_TO_EVAL"

$JOB_COMMANDS
EOT

echo "🟢 Submitting job with model: $MODEL_TO_EVAL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

JOB_OUTPUT=$(sbatch "$SBATCH_SCRIPT" 2>&1)

if [ $? -eq 0 ]; then
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
    echo "✅ Job ID: $JOB_ID"
    echo "📁 Output: $LOG_DIR/slurm-$JOB_ID.out"
else
    echo "❌ Failed: $JOB_OUTPUT"
    exit 1
fi
