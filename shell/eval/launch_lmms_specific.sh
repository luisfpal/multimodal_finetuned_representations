#!/bin/bash
# =============================================================================
# Submit a SLURM job to evaluate ONE model against ALL tasks.
# Uses lmms_evaluation.sh as the worker (loops over tasks inside the job).
#
# Usage: bash shell/eval/launch_lmms_specific.sh
# =============================================================================

# SLURM settings
ACCOUNT="lade"
JOB_NAME="experiment"
PARTITION="H100"
GPU_TYPE="H100"
NUM_GPUS=1
CPUS=16
MEM="64GB"
TIME="00:30:00"
NODELIST=""
EXCLUDE=""

# Project paths
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
SCRIPTS_DIR="$PROJECT_ROOT/shell/eval"
LOG_DIR="$PROJECT_ROOT/logs"
SCRIPT_NAME="lmms_evaluation.sh"

# =============================================================================
# MODEL SELECTION (uncomment one)
# =============================================================================

# MODEL_TO_EVAL="lbasile/llava_7b"
# MODEL_TO_EVAL="lbasile/llava_13b"
MODEL_TO_EVAL="OpenGVLab/InternVL2-8B"
# MODEL_TO_EVAL="OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"
# MODEL_TO_EVAL="nyu-visionx/cambrian-8b"
# MODEL_TO_EVAL="lbasile/cambrian-8b-pretrained"
# MODEL_TO_EVAL="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
# MODEL_TO_EVAL="lmms-lab/LLaVA-OneVision-1.5-8B-Base"
# MODEL_TO_EVAL="lbasile/LLaVA-OneVision-1.5-8B-stage1"
# MODEL_TO_EVAL="lbasile/LLaVA-OneVision-1.5-8B-stage2-2600"

OUTPUT_MODEL_DIR="${MODEL_TO_EVAL##*/}"
NO_IMAGE="false"
VENV="$PROJECT_ROOT/.venv"

# =============================================================================
# JOB SCRIPT (heredoc — runs inside SLURM)
# =============================================================================

read -r -d '' JOB_COMMANDS << 'JOB_EOF' || true

TASKS=(
    # "scienceqa_img"
    # "mme"
    # "mmbench_en_dev"
    # "textvqa_val"
    # "llava_in_the_wild"
    # "gqa"
    # "vizwiz_vqa_val"
    # "vqav2_val"
    # "mmbench_cn_dev"
    # "pope"
    # "seedbench"
    "mmmu_val"
)

# CUDA setup
setup_cuda() {
    if command -v module &> /dev/null; then
        module load cuda/12.1 2>/dev/null || true
    fi
    if ! command -v nvcc &> /dev/null; then
        for path in /usr/local/cuda-12.1 /usr/local/cuda-12.2 /usr/local/cuda-12 /usr/local/cuda; do
            if [ -x "$path/bin/nvcc" ]; then
                export CUDA_HOME="$path"
                export PATH="$path/bin:$PATH"
                export LD_LIBRARY_PATH="$path/lib64:$LD_LIBRARY_PATH"
                break
            fi
        done
    fi
    if command -v nvcc &> /dev/null; then
        local cuda_root
        cuda_root=$(dirname "$(dirname "$(command -v nvcc)")")
        export CUDA_HOME="$cuda_root"
        export PATH="$cuda_root/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_root/lib64:$LD_LIBRARY_PATH"
    fi
}
setup_cuda

# Load env
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

source "$VENV/bin/activate"
cd "$SCRIPTS_DIR"
chmod +x "$SCRIPT_NAME"

export TOKENIZERS_PARALLELISM=false
export NUM_GPUS
export NO_IMAGE
export OUTPUT_MODEL_DIR

# Orfeo JWT (optional)
if [ -n "${ORFEO_AUTH_URL:-}" ] && [ -n "${ORFEO_CLIENT_ID:-}" ]; then
    JWT=$(curl -s -X POST "${ORFEO_AUTH_URL}" \
      -H "Content-Type: application/x-www-form-urlencoded" \
      -d "grant_type=password" \
      -d "client_id=${ORFEO_CLIENT_ID}" \
      -d "username=${ORFEO_USERNAME}" \
      -d "password=${ORFEO_CLIENT_SECRET}" \
      -d "scope=profile" | jq -r '.access_token')

    if [ -n "$JWT" ] && [ "$JWT" != "null" ]; then
        export OPENAI_API_KEY="$JWT"
        export OPENAI_API_URL="${ORFEO_LLM_URL}"
        export MODEL_VERSION="${ORFEO_MODEL}"
    fi
fi

# Run each task
export SINGLE_MODEL="$MODEL_TO_EVAL"

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ $TASK"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    export TASK_NAME="$TASK"
    ./"$SCRIPT_NAME"
    echo "✓ $TASK"
done
JOB_EOF

# =============================================================================
# SUBMIT
# =============================================================================

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

export PROJECT_ROOT="$PROJECT_ROOT"
export SCRIPTS_DIR="$SCRIPTS_DIR"
export SCRIPT_NAME="$SCRIPT_NAME"
export MODEL_TO_EVAL="$MODEL_TO_EVAL"
export NO_IMAGE="$NO_IMAGE"
export OUTPUT_MODEL_DIR="$OUTPUT_MODEL_DIR"
export NUM_GPUS="$NUM_GPUS"
export VENV="$VENV"

$JOB_COMMANDS
EOT

echo "Submitting: $MODEL_TO_EVAL"
JOB_OUTPUT=$(sbatch "$SBATCH_SCRIPT" 2>&1)

if [ $? -eq 0 ]; then
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
    echo "Job ID: $JOB_ID"
    echo "Log: $LOG_DIR/slurm-$JOB_ID.out"
else
    echo "Failed: $JOB_OUTPUT"
    exit 1
fi
