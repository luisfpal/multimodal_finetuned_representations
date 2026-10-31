#!/bin/bash
# =============================================================================
# Unified launcher for captioned + visual lmms-eval evaluation.
#
# This script handles SLURM submission, environment setup, and task iteration
# in a single file. It replaces the two-file pattern (launcher + worker).
#
# Usage:
#   1. Configure MODEL, TASKS, and SLURM settings below.
#   2. Run:  bash shell/eval/launch_lmms_captioned.sh
#
# Available tasks:
#   Captioned (images → text captions):
#     mme_captioned, mmbench_en_dev_captioned,
#     mmbench_en_dev_open_ended, mmbench_en_dev_captioned_open_ended,
#     seedbench_open_ended, seedbench_captioned_open_ended,
#     vqav2_val_captioned_synthetic, vqav2_val_captioned_human,
#     gqa_val_balanced_captioned
#   Visual (new GQA val-balanced with images):
#     gqa_val_balanced
#   Original built-in tasks (also supported):
#     mme, mmbench_en_dev, vqav2_val, gqa, pope, seedbench, etc.
# =============================================================================

# =============================================================================
# SLURM SETTINGS
# =============================================================================
ACCOUNT="dssc"
JOB_NAME="experiment"
PARTITION="DGX"
GPU_TYPE="A100"
NUM_GPUS=1
CPUS=16
MEM="64GB"
TIME="08:00:00"
NODELIST=""
EXCLUDE=""

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
LOG_DIR="$PROJECT_ROOT/logs"
PYTHON_SCRIPT="scripts/eval/lmms_evaluation_captioned.py"

# =============================================================================
# MODEL SELECTION
# =============================================================================
# Uncomment the model you want to evaluate:

# --- LLaVA ---
MODEL_TO_EVAL="lbasile/llava_7b"

# --- InternVL2 ---
# MODEL_TO_EVAL="OpenGVLab/InternVL2-8B"
# MODEL_TO_EVAL="OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"

# Automatically extract model short name for output directory
OUTPUT_MODEL_DIR="${MODEL_TO_EVAL##*/}"

# =============================================================================
# EVALUATION PARAMETERS
# =============================================================================
BATCH_SIZE=1
NUM_FEWSHOT=""
LIMIT=""
MAX_NEW_TOKENS=""

# =============================================================================
# TASK SELECTION
# =============================================================================
# Uncomment/modify the tasks you want to run:

TASKS=(
    # Captioned variants
    # "mme_captioned"
    # "mmbench_en_dev_captioned"
    "mmbench_en_dev_open_ended"
    "mmbench_en_dev_captioned_open_ended"
    "seedbench_open_ended"
    "seedbench_captioned_open_ended"
    # "vqav2_val_captioned_synthetic"
    # "vqav2_val_captioned_human"
    # "gqa_val_balanced_captioned"

    # GQA val-balanced with images (visual baseline for new dataset)
    # "gqa_val_balanced"

    # Original tasks (also work with this script)
    # "mme"
    # "mmbench_en_dev"
    # "vqav2_val"
    # "gqa"
)

# =============================================================================
# CACHE DIRECTORIES
# =============================================================================
# Hardcoded path: edit this path for your setup.
HF_HOME="$HOME/scratch/huggingface"
MODEL_CACHE_DIR="$HF_HOME/hub"
DATASETS_CACHE_DIR="$HF_HOME/datasets"

# =============================================================================
# BUILD JOB COMMANDS
# =============================================================================
read -r -d '' JOB_COMMANDS << 'JOB_EOF' || true

# ─── CUDA Setup ──────────────────────────────────────────────────────────────
setup_cuda() {
    if command -v module &> /dev/null; then
        module load cuda/12.1 2>/dev/null || true
    fi
    if ! command -v nvcc &> /dev/null || [ ! -x "$(command -v nvcc)" ]; then
        local candidates=("/usr/local/cuda-12.1" "/usr/local/cuda-12.2" "/usr/local/cuda-12" "/usr/local/cuda")
        for path in "${candidates[@]}"; do
            if [ -x "$path/bin/nvcc" ]; then
                echo "🔧 Configuring CUDA from: $path"
                export CUDA_HOME="$path"
                export PATH="$path/bin:$PATH"
                export LD_LIBRARY_PATH="$path/lib64:$LD_LIBRARY_PATH"
                break
            fi
        done
    fi
    if command -v nvcc &> /dev/null; then
        local nvcc_path=$(command -v nvcc)
        local cuda_root=$(dirname $(dirname "$nvcc_path"))
        echo "✓ CUDA: $nvcc_path"
        export CUDA_HOME="$cuda_root"
        export PATH="$cuda_root/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_root/lib64:$LD_LIBRARY_PATH"
    else
        echo "⚠️ WARNING: nvcc not found!"
    fi
}
setup_cuda

# ─── Environment ─────────────────────────────────────────────────────────────
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export HF_HOME
export HF_HUB_CACHE="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="$DATASETS_CACHE_DIR"
export LMMS_EVAL_USE_CACHE="True"
export LMMS_EVAL_HOME="$HOME/scratch/lmms_eval_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export NUM_GPUS

source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"

# ─── JWT Authentication (Orfeo) ─────────────────────────────────────────────
if [ -n "${ORFEO_AUTH_URL}" ] && [ -n "${ORFEO_CLIENT_ID}" ]; then
    echo "🔐 Generating JWT token for Orfeo LLM..."
    JWT=$(curl -s -X POST "${ORFEO_AUTH_URL}" \
      -H "Content-Type: application/x-www-form-urlencoded" \
      -d "grant_type=password" \
      -d "client_id=${ORFEO_CLIENT_ID}" \
      -d "username=${ORFEO_USERNAME}" \
      -d "password=${ORFEO_CLIENT_SECRET}" \
      -d "scope=profile" | jq -r '.access_token')
    if [ -z "$JWT" ] || [ "$JWT" = "null" ]; then
        echo "❌ Failed to get JWT token!"
        exit 1
    fi
    export OPENAI_API_KEY="$JWT"
    export OPENAI_API_URL="${ORFEO_LLM_URL}"
    export MODEL_VERSION="${ORFEO_MODEL}"
fi

# ─── Evaluation Loop ────────────────────────────────────────────────────────

run_task() {
    local TASK="$1"
    local OUTPUT_DIR="$PROJECT_ROOT/evals/$OUTPUT_MODEL_DIR/lmms_eval/$TASK"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ TASK: $TASK  |  MODEL: $OUTPUT_MODEL_DIR"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Build arguments
    local args=(
        --model-name-or-path "$MODEL_TO_EVAL"
        --model-cache-dir "$MODEL_CACHE_DIR"
        --output-dir "$OUTPUT_DIR"
        --batch-size "$BATCH_SIZE"
        --task-name "$TASK"
    )
    [ -n "$NUM_FEWSHOT" ] && args+=(--num-fewshot "$NUM_FEWSHOT")
    [ -n "$LIMIT" ] && args+=(--limit "$LIMIT")
    [ -n "$MAX_NEW_TOKENS" ] && args+=(--max-new-tokens "$MAX_NEW_TOKENS")

    # Execute
    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "[Accelerate: $NUM_GPUS GPUs]"
        accelerate launch --num_processes="$NUM_GPUS" \
            "$PYTHON_SCRIPT" "${args[@]}"
    else
        echo "[Single GPU: device_map=auto]"
        python "$PYTHON_SCRIPT" "${args[@]}"
    fi

    echo "✓ COMPLETED: $TASK"
}

for TASK in "${TASKS[@]}"; do
    run_task "$TASK"
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✓ ALL TASKS COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════"
JOB_EOF

# =============================================================================
# GENERATE SBATCH SCRIPT & SUBMIT
# =============================================================================
mkdir -p "$LOG_DIR"
SBATCH_SCRIPT=$(mktemp /tmp/sbatch_captioned.XXXXXX.sh)

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

# Build the tasks array declaration for inside the job
TASKS_DECLARATION="TASKS=("
for t in "${TASKS[@]}"; do
    TASKS_DECLARATION+="\"$t\" "
done
TASKS_DECLARATION+=")"

cat <<EOT >"$SBATCH_SCRIPT"
#!/bin/bash
$SBATCH_DIRECTIVES

# Propagate variables from the launch script
export PROJECT_ROOT="$PROJECT_ROOT"
export MODEL_TO_EVAL="$MODEL_TO_EVAL"
export OUTPUT_MODEL_DIR="$OUTPUT_MODEL_DIR"
export PYTHON_SCRIPT="$PYTHON_SCRIPT"
export HF_HOME="$HF_HOME"
export MODEL_CACHE_DIR="$MODEL_CACHE_DIR"
export DATASETS_CACHE_DIR="$DATASETS_CACHE_DIR"
export NUM_GPUS=$NUM_GPUS
export BATCH_SIZE=$BATCH_SIZE
export NUM_FEWSHOT="$NUM_FEWSHOT"
export LIMIT="$LIMIT"
export MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
$TASKS_DECLARATION

$JOB_COMMANDS
EOT

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              CAPTIONED EVALUATION LAUNCHER                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║ Model: %-52s ║\n" "$MODEL_TO_EVAL"
printf "║ Tasks: %-52s ║\n" "${#TASKS[@]} task(s)"
printf "║ GPUs: %-53s ║\n" "$NUM_GPUS x $GPU_TYPE"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

for t in "${TASKS[@]}"; do
    echo "  → $t"
done
echo ""

JOB_OUTPUT=$(sbatch "$SBATCH_SCRIPT" 2>&1)

if [ $? -eq 0 ]; then
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
    echo "🟢 Submitted! Job ID: $JOB_ID"
    echo "📁 Log: $LOG_DIR/slurm-$JOB_ID.out"
else
    echo "❌ Submission failed: $JOB_OUTPUT"
    exit 1
fi
