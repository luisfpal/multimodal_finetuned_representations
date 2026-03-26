#!/bin/bash
# =============================================================================
# SLURM launcher for compositionality evaluation (ARO + BiVLC).
#
# Evaluates generative VLMs on compositionality benchmarks using
# log-likelihood scoring and MCQ prompting.
#
# Usage:
#   1. Configure MODEL, BENCHMARK, and SLURM settings below.
#   2. Run:  bash shell/eval/launch_compositionality.sh
# =============================================================================

# =============================================================================
# SLURM SETTINGS
# =============================================================================
ACCOUNT="dssc"
JOB_NAME="compositionality"
PARTITION="DGX"
GPU_TYPE="A100"
NUM_GPUS=1
CPUS=16
MEM="64GB"
TIME="12:00:00"
NODELIST=""
EXCLUDE=""

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
LOG_DIR="$PROJECT_ROOT/logs"
PYTHON_SCRIPT="scripts/eval/compositionality_evaluation.py"

# =============================================================================
# MODEL SELECTION
# =============================================================================
# Uncomment the model you want to evaluate:

# --- LLaVA ---
MODEL_TO_EVAL="lbasile/llava_7b"

# --- InternVL2 ---
# MODEL_TO_EVAL="OpenGVLab/InternVL2-8B"
# MODEL_TO_EVAL="OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"

# --- LLaVA OneVision ---
# MODEL_TO_EVAL="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"

# --- InternVL3.5 ---
# MODEL_TO_EVAL="OpenGVLab/InternVL3_5-8B"

# Automatically extract model short name for output directory
OUTPUT_MODEL_DIR="${MODEL_TO_EVAL##*/}"

# =============================================================================
# EVALUATION PARAMETERS
# =============================================================================

# Benchmark to evaluate: "aro", "bivlc", or "all"
BENCHMARK="all"

# Evaluation method: "loglikelihood", "mcq", or "both"
EVAL_METHOD="both"

# Sample limit (empty = no limit, use for debugging)
LIMIT=""

# ARO data root (where ARO annotations and images are downloaded)
ARO_DATA_ROOT="$HOME/scratch/datasets/vision-language-models-are-bows"

# =============================================================================
# CACHE DIRECTORIES
# =============================================================================
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
                echo "Configuring CUDA from: $path"
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
        echo "CUDA: $nvcc_path"
        export CUDA_HOME="$cuda_root"
        export PATH="$cuda_root/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_root/lib64:$LD_LIBRARY_PATH"
    else
        echo "WARNING: nvcc not found!"
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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"

# ─── Run Evaluation ──────────────────────────────────────────────────────────
OUTPUT_DIR="$PROJECT_ROOT/evals/$OUTPUT_MODEL_DIR/compositionality"

echo ""
echo "================================================================"
echo "  COMPOSITIONALITY EVALUATION"
echo "  Model:     $MODEL_TO_EVAL"
echo "  Benchmark: $BENCHMARK"
echo "  Method:    $EVAL_METHOD"
echo "================================================================"

args=(
    --model-name-or-path "$MODEL_TO_EVAL"
    --output-dir "$OUTPUT_DIR"
    --benchmark "$BENCHMARK"
    --eval-method "$EVAL_METHOD"
    --datasets-cache-dir "$DATASETS_CACHE_DIR"
    --aro-data-root "$ARO_DATA_ROOT"
)
[ -n "$LIMIT" ] && args+=(--limit "$LIMIT")

python "$PYTHON_SCRIPT" "${args[@]}"

echo ""
echo "================================================================"
echo "  COMPLETE"
echo "================================================================"
JOB_EOF

# =============================================================================
# GENERATE SBATCH SCRIPT & SUBMIT
# =============================================================================
mkdir -p "$LOG_DIR"
SBATCH_SCRIPT=$(mktemp /tmp/sbatch_compositionality.XXXXXX.sh)

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
export MODEL_TO_EVAL="$MODEL_TO_EVAL"
export OUTPUT_MODEL_DIR="$OUTPUT_MODEL_DIR"
export PYTHON_SCRIPT="$PYTHON_SCRIPT"
export HF_HOME="$HF_HOME"
export MODEL_CACHE_DIR="$MODEL_CACHE_DIR"
export DATASETS_CACHE_DIR="$DATASETS_CACHE_DIR"
export BENCHMARK="$BENCHMARK"
export EVAL_METHOD="$EVAL_METHOD"
export LIMIT="$LIMIT"
export ARO_DATA_ROOT="$ARO_DATA_ROOT"

$JOB_COMMANDS
EOT

echo ""
echo "================================================================"
echo "  COMPOSITIONALITY EVALUATION LAUNCHER"
printf "  Model:     %s\n" "$MODEL_TO_EVAL"
printf "  Benchmark: %s\n" "$BENCHMARK"
printf "  Method:    %s\n" "$EVAL_METHOD"
printf "  GPUs:      %s x %s\n" "$NUM_GPUS" "$GPU_TYPE"
echo "================================================================"
echo ""

JOB_OUTPUT=$(sbatch "$SBATCH_SCRIPT" 2>&1)

if [ $? -eq 0 ]; then
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
    echo "Submitted! Job ID: $JOB_ID"
    echo "Log: $LOG_DIR/slurm-$JOB_ID.out"
else
    echo "Submission failed: $JOB_OUTPUT"
    exit 1
fi
