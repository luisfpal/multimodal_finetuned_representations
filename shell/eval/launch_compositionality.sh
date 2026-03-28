#!/bin/bash
# =============================================================================
# SLURM launcher for compositionality evaluation (ARO + BiVLC).
#
# Evaluates generative VLMs on compositionality benchmarks using
# log-likelihood scoring and MCQ prompting.
#
# Usage:
#   1. Configure MODELS, BENCHMARK, ARO_SUBTASKS, and SLURM settings below.
#   2. Run:  bash shell/eval/launch_compositionality.sh
# =============================================================================

# =============================================================================
# SLURM SETTINGS
# =============================================================================
ACCOUNT="lade"
JOB_NAME="experiment"
PARTITION="H100"
GPU_TYPE="H100"
NUM_GPUS=1
CPUS=12
MEM="48GB"
TIME="10:00:00"
NODELIST="dgx003"
EXCLUDE=""

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
LOG_DIR="$PROJECT_ROOT/logs"
PYTHON_SCRIPT="scripts/eval/compositionality_evaluation.py"
PYTHON_SCRIPT_PATH="$PROJECT_ROOT/$PYTHON_SCRIPT"

# =============================================================================
# MODEL SELECTION (loop over all uncommented models)
# =============================================================================
MODELS=(
    # "lbasile/llava_7b"
    # "l11p/llava-v1.5-7b-pretrained-lbasile"
    # "OpenGVLab/InternVL2-8B"
    # "OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Base"
    "nyu-visionx/cambrian-8b"
    "lbasile/cambrian-8b-pretrained"
)

# =============================================================================
# EVALUATION PARAMETERS
# =============================================================================

# Benchmark to evaluate: "aro", "bivlc", or "all"
BENCHMARK="aro"

# Evaluation method: "loglikelihood", "mcq", or "both"
EVAL_METHOD="mcq"

# Sample limit (empty = no limit, use for debugging)
LIMIT=""

# ARO subtasks to evaluate (comment out subtasks you want to skip)
# Order subtasks require images — run: bash shell/data/download_aro_images.sh
ARO_SUBTASKS=(
    # "vg_relation"
    # "vg_attribution"
    # "coco_order"
    "flickr30k_order"
)

# ARO data root (where ARO annotations and images are downloaded)
ARO_DATA_ROOT="$HOME/scratch/datasets/vision-language-models-are-bows"

# =============================================================================
# CACHE DIRECTORIES
# =============================================================================
HF_HOME="$HOME/scratch/huggingface"
MODEL_CACHE_DIR="$HF_HOME/hub"
DATASETS_CACHE_DIR="$HF_HOME/datasets"

# =============================================================================
# SERIALIZE ARRAYS FOR SBATCH HEREDOC
# =============================================================================
# bash arrays cannot be exported into heredocs directly; inline them as strings
_MODELS_STR=""
for m in "${MODELS[@]}"; do _MODELS_STR+="\"$m\" "; done

_ARO_SUBTASKS_STR=""
for s in "${ARO_SUBTASKS[@]}"; do _ARO_SUBTASKS_STR+="\"$s\" "; done

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

# ─── Reconstruct arrays from serialized strings ─────────────────────────────
eval "MODELS=($_MODELS_STR)"
eval "ARO_SUBTASKS=($_ARO_SUBTASKS_STR)"

# ─── Run Evaluation ──────────────────────────────────────────────────────────
for MODEL_TO_EVAL in "${MODELS[@]}"; do
    OUTPUT_MODEL_DIR="${MODEL_TO_EVAL##*/}"
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
    [ ${#ARO_SUBTASKS[@]} -gt 0 ] && args+=(--aro-subtasks "${ARO_SUBTASKS[@]}")
    [ -n "$LIMIT" ] && args+=(--limit "$LIMIT")

    python "$SNAPSHOT_PYTHON_SCRIPT" "${args[@]}"

    echo ""
    echo "================================================================"
    echo "  COMPLETE: $MODEL_TO_EVAL"
    echo "================================================================"
done
JOB_EOF

# =============================================================================
# GENERATE SBATCH SCRIPT & SUBMIT
# =============================================================================
mkdir -p "$LOG_DIR"

if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Python script not found: $PYTHON_SCRIPT_PATH"
    exit 1
fi

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
export PYTHON_SCRIPT="$PYTHON_SCRIPT"
export HF_HOME="$HF_HOME"
export MODEL_CACHE_DIR="$MODEL_CACHE_DIR"
export DATASETS_CACHE_DIR="$DATASETS_CACHE_DIR"
export BENCHMARK="$BENCHMARK"
export EVAL_METHOD="$EVAL_METHOD"
export LIMIT="$LIMIT"
export ARO_DATA_ROOT="$ARO_DATA_ROOT"
export _MODELS_STR="$_MODELS_STR"
export _ARO_SUBTASKS_STR="$_ARO_SUBTASKS_STR"
export SNAPSHOT_PYTHON_SCRIPT="/tmp/compositionality_evaluation_\${SLURM_JOB_ID}.py"

cat <<'PYTHON_SNAPSHOT_EOF' > "\$SNAPSHOT_PYTHON_SCRIPT"
EOT

cat "$PYTHON_SCRIPT_PATH" >> "$SBATCH_SCRIPT"

cat <<'EOT' >>"$SBATCH_SCRIPT"
PYTHON_SNAPSHOT_EOF

chmod 600 "$SNAPSHOT_PYTHON_SCRIPT"
trap 'rm -f "$SNAPSHOT_PYTHON_SCRIPT"' EXIT

EOT

printf "%s\n" "$JOB_COMMANDS" >>"$SBATCH_SCRIPT"

echo ""
echo "================================================================"
echo "  COMPOSITIONALITY EVALUATION LAUNCHER"
printf "  Models:    %s\n" "${MODELS[*]}"
printf "  Benchmark: %s\n" "$BENCHMARK"
printf "  Method:    %s\n" "$EVAL_METHOD"
printf "  Snapshot:  %s\n" "$PYTHON_SCRIPT"
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
