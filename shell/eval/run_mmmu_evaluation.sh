#!/bin/bash
# MMMU n-shot ICL evaluation across LLaVA model variants.
# Edit MODELS, N_SHOTS, and MAX_SAMPLES below, then run:
#   bash shell/eval/run_mmmu_evaluation.sh
set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS=(
    # "lbasile/llava_7b"
    # "l11p/llava-v1.5-7b-pretrained-lbasile"
    # "lbasile/llava_7b_frozen"

    # "lbasile/llava_13b"
    # "l11p/llava-v1.5-13b-pretrained-lbasile"
    # "lbasile/llava_13b_6_21"
    # "lbasile/llava_13b_6_21_complementary"

    # "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-4B-Base"
)

N_SHOTS=0           # 0 = zero-shot, >0 = n-shot ICL
MAX_SAMPLES=""      # leave empty to evaluate all samples

PROJECT_DIR="$HOME/multimodal_finetuned_representations"
OUTPUT_DIR="$PROJECT_DIR/results/mmmu_evaluation"
PY_SCRIPT="$PROJECT_DIR/scripts/eval/mmmu_evaluation.py"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# =============================================================================
# RUN
# =============================================================================

START_EPOCH=$(date +%s)
echo "====== MMMU Evaluation ======"
echo "N-shots:    $N_SHOTS"
echo "Output dir: $OUTPUT_DIR"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"

for MODEL_NAME in "${MODELS[@]}"; do
    echo -e "\n────────────────────────────────────────────────────────────────"
    echo "Model: $MODEL_NAME"
    echo "────────────────────────────────────────────────────────────────"

    CMD=(
        "$PYTHON_BIN" "$PY_SCRIPT"
        --model-name "$MODEL_NAME"
        --n-shots "$N_SHOTS"
        --output-dir "$OUTPUT_DIR"
    )
    [ -n "$MAX_SAMPLES" ] && CMD+=(--max-samples "$MAX_SAMPLES")

    "${CMD[@]}"
done

END_EPOCH=$(date +%s)
ELAPSED=$(( END_EPOCH - START_EPOCH ))
printf '\n====== Done ======\nFinished at: %s\nElapsed: %02d:%02d:%02d\n' \
    "$(date '+%Y-%m-%d %H:%M:%S %Z')" \
    $(( ELAPSED / 3600 )) $(( (ELAPSED % 3600) / 60 )) $(( ELAPSED % 60 ))
