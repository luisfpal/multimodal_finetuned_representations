#!/bin/bash
# This script runs encoder-decoder CKA similarity analysis
# Computes Linear CKA between an encoder (Jina CLIP or OpenAI CLIP) and VLM decoder residual stream

# --- Configuration Section ---

# Datasets (only OpenVQA datasets with deterministic splits)
DATASETS=(
    # "l11p/COCOQA-restval-ICQA"
    "l11p/VQAv2-COCO-restval-ICQ"
)

# Encoder model: "jina" (Jina CLIP v2) or "clip" (openai/clip-vit-large-patch14-336)
# Note: CLIP only supports caption_source="human"
ENCODER_MODEL="jina"

# Similarity measure: "cka" (Linear CKA) or "ols" (OLS linear regression R², returns train+test)
SIMILARITY_MEASURE="cka"

# Caption source: "human" or "synthetic" (clip only supports "human")
CAPTION_SOURCE="human"

# Token pooling methods for decoder
POOLING_METHODS=(
    "last"
    "mean_context"
)

# Processing parameters
BATCH_SIZE="1"
SEED="42"
MAX_SAMPLES="3000"
DATASET_SPLIT="split01"
NORMALIZE="false"

# Guide text for decoder VQA prompts
GUIDE_TEXT="Answer the question using a single word or phrase."

# Models to evaluate
MODELS=(
    # "lbasile/llava_7b"
    "l11p/llava-v1.5-7b-pretrained-lbasile"
    # "lbasile/llava_13b"
    # "l11p/llava-v1.5-13b-pretrained-lbasile"
    # "OpenGVLab/InternVL2-8B"
    # "OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"
    # "nyu-visionx/cambrian-8b"
    # "lbasile/cambrian-8b-pretrained"
)

# Directory configuration
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"

# Script configuration
FILENAME="similarity_vlms_encoder_decoder.py"
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/$FILENAME"

# --- Environment Setup ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Run Analysis ---
START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')

echo "====== Encoder-Decoder CKA Analysis ======"
echo "Results directory: $RESULTS_DIR"
echo "Started at: $START_HUMAN"

for POOLING_METHOD in "${POOLING_METHODS[@]}"; do
    echo -e "\n════════════════════════════════════════════════════════════════"
    echo "Pooling Method: $POOLING_METHOD"
    echo "════════════════════════════════════════════════════════════════"

    for MODEL_NAME in "${MODELS[@]}"; do
        echo -e "\n────────────────────────────────────────────────────────────────"
        echo "Processing Model: $MODEL_NAME"
        echo "────────────────────────────────────────────────────────────────"

        source "$PROJECT_DIR/.venv/bin/activate"

        for DATASET_PATH_OR_NAME in "${DATASETS[@]}"; do
            echo -e "\n  Dataset: $DATASET_PATH_OR_NAME"

            CMD_ARGS=(
                --encoder-model "$ENCODER_MODEL"
                --similarity-measure "$SIMILARITY_MEASURE"
                --dataset-path-or-name "$DATASET_PATH_OR_NAME"
                --decoder-model "$MODEL_NAME"
                --pooling-method "$POOLING_METHOD"
                --dataset-split "$DATASET_SPLIT"
                --max-samples "$MAX_SAMPLES"
                --batch-size "$BATCH_SIZE"
                --seed "$SEED"
                --guide-text "$GUIDE_TEXT"
                --caption-source "$CAPTION_SOURCE"
                --results-dir "$RESULTS_DIR"
            )

            if [ "$NORMALIZE" = "true" ]; then
                CMD_ARGS+=(--normalize)
            fi

            python3 "$FILEPATH" "${CMD_ARGS[@]}"

            echo "    Done: $MODEL_NAME | $POOLING_METHOD | $DATASET_PATH_OR_NAME"
        done
    done
done

END_EPOCH=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
ELAPSED_SECONDS=$((END_EPOCH - START_EPOCH))
ELAPSED_HMS=$(printf '%02d:%02d:%02d' \
    $((ELAPSED_SECONDS / 3600)) \
    $(((ELAPSED_SECONDS % 3600) / 60)) \
    $((ELAPSED_SECONDS % 60)))

echo ""
echo "====== All Analyses Complete ======"
echo "Finished at: $END_HUMAN"
echo "Elapsed: $ELAPSED_HMS"
