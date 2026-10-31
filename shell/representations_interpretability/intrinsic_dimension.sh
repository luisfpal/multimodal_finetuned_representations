#!/bin/bash
# Compute intrinsic dimension from pre-cached representations (last pooling, unique_images).
# Representations must be extracted first via multimodality_similarity.py.

set -e

# ── Dataset run matrix ───────────────────────────────────────────────────────
# Format: "dataset_path_or_name|caption_source|dataset_split|downsample_size|repeated_images|open_ended"
# Fields 4 (downsample_size) and 5 (repeated_images) are unused for ID — kept for format
# compatibility with multimodality_similarity.sh.
DATASET_RUNS=(
    # "l11p/MME-test-captioned|synthetic|||false|false"
    # "l11p/POPE-test-captioned|synthetic|||false|false"
    # "l11p/MMBench_EN-dev-captioned|synthetic|||false|true"
    # "l11p/SEED-Bench-images-captioned|synthetic|||false|true"
    # "l11p/GQA-val-balanced-noyesno-captioned|synthetic|||false|true"
    "l11p/COCOQA-restval-ICQA|human|split01||false|true"
    "l11p/VQAv2-COCO-restval-ICQ|human|split01||false|true"
)

# ── Dataset capability map ───────────────────────────────────────────────────
# Format: "kind|caption_policy|split_support"
#   kind: openvqa | mc_openended | mc_yesno
#   caption_policy: human_or_synthetic | synthetic_only
#   split_support: true | false
declare -A DATASET_PROFILE=(
    ["l11p/MMBench_EN-dev-captioned"]="mc_openended|synthetic_only|false"
    ["l11p/SEED-Bench-images-captioned"]="mc_openended|synthetic_only|false"
    ["l11p/MME-test-captioned"]="mc_yesno|synthetic_only|false"
    ["l11p/POPE-test-captioned"]="mc_yesno|synthetic_only|false"
    ["l11p/GQA-val-balanced-noyesno-captioned"]="openvqa|synthetic_only|false"
    ["l11p/COCOQA-restval-ICQA"]="openvqa|human_or_synthetic|true"
    ["l11p/VQAv2-COCO-restval-ICQ"]="openvqa|human_or_synthetic|true"
)

# ── ID algorithm ─────────────────────────────────────────────────────────────
ID_ALGORITHM="scaling_gride"  # scaling_gride | MLE | twoNN
ID_NN_RANK="8"
ID_NN_RANGE_MAX="100"
ID_FRACTION="0.9"

# ── Models ───────────────────────────────────────────────────────────────────
MODELS=(
    "lbasile/llava_7b"
    "l11p/llava-v1.5-7b-pretrained-lbasile"
    "nyu-visionx/cambrian-8b"
    "lbasile/cambrian-8b-pretrained"
    "lbasile/llava_13b"
    "l11p/llava-v1.5-13b-pretrained-lbasile"
    "OpenGVLab/InternVL2-8B"
    "OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"
    "lmms-lab/LLaVA-OneVision-1.5-4B-Base"
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Base"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
)

# ── Environment (override via env vars or set by launch_job.sh) ──────────────
# Hardcoded path: edit this path for your setup.
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$HOME/scratch}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-false}"
DRY_RUN="${DRY_RUN:-false}"

PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/intrinsic_dimension.py"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$PROJECT_DIR/.venv/bin/activate"

START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')

echo "====== Intrinsic Dimension Analysis ======"
echo "Representations dir: $REPRESENTATIONS_DIR"
echo "Results dir:         $RESULTS_DIR"
echo "Algorithm:           $ID_ALGORITHM (k=$ID_NN_RANK, range_max=$ID_NN_RANGE_MAX, fraction=$ID_FRACTION)"
echo "Started at:          $START_HUMAN"

for MODEL_NAME in "${MODELS[@]}"; do
    echo -e "\n════════════════════════════════════════════════════════════════"
    echo "🤖 Model: $MODEL_NAME"
    echo "════════════════════════════════════════════════════════════════"

    for DATASET_RUN in "${DATASET_RUNS[@]}"; do
        IFS='|' read -r DATASET_PATH_OR_NAME RUN_CAPTION_SOURCE RUN_DATASET_SPLIT _ _ RUN_OPEN_ENDED <<< "$DATASET_RUN"

        echo -e "\n  🌍 Dataset: $DATASET_PATH_OR_NAME"

        PROFILE="${DATASET_PROFILE[$DATASET_PATH_OR_NAME]:-}"
        if [ -z "$PROFILE" ]; then
            echo "❌ Unknown dataset: $DATASET_PATH_OR_NAME"
            echo "   Add it to DATASET_PROFILE as: kind|caption_policy|split_support."
            exit 1
        fi
        IFS='|' read -r DATASET_KIND CAPTION_POLICY SPLIT_SUPPORT <<< "$PROFILE"

        EFFECTIVE_OPEN_ENDED=false
        if [ "$DATASET_KIND" = "mc_openended" ] && [ "$RUN_OPEN_ENDED" = "true" ]; then
            EFFECTIVE_OPEN_ENDED=true
        fi

        EFFECTIVE_DATASET_SPLIT=""
        if [ "$SPLIT_SUPPORT" = "true" ] && [ -n "$RUN_DATASET_SPLIT" ]; then
            EFFECTIVE_DATASET_SPLIT="$RUN_DATASET_SPLIT"
        fi

        CMD_ARGS=(
            --results-dir "$RESULTS_DIR"
            --representations-dir "$REPRESENTATIONS_DIR"
            --dataset-path-or-name "$DATASET_PATH_OR_NAME"
            --model-name-or-path "$MODEL_NAME"
            --caption-source "$RUN_CAPTION_SOURCE"
            --id-algorithm "$ID_ALGORITHM"
            --id-nn-rank "$ID_NN_RANK"
            --id-nn-range-max "$ID_NN_RANGE_MAX"
            --id-fraction "$ID_FRACTION"
        )

        [ "$EFFECTIVE_OPEN_ENDED" = "true" ]    && CMD_ARGS+=(--open-ended)
        [ -n "$EFFECTIVE_DATASET_SPLIT" ]        && CMD_ARGS+=(--dataset-split "$EFFECTIVE_DATASET_SPLIT")
        [ "$FORCE_RECOMPUTE" = "true" ]          && CMD_ARGS+=(--force)
        [ "$DRY_RUN" = "true" ]                  && CMD_ARGS+=(--dry-run)

        python3 "$FILEPATH" "${CMD_ARGS[@]}"

        echo "    ✅ Done: $MODEL_NAME | $DATASET_PATH_OR_NAME"
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
echo "====== Intrinsic Dimension Analysis Complete ======"
echo "Finished at: $END_HUMAN"
echo "Elapsed:     $ELAPSED_HMS"
