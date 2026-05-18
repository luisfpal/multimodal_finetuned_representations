#!/bin/bash
# This script runs multimodality similarity analysis (Text vs Image)

# --- Configuration Section ---
# Dataset run matrix (tuple-like entries)
#
# Format per entry:
#   "dataset_path_or_name|caption_source|dataset_split|downsample_size|repeated_images|open_ended"
#
# Rules:
# - caption_source: human | synthetic for VQA; empty for captioning
# - dataset_split/downsample_size: use empty field when not applicable ("" via ||)
# - repeated_images/open_ended: true | false
# - Same dataset can appear multiple times with different configurations.
#
# Examples:
#   "l11p/COCOQA-restval-ICQA|human|split01||false|false"
#   "l11p/COCOQA-restval-ICQA|synthetic||2000|false|false"
#   "liuhaotian/LLaVA-Instruct-150K|||10000|false|false"
DATASET_RUNS=(
    # Captioning task: top-N detail examples by response length.
    # "liuhaotian/LLaVA-Instruct-150K|||10000|false|false"

    # "l11p/MME-test-captioned|synthetic|||false|false"
    # "l11p/POPE-test-captioned|synthetic|||false|false"
    # "l11p/MMBench_EN-dev-captioned|synthetic||2000|false|true"
    # "l11p/SEED-Bench-images-captioned|synthetic||2000|false|true"
    # "l11p/GQA-val-balanced-noyesno-captioned|synthetic||2000|false|true"
    # "l11p/COCOQA-restval-ICQA|human||2000|false|true"
    # "l11p/VQAv2-COCO-restval-ICQ|human||2000|false|true"
    # "l11p/MMBench_EN-dev-captioned|synthetic|||false|true"
    # "l11p/SEED-Bench-images-captioned|synthetic|||false|true"
    # "l11p/GQA-val-balanced-noyesno-captioned|synthetic|||false|true"
    # "l11p/COCOQA-restval-ICQA|human|split01||false|true"
    # "l11p/VQAv2-COCO-restval-ICQ|human|split01||false|true"
)

# Analysis configuration
# Format: "img_pooling_method|txt_pooling_method"
# First field  → pooling method for image representations (res_img)
# Second field → pooling method for text representations (res_txt)
TOKENS_POOLING_METHODS=(
    "last|last"
    # "mean_image|mean_text_context"
    # "mean_image|last_caption"
    # "last_image|last_caption"
    # "end_image|mean_text_context"
    # "end_image|last"
    # "last10_image|last" #!llava and cambrian
    # "last_img_and_next|last" #!llava and cambrian
    # "end_user_role|mean_text_context"
    # "end_user_role|last"
    # "end_user_role|end_user_role"
    # "mean_image|end_user_role"
)

# Image-vs-image pooling specs (--img-img-pair mode).
# Both representations are extracted from the image dataloader.
# First field → pooling for res_img (row), Second → pooling for res_img (col).
IMG_IMG_POOLING_METHODS=(
    # "mean_image|end_user_role"
)
# SIMILARITY_MEASURES="neighborhood_overlap linear_cka information_imbalance information_imbalance_reversed"
SIMILARITY_MEASURES="information_imbalance information_imbalance_reversed"
# SIMILARITY_MEASURES="${SIMILARITY_MEASURES} linear_regression_r2 linear_regression_r2_reversed"

# Cambrian-only vision encoder selection.
# Empty array preserves the original full Cambrian encoder set.
# Valid aliases: siglip clip dinov2 convnext
CAMBRIAN_VISION_ENCODERS=(
    # "siglip"
    # "clip"
    # "dinov2"
    # "convnext"
)

# Processing parameters
BATCH_SIZE="1" # 1 for MMBench
SEED="42"
REGRESSION_TYPE="ols" # Valid: ols | ridge | lasso

# Guide text per dataset (auto-selected in the loop below)
GUIDE_TEXT_OPENENDED="Answer the question using a single word or phrase."
GUIDE_TEXT_MULTICHOICE="Answer with the option's letter from the given choices directly."
CAPTIONING_IMAGE_DIR="${CAPTIONING_IMAGE_DIR:-/orfeo/cephfs/scratch/area/ddoimo/data/mscoco/train2014}"

# Dataset capability map (single source of truth for conditional behavior)
#
# Fields:
#   "kind|caption_policy|split_support"
#   kind: openvqa | mc_openended | mc_yesno | captioning
#   caption_policy: human_or_synthetic | synthetic_only | none
#   split_support: true for OpenVQA precomputed splits
#
# Notes:
# - DATASET_SPLIT and DOWNSAMPLE_SIZE are operationally exclusive for OpenVQA split-capable
#   datasets. If DATASET_SPLIT is active in a run entry, DOWNSAMPLE_SIZE is ignored by design.
# - REPEATED_IMAGES applies only to MC datasets (mc_openended and mc_yesno).
# - OPEN_ENDED applies only to mc_openended datasets.
declare -A DATASET_PROFILE=(
    ["liuhaotian/LLaVA-Instruct-150K"]="captioning|none|false"
    ["l11p/MMBench_EN-dev-captioned"]="mc_openended|synthetic_only|false"
    ["l11p/SEED-Bench-images-captioned"]="mc_openended|synthetic_only|false"
    ["l11p/MME-test-captioned"]="mc_yesno|synthetic_only|false"
    ["l11p/POPE-test-captioned"]="mc_yesno|synthetic_only|false"
    ["l11p/GQA-val-balanced-noyesno-captioned"]="openvqa|synthetic_only|false"
    ["l11p/COCOQA-restval-ICQA"]="openvqa|human_or_synthetic|true"
    ["l11p/VQAv2-COCO-restval-ICQ"]="openvqa|human_or_synthetic|true"
)

# Models to evaluate
MODELS=(
    # "lbasile/llava_7b"
    # "l11p/llava-v1.5-7b-pretrained-lbasile"
    # "nyu-visionx/cambrian-8b"
    # "lbasile/cambrian-8b-pretrained"
    # "lbasile/llava_13b"
    # "l11p/llava-v1.5-13b-pretrained-lbasile"

    # "OpenGVLab/InternVL2-8B"
    # "OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"
    # "lmms-lab/LLaVA-OneVision-1.5-4B-Base"
    # "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    # "lmms-lab/LLaVA-OneVision-1.5-8B-Base"
    # "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"

    # "lbasile/LLaVA-OneVision-1.5-8B-stage1"
    # "lbasile/LLaVA-OneVision-1.5-8B-stage2-2600"
    # "lbasile/llava_7b_6_17"
    # "lbasile/llava_7b_6_17_complementary"
    # "OpenGVLab/InternVL3_5-8B-Pretrained"
    # "OpenGVLab/InternVL3_5-8B-Instruct"
    # "OpenGVLab/InternVL3_5-8B-MPO"
    # "OpenGVLab/InternVL3_5-8B"

)

# Configuration from environment (set by launch_job.sh)
FORCE_METRICS_ONLY="false"
REPRESENTATIONS_ONLY="false"
ENABLE_SEMANTIC_FILTER="false"
ENABLE_CLIP_METRICS="false"   # set to "true" to also compute CLIP similarity metrics
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$HOME/scratch/pooling_tests}"
SAVE_REPRESENTATIONS="true"
FORCE_RECOMPUTE="false"

if [ "$FORCE_RECOMPUTE" = "true" ] && [ "$FORCE_METRICS_ONLY" = "true" ]; then
    echo "❌ Invalid config: FORCE_RECOMPUTE and FORCE_METRICS_ONLY cannot both be true."
    exit 1
fi

# Directory configuration
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results/pooling_tests"

# Script configuration
FILENAME="multimodality_similarity.py"
FILEPATH="$PROJECT_DIR/scripts/representations_interpretability/$FILENAME"

# --- Environment Setup ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV="$PROJECT_DIR/.venv"
source "$VENV/bin/activate"

# --- Run Analysis ---
START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')

echo "====== Multimodality Similarity Analysis ======"
echo "Results directory: $RESULTS_DIR"
echo "Started at: $START_HUMAN"

for POOLING_METHOD in "${TOKENS_POOLING_METHODS[@]}"; do
    echo -e "\n════════════════════════════════════════════════════════════════"
    echo "⚙️  Pooling Method: $POOLING_METHOD"
    echo "════════════════════════════════════════════════════════════════"

    for MODEL_NAME in "${MODELS[@]}"; do
        echo -e "\n────────────────────────────────────────────────────────────────"
        echo "🤖 Processing Model: $MODEL_NAME"
        echo "────────────────────────────────────────────────────────────────"

        # Activate model-specific venv
        VENV="$PROJECT_DIR/.venv"
        source "$VENV/bin/activate"

        for DATASET_RUN in "${DATASET_RUNS[@]}"; do
            IFS='|' read -r DATASET_PATH_OR_NAME RUN_CAPTION_SOURCE RUN_DATASET_SPLIT RUN_DOWNSAMPLE_SIZE RUN_REPEATED_IMAGES RUN_OPEN_ENDED <<< "$DATASET_RUN"

            echo -e "\n  🌍 Dataset: $DATASET_PATH_OR_NAME"

            if [ -z "$DATASET_PATH_OR_NAME" ]; then
                echo "❌ Invalid DATASET_RUN entry (missing dataset): $DATASET_RUN"
                exit 1
            fi
            if [ "$RUN_REPEATED_IMAGES" != "true" ] && [ "$RUN_REPEATED_IMAGES" != "false" ]; then
                echo "❌ Invalid repeated_images in DATASET_RUN: '$RUN_REPEATED_IMAGES'"
                echo "   Expected: true | false"
                exit 1
            fi
            if [ "$RUN_OPEN_ENDED" != "true" ] && [ "$RUN_OPEN_ENDED" != "false" ]; then
                echo "❌ Invalid open_ended in DATASET_RUN: '$RUN_OPEN_ENDED'"
                echo "   Expected: true | false"
                exit 1
            fi

            DATASET_PROFILE_VALUE="${DATASET_PROFILE[$DATASET_PATH_OR_NAME]}"
            IFS='|' read -r DATASET_KIND_VALUE DATASET_CAPTION_POLICY_VALUE DATASET_SPLIT_SUPPORT_VALUE <<< "$DATASET_PROFILE_VALUE"

            if [ -z "$DATASET_PROFILE_VALUE" ] || [ -z "$DATASET_KIND_VALUE" ] || [ -z "$DATASET_CAPTION_POLICY_VALUE" ] || [ -z "$DATASET_SPLIT_SUPPORT_VALUE" ]; then
                echo "❌ Unknown or invalid dataset profile: $DATASET_PATH_OR_NAME"
                echo "   Add it to DATASET_PROFILE as: kind|caption_policy|split_support."
                exit 1
            fi

            IS_OPENVQA_DATASET=false
            IS_MC_DATASET=false
            IS_MC_OPENENDED_CAPABLE=false
            IS_CAPTIONING_DATASET=false
            if [ "$DATASET_KIND_VALUE" = "openvqa" ]; then
                IS_OPENVQA_DATASET=true
            elif [ "$DATASET_KIND_VALUE" = "mc_openended" ]; then
                IS_MC_DATASET=true
                IS_MC_OPENENDED_CAPABLE=true
            elif [ "$DATASET_KIND_VALUE" = "mc_yesno" ]; then
                IS_MC_DATASET=true
            elif [ "$DATASET_KIND_VALUE" = "captioning" ]; then
                IS_CAPTIONING_DATASET=true
            else
                echo "❌ Invalid DATASET_KIND for $DATASET_PATH_OR_NAME: $DATASET_KIND_VALUE"
                exit 1
            fi

            if [ "$IS_CAPTIONING_DATASET" = "true" ]; then
                if [ -n "$RUN_CAPTION_SOURCE" ]; then
                    echo "❌ Invalid caption_source for captioning dataset $DATASET_PATH_OR_NAME: '$RUN_CAPTION_SOURCE'"
                    echo "   Leave the caption_source field empty for captioning runs."
                    exit 1
                fi
                if [ -n "$RUN_DATASET_SPLIT" ]; then
                    echo "❌ Invalid dataset_split for captioning dataset $DATASET_PATH_OR_NAME: '$RUN_DATASET_SPLIT'"
                    echo "   Captioning always uses cached detail_23k.json internally."
                    exit 1
                fi
                EFFECTIVE_CAPTION_SOURCE=""
            else
                if [ "$RUN_CAPTION_SOURCE" != "human" ] && [ "$RUN_CAPTION_SOURCE" != "synthetic" ]; then
                    echo "❌ Invalid caption_source in DATASET_RUN: '$RUN_CAPTION_SOURCE'"
                    echo "   Expected: human | synthetic"
                    exit 1
                fi
                EFFECTIVE_CAPTION_SOURCE="$RUN_CAPTION_SOURCE"
                if [ "$DATASET_CAPTION_POLICY_VALUE" = "synthetic_only" ] && [ "$RUN_CAPTION_SOURCE" != "synthetic" ]; then
                    echo "❌ Invalid caption_source for $DATASET_PATH_OR_NAME: '$RUN_CAPTION_SOURCE'"
                    echo "   This dataset only supports synthetic captions."
                    exit 1
                fi
                if [ "$DATASET_CAPTION_POLICY_VALUE" = "human_or_synthetic" ] && [ "$RUN_CAPTION_SOURCE" != "human" ] && [ "$RUN_CAPTION_SOURCE" != "synthetic" ]; then
                    echo "❌ Invalid caption_source for $DATASET_PATH_OR_NAME: '$RUN_CAPTION_SOURCE'"
                    echo "   This dataset supports human or synthetic captions."
                    exit 1
                fi
            fi

            EFFECTIVE_DATASET_SPLIT="$RUN_DATASET_SPLIT"
            EFFECTIVE_DOWNSAMPLE_SIZE="$RUN_DOWNSAMPLE_SIZE"
            if [ "$IS_CAPTIONING_DATASET" = "true" ]; then
                EFFECTIVE_DATASET_SPLIT=""
            elif [ "$DATASET_SPLIT_SUPPORT_VALUE" = "false" ] && [ -n "$RUN_DATASET_SPLIT" ]; then
                echo "❌ Invalid dataset_split for $DATASET_PATH_OR_NAME: '$RUN_DATASET_SPLIT'"
                echo "   This dataset does not support --dataset-split."
                exit 1
            fi
            if [ "$IS_CAPTIONING_DATASET" != "true" ] && [ "$DATASET_SPLIT_SUPPORT_VALUE" = "true" ] && [ -n "$RUN_DATASET_SPLIT" ] && [ -n "$RUN_DOWNSAMPLE_SIZE" ]; then
                echo "ℹ️  Downsample ignored for $DATASET_PATH_OR_NAME because dataset_split is active."
                EFFECTIVE_DOWNSAMPLE_SIZE=""
            fi

            EFFECTIVE_REPEATED_IMAGES="$RUN_REPEATED_IMAGES"
            if [ "$IS_CAPTIONING_DATASET" = "true" ]; then
                EFFECTIVE_REPEATED_IMAGES="false"
            elif [ "$IS_MC_DATASET" != "true" ] && [ "$RUN_REPEATED_IMAGES" = "true" ]; then
                echo "ℹ️  repeated_images ignored for $DATASET_PATH_OR_NAME (not a multiple-choice dataset)."
                EFFECTIVE_REPEATED_IMAGES="false"
            fi

            EFFECTIVE_OPEN_ENDED="$RUN_OPEN_ENDED"
            if [ "$IS_CAPTIONING_DATASET" = "true" ]; then
                EFFECTIVE_OPEN_ENDED="false"
            elif [ "$IS_MC_OPENENDED_CAPABLE" != "true" ] && [ "$RUN_OPEN_ENDED" = "true" ]; then
                echo "ℹ️  open_ended ignored for $DATASET_PATH_OR_NAME (not open-ended capable MC)."
                EFFECTIVE_OPEN_ENDED="false"
            fi

            # Guide text: derived entirely from the flags above — no string matching needed
            # - OpenVQA (COCOQA, VQAv2, GQA)  → always open-ended
            # - MC open-ended-capable (MMBench, SEED) → open-ended if run field open_ended=true, else multichoice
            # - MC yes/no (MME, POPE)           → always multichoice
            if [ "$IS_CAPTIONING_DATASET" = "true" ]; then
                GUIDE_TEXT=""
            elif [ "$IS_OPENVQA_DATASET" = "true" ]; then
                GUIDE_TEXT="$GUIDE_TEXT_OPENENDED"
            elif [ "$IS_MC_OPENENDED_CAPABLE" = "true" ]; then
                if [ "$EFFECTIVE_OPEN_ENDED" = "true" ]; then
                    GUIDE_TEXT="$GUIDE_TEXT_OPENENDED"
                else
                    GUIDE_TEXT="$GUIDE_TEXT_MULTICHOICE"
                fi
            else
                GUIDE_TEXT="$GUIDE_TEXT_MULTICHOICE"
            fi
            echo "    📝 Guide: $GUIDE_TEXT"

            echo "    🔧 Effective config: kind=$DATASET_KIND_VALUE caption=$EFFECTIVE_CAPTION_SOURCE split=${EFFECTIVE_DATASET_SPLIT:-none} downsample=${EFFECTIVE_DOWNSAMPLE_SIZE:-none} repeated_images=$EFFECTIVE_REPEATED_IMAGES open_ended=$EFFECTIVE_OPEN_ENDED"

            # Build command with conditional flags
            CMD_ARGS=(
                --results-dir "$RESULTS_DIR"
                --representations-dir "$REPRESENTATIONS_DIR"
                --dataset-path-or-name "$DATASET_PATH_OR_NAME"
                --model-name-or-path "$MODEL_NAME"
                --task "$([ "$IS_CAPTIONING_DATASET" = "true" ] && echo captioning || echo vqa)"
                --batch-size "$BATCH_SIZE"
                --seed "$SEED"
                --guide-text "$GUIDE_TEXT"
                --pooling-methods "$POOLING_METHOD"
                --similarity-measures ${SIMILARITY_MEASURES}
                --regression-type "$REGRESSION_TYPE"
            )

            if [ "$IS_CAPTIONING_DATASET" != "true" ]; then
                CMD_ARGS+=(--caption-source "$EFFECTIVE_CAPTION_SOURCE")
            fi

            # Only pass --downsample-size when explicitly set AND not superseded by split.
            if [ -n "$EFFECTIVE_DOWNSAMPLE_SIZE" ]; then
                CMD_ARGS+=(--downsample-size "$EFFECTIVE_DOWNSAMPLE_SIZE")
            fi

            if [ "$SAVE_REPRESENTATIONS" = "true" ]; then
                CMD_ARGS+=(--save-representations)
            fi

            if [ "$ENABLE_SEMANTIC_FILTER" = "true" ]; then
                CMD_ARGS+=(--enable-semantic-filter)
            fi

            if [ "$ENABLE_CLIP_METRICS" = "true" ]; then
                CMD_ARGS+=(--enable-clip-metrics)
            fi

            if [ "$FORCE_RECOMPUTE" = "true" ]; then
                CMD_ARGS+=(--force)
            fi

            if [ "$FORCE_METRICS_ONLY" = "true" ]; then
                CMD_ARGS+=(--force-metrics-only)
            fi

            # Apply --repeated-images only to MC datasets
            if [ "$EFFECTIVE_REPEATED_IMAGES" = "true" ] && [ "$IS_MC_DATASET" = "true" ]; then
                CMD_ARGS+=(--repeated-images)
            fi

            # Apply --open-ended only to MC datasets
            if [ "$EFFECTIVE_OPEN_ENDED" = "true" ] && [ "$IS_MC_OPENENDED_CAPABLE" = "true" ]; then
                CMD_ARGS+=(--open-ended)
            fi

            if [ "$REPRESENTATIONS_ONLY" = "true" ]; then
                CMD_ARGS+=(--representations-only)
            fi

            if [ "$IS_CAPTIONING_DATASET" = "true" ]; then
                CMD_ARGS+=(--captioning-image-dir "$CAPTIONING_IMAGE_DIR")
            elif [ -n "$EFFECTIVE_DATASET_SPLIT" ]; then
                CMD_ARGS+=(--dataset-split "$EFFECTIVE_DATASET_SPLIT")
            fi

            if [ "${#CAMBRIAN_VISION_ENCODERS[@]}" -gt 0 ]; then
                CMD_ARGS+=(--cambrian-vision-encoders "${CAMBRIAN_VISION_ENCODERS[@]}")
            fi

            python3 "$FILEPATH" "${CMD_ARGS[@]}"

            echo "    ✅ Done: $MODEL_NAME | $POOLING_METHOD | $DATASET_PATH_OR_NAME"
        done
    done
done

# ── Image-vs-image analyses ──────────────────────────────────────────────────
if [ "${#IMG_IMG_POOLING_METHODS[@]}" -gt 0 ]; then
    echo -e "\n════════════════════════════════════════════════════════════════"
    echo "🖼️  Image-vs-Image Analyses (--img-img-pair)"
    echo "════════════════════════════════════════════════════════════════"

    for IMG_IMG_SPEC in "${IMG_IMG_POOLING_METHODS[@]}"; do
        echo -e "\n  ⚙️  Pooling Spec: $IMG_IMG_SPEC"
        for MODEL_NAME in "${MODELS[@]}"; do
            echo -e "\n  🤖 Model: $MODEL_NAME"
            for RUN_ENTRY in "${DATASET_RUNS[@]}"; do
                IFS='|' read -r DATASET_PATH_OR_NAME RUN_CAPTION_SOURCE RUN_DATASET_SPLIT RUN_DOWNSAMPLE_SIZE RUN_REPEATED_IMAGES RUN_OPEN_ENDED <<< "$RUN_ENTRY"

                # ── Dataset profile lookup ──────────────────────────────────
                PROFILE="${DATASET_PROFILE[$DATASET_PATH_OR_NAME]:-}"
                if [ -z "$PROFILE" ]; then
                    echo "⚠️  Unknown dataset profile for '$DATASET_PATH_OR_NAME'. Skipping."
                    continue
                fi
                IFS='|' read -r DATASET_KIND_VALUE CAPTION_POLICY SPLIT_SUPPORT <<< "$PROFILE"
                IS_MC_DATASET="false"
                IS_MC_OPENENDED_CAPABLE="false"
                IS_OPENVQA_DATASET="false"
                IS_CAPTIONING_DATASET="false"
                if [ "$DATASET_KIND_VALUE" = "mc_openended" ] || [ "$DATASET_KIND_VALUE" = "mc_yesno" ]; then
                    IS_MC_DATASET="true"
                fi
                if [ "$DATASET_KIND_VALUE" = "mc_openended" ]; then
                    IS_MC_OPENENDED_CAPABLE="true"
                fi
                if [ "$DATASET_KIND_VALUE" = "openvqa" ]; then
                    IS_OPENVQA_DATASET="true"
                fi
                if [ "$DATASET_KIND_VALUE" = "captioning" ]; then
                    IS_CAPTIONING_DATASET="true"
                fi

                if [ "$IS_CAPTIONING_DATASET" = "true" ]; then
                    if [ -n "$RUN_CAPTION_SOURCE" ] || [ -n "$RUN_DATASET_SPLIT" ]; then
                        echo "⚠️  captioning runs require empty caption_source and dataset_split. Skipping '$DATASET_PATH_OR_NAME'."
                        continue
                    fi
                    EFFECTIVE_CAPTION_SOURCE=""
                else
                    if [ "$RUN_CAPTION_SOURCE" != "human" ] && [ "$RUN_CAPTION_SOURCE" != "synthetic" ]; then
                        echo "⚠️  human or synthetic captions required for '$DATASET_PATH_OR_NAME'. Skipping."
                        continue
                    fi
                    EFFECTIVE_CAPTION_SOURCE="$RUN_CAPTION_SOURCE"
                    if [ "$CAPTION_POLICY" = "synthetic_only" ] && [ "$RUN_CAPTION_SOURCE" != "synthetic" ]; then
                        echo "⚠️  synthetic captions required for '$DATASET_PATH_OR_NAME'. Skipping."
                        continue
                    fi
                    if [ "$CAPTION_POLICY" = "human_or_synthetic" ] && [ "$RUN_CAPTION_SOURCE" != "human" ] && [ "$RUN_CAPTION_SOURCE" != "synthetic" ]; then
                        echo "⚠️  human or synthetic captions required for '$DATASET_PATH_OR_NAME'. Skipping."
                        continue
                    fi
                fi
                EFFECTIVE_DATASET_SPLIT=""
                if [ "$IS_CAPTIONING_DATASET" = "true" ]; then
                    EFFECTIVE_DATASET_SPLIT=""
                elif [ "$SPLIT_SUPPORT" = "true" ] && [ -n "$RUN_DATASET_SPLIT" ]; then
                    EFFECTIVE_DATASET_SPLIT="$RUN_DATASET_SPLIT"
                elif [ "$SPLIT_SUPPORT" = "false" ] && [ -n "$RUN_DATASET_SPLIT" ]; then
                    echo "⚠️  dataset_split not supported for '$DATASET_PATH_OR_NAME'. Skipping."
                    continue
                fi
                EFFECTIVE_DOWNSAMPLE_SIZE="$RUN_DOWNSAMPLE_SIZE"
                if [ "$IS_CAPTIONING_DATASET" != "true" ] && [ -n "$EFFECTIVE_DATASET_SPLIT" ]; then
                    EFFECTIVE_DOWNSAMPLE_SIZE=""
                fi
                EFFECTIVE_REPEATED_IMAGES="$RUN_REPEATED_IMAGES"
                if [ "$IS_CAPTIONING_DATASET" = "true" ]; then
                    EFFECTIVE_REPEATED_IMAGES="false"
                elif [ "$IS_MC_DATASET" != "true" ] && [ "$RUN_REPEATED_IMAGES" = "true" ]; then
                    EFFECTIVE_REPEATED_IMAGES="false"
                fi
                EFFECTIVE_OPEN_ENDED="$RUN_OPEN_ENDED"
                if [ "$IS_CAPTIONING_DATASET" = "true" ]; then
                    EFFECTIVE_OPEN_ENDED="false"
                elif [ "$IS_MC_OPENENDED_CAPABLE" != "true" ] && [ "$RUN_OPEN_ENDED" = "true" ]; then
                    EFFECTIVE_OPEN_ENDED="false"
                fi

                if [ "$IS_CAPTIONING_DATASET" = "true" ]; then
                    GUIDE_TEXT=""
                elif [ "$IS_OPENVQA_DATASET" = "true" ]; then
                    GUIDE_TEXT="$GUIDE_TEXT_OPENENDED"
                elif [ "$IS_MC_OPENENDED_CAPABLE" = "true" ]; then
                    if [ "$EFFECTIVE_OPEN_ENDED" = "true" ]; then
                        GUIDE_TEXT="$GUIDE_TEXT_OPENENDED"
                    else
                        GUIDE_TEXT="$GUIDE_TEXT_MULTICHOICE"
                    fi
                else
                    GUIDE_TEXT="$GUIDE_TEXT_MULTICHOICE"
                fi

                CMD_ARGS=(
                    --results-dir "$RESULTS_DIR"
                    --representations-dir "$REPRESENTATIONS_DIR"
                    --dataset-path-or-name "$DATASET_PATH_OR_NAME"
                    --model-name-or-path "$MODEL_NAME"
                    --task "$([ "$IS_CAPTIONING_DATASET" = "true" ] && echo captioning || echo vqa)"
                    --batch-size "$BATCH_SIZE"
                    --seed "$SEED"
                    --guide-text "$GUIDE_TEXT"
                    --pooling-methods "$IMG_IMG_SPEC"
                    --similarity-measures ${SIMILARITY_MEASURES}
                    --regression-type "$REGRESSION_TYPE"
                    --img-img-pair
                )

                if [ "$IS_CAPTIONING_DATASET" != "true" ]; then
                    CMD_ARGS+=(--caption-source "$EFFECTIVE_CAPTION_SOURCE")
                fi

                if [ -n "$EFFECTIVE_DOWNSAMPLE_SIZE" ]; then CMD_ARGS+=(--downsample-size "$EFFECTIVE_DOWNSAMPLE_SIZE"); fi
                if [ "$SAVE_REPRESENTATIONS" = "true" ]; then CMD_ARGS+=(--save-representations); fi
                if [ "$ENABLE_SEMANTIC_FILTER" = "true" ]; then CMD_ARGS+=(--enable-semantic-filter); fi
                if [ "$ENABLE_CLIP_METRICS" = "true" ]; then CMD_ARGS+=(--enable-clip-metrics); fi
                if [ "$FORCE_RECOMPUTE" = "true" ]; then CMD_ARGS+=(--force); fi
                if [ "$FORCE_METRICS_ONLY" = "true" ]; then CMD_ARGS+=(--force-metrics-only); fi
                if [ "$EFFECTIVE_REPEATED_IMAGES" = "true" ] && [ "$IS_MC_DATASET" = "true" ]; then CMD_ARGS+=(--repeated-images); fi
                if [ "$EFFECTIVE_OPEN_ENDED" = "true" ] && [ "$IS_MC_OPENENDED_CAPABLE" = "true" ]; then CMD_ARGS+=(--open-ended); fi
                if [ "$REPRESENTATIONS_ONLY" = "true" ]; then CMD_ARGS+=(--representations-only); fi
                if [ "$IS_CAPTIONING_DATASET" = "true" ]; then
                    CMD_ARGS+=(--captioning-image-dir "$CAPTIONING_IMAGE_DIR")
                elif [ -n "$EFFECTIVE_DATASET_SPLIT" ]; then
                    CMD_ARGS+=(--dataset-split "$EFFECTIVE_DATASET_SPLIT")
                fi
                if [ "${#CAMBRIAN_VISION_ENCODERS[@]}" -gt 0 ]; then CMD_ARGS+=(--cambrian-vision-encoders "${CAMBRIAN_VISION_ENCODERS[@]}"); fi

                python3 "$FILEPATH" "${CMD_ARGS[@]}"
                echo "    ✅ Done (img-img): $MODEL_NAME | $IMG_IMG_SPEC | $DATASET_PATH_OR_NAME"
            done
        done
    done
fi

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
