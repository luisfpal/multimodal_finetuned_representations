#!/bin/bash
# Runs representations classification metrics on VLM residual stream representations.
set -euo pipefail

MODELS=(
    "lbasile/llava_7b"
    "l11p/llava-v1.5-7b-pretrained-lbasile"
    # "lbasile/llava_7b_frozen"
    # "lbasile/llava_7b_0_16"
    # "lbasile/llava_7b_16_32"
    # "lbasile/llava_7b_6_16"
    # "lbasile/llava_7b_6_17_complementary"

    "lbasile/llava_13b"
    "l11p/llava-v1.5-13b-pretrained-lbasile"
    # "lbasile/llava_13b_6_21"
    # "lbasile/llava_13b_6_21_complementary"

    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-4B-Base"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Base"

    "OpenGVLab/InternVL2-8B"
    "OpenGVLab/InternVL2-Pretrain-Models/InternVL2-8B-Pretrain"

    "nyu-visionx/cambrian-8b"
    "lbasile/cambrian-8b-pretrained"
)

# Dataset entries are "dataset|split|experiment_family".
# Families:
#   fixed              ScienceQA/MMMU; no --n-choices, no accuracy auto-addition.
#   captioning         Non-MCQ image prompts; optional --subsample-size, no accuracy.
#   mcq_option_width   Non-filtered synthetic MCQ; n = answer-option count.
#   mcq_filter_classes Filtered synthetic MCQ; n = selected class count and option count.
DATASET_EXPERIMENTS=(
    # "derek-thomas/ScienceQA|test|fixed"
    # "lmms-lab/MMMU|test|fixed"
    # "timm/mini-imagenet|validation|captioning"
    # "timm/mini-imagenet|validation|mcq_option_width"
    # "timm/mini-imagenet|validation|mcq_filter_classes"
    # "RitAreaSciencePark/imagenet_short_text_100_classes_x_100_samples|train|captioning"
    # "RitAreaSciencePark/imagenet_short_text_100_classes_x_100_samples|train|mcq_option_width"
    # "RitAreaSciencePark/imagenet_short_text_100_classes_x_100_samples|train|mcq_filter_classes"
    # "ethz/food101|validation|mcq_option_width"   # generic_mcq keeps all 101 classes, 100 samples/class
    # "ethz/food101|validation|mcq_filter_classes"
    # "tanganke/sun397|test|mcq_option_width"       # generic_mcq keeps 200 seeded classes, 50 samples/class
    # "tanganke/sun397|test|mcq_filter_classes"
    # "flwrlabs/caltech101|train|mcq_option_width"  # generic_mcq keeps the full imbalanced train split
    "flwrlabs/caltech101|train|mcq_filter_classes"
)

TOKENS_POOLING_METHODS=(
    "last"
    # "mean_image"
)

# Select any subset. Synthetic MCQ extraction runs can auto-add "accuracy" below.
METRICS=(
    # "logistic_regression"
    "label_overlap"
    "clip_neighborhood_overlap"
    "accuracy"
    # "knn"
    # "homogeneity"
)

# Prompt modes to try for non-fixed families. Unsupported prompt/dataset/family
# combinations are skipped with a message rather than sent to Python.
PROMPT_MODES=(
    # captioning: describe continuation identify category_id animal_id
    # mcq_option_width: mcq mcq_nocat animal_mcq generic_mcq
    # mcq_filter_classes: mcq_nocat_filter animal_mcq_filter generic_mcq_filter generic_mcq_top_filter
    # "mcq_nocat_filter"
    # "animal_mcq_filter"
    # "generic_mcq_filter"
    "generic_mcq_top_filter"
    # "describe"
    # "generic_mcq"
    # "animal_mcq"
    # "mcq_nocat"
)

# MCQ sweep axes. The same --n-choices flag is intentionally fed from different
# lists depending on family, because the meaning is different.
MCQ_OPTION_COUNTS_LIST=("")        # n = number of answer options
MCQ_FILTER_CLASS_COUNTS_LIST=("20")  # n = number of filtered classes/options
SUBSAMPLE_SIZE_LIST=("")         # "" means no subsample; ignored by filter modes

DEFAULT_GUIDE_TEXT="Answer with the option's letter from the given choices directly."
MAX_SAMPLES="${MAX_SAMPLES:-}"
GUIDE_TEXT="${GUIDE_TEXT:-$DEFAULT_GUIDE_TEXT}"
SAVE_REPRESENTATIONS="${SAVE_REPRESENTATIONS:-true}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-true}"
REPRESENTATIONS_ONLY="${REPRESENTATIONS_ONLY:-false}"
METRICS_ONLY="${METRICS_ONLY:-false}"
CLUSTER_LABELS="${CLUSTER_LABELS:-false}"
SAVE_PROMPTS="${SAVE_PROMPTS:-false}"
AUTO_ACCURACY_FOR_SYNTHETIC_MCQ="${AUTO_ACCURACY_FOR_SYNTHETIC_MCQ:-true}"

# Label source for representation metrics: "class_name" or "option_letter".
# Non-MCQ and fixed families always use class_name.
LABEL_MODE="${LABEL_MODE:-class_name}"

# MMMU label granularity: "intermediate" (subjects) or "detailed" (subfields >= mean count).
MMMU_LABEL_GRANULARITY="${MMMU_LABEL_GRANULARITY:-detailed}"
N_SHOTS="${N_SHOTS:-0}"

# Label-overlap sweep. "" omits the flag and uses the Python default.
CLASS_FRACTION_LIST=("0.3")
MAXK_LIST=("")

# Dry run prints commands instead of executing them. DRY_RUN_LIMIT=0 means no cap.
DRY_RUN="${DRY_RUN:-false}"
DRY_RUN_LIMIT="${DRY_RUN_LIMIT:-0}"

if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
    IFS=',' read -r -a MODELS <<<"$MODELS_OVERRIDE"
fi
if [[ -n "${DATASET_EXPERIMENTS_OVERRIDE:-}" ]]; then
    IFS=',' read -r -a DATASET_EXPERIMENTS <<<"$DATASET_EXPERIMENTS_OVERRIDE"
fi
if [[ -n "${PROMPT_MODES_OVERRIDE:-}" ]]; then
    IFS=',' read -r -a PROMPT_MODES <<<"$PROMPT_MODES_OVERRIDE"
fi
if [[ -n "${METRICS_OVERRIDE:-}" ]]; then
    IFS=',' read -r -a METRICS <<<"$METRICS_OVERRIDE"
fi
if [[ -n "${MCQ_OPTION_COUNTS_OVERRIDE:-}" ]]; then
    IFS=',' read -r -a MCQ_OPTION_COUNTS_LIST <<<"$MCQ_OPTION_COUNTS_OVERRIDE"
fi
if [[ -n "${MCQ_FILTER_CLASS_COUNTS_OVERRIDE:-}" ]]; then
    IFS=',' read -r -a MCQ_FILTER_CLASS_COUNTS_LIST <<<"$MCQ_FILTER_CLASS_COUNTS_OVERRIDE"
fi
if [[ -n "${SUBSAMPLE_SIZE_OVERRIDE:-}" ]]; then
    IFS=',' read -r -a SUBSAMPLE_SIZE_LIST <<<"$SUBSAMPLE_SIZE_OVERRIDE"
fi

PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$HOME/scratch}"
PY_SCRIPT="$PROJECT_DIR/scripts/representations_interpretability/representations_classification.py"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

is_true() {
    [[ "${1:-false}" == "true" ]]
}

contains_metric() {
    local needle="$1"
    shift
    local metric
    for metric in "$@"; do
        [[ "$metric" == "$needle" ]] && return 0
    done
    return 1
}

is_filter_prompt() {
    case "$1" in
        mcq_nocat_filter|animal_mcq_filter|generic_mcq_filter|generic_mcq_top_filter) return 0 ;;
        *) return 1 ;;
    esac
}

is_mcq_prompt() {
    case "$1" in
        mcq|mcq_nocat|mcq_nocat_filter|animal_mcq|animal_mcq_filter|generic_mcq|generic_mcq_filter|generic_mcq_top_filter) return 0 ;;
        *) return 1 ;;
    esac
}

is_synthetic_mcq_dataset() {
    case "$1" in
        *mini-imagenet*|*imagenet_short_text*|*food101*|*sun397*|*caltech101*) return 0 ;;
        *) return 1 ;;
    esac
}

prompt_supported_for_dataset() {
    local dataset_name="$1"
    local prompt_mode="$2"
    case "$dataset_name" in
        *ScienceQA*|*MMMU*)
            [[ "$prompt_mode" == "describe" ]]
            ;;
        *mini-imagenet*)
            case "$prompt_mode" in
                describe|continuation|identify|category_id|mcq|mcq_nocat|mcq_nocat_filter) return 0 ;;
                *) return 1 ;;
            esac
            ;;
        *imagenet_short_text*)
            case "$prompt_mode" in
                describe|continuation|animal_id|animal_mcq|animal_mcq_filter) return 0 ;;
                *) return 1 ;;
            esac
            ;;
        *food101*|*sun397*)
            case "$prompt_mode" in
                generic_mcq|generic_mcq_filter) return 0 ;;
                *) return 1 ;;
            esac
            ;;
        *caltech101*)
            case "$prompt_mode" in
                generic_mcq|generic_mcq_top_filter) return 0 ;;
                *) return 1 ;;
            esac
            ;;
        *)
            return 1
            ;;
    esac
}

prompt_matches_family() {
    local family="$1"
    local prompt_mode="$2"
    case "$family" in
        fixed)
            [[ "$prompt_mode" == "describe" ]]
            ;;
        captioning)
            ! is_mcq_prompt "$prompt_mode"
            ;;
        mcq_option_width)
            is_mcq_prompt "$prompt_mode" && ! is_filter_prompt "$prompt_mode"
            ;;
        mcq_filter_classes)
            is_filter_prompt "$prompt_mode"
            ;;
        *)
            return 1
            ;;
    esac
}

build_metrics_for_run() {
    local synthetic_mcq="$1"
    RUN_METRICS=()
    local metric
    for metric in "${METRICS[@]}"; do
        if [[ "$metric" == "accuracy" && "$synthetic_mcq" != "true" ]]; then
            continue
        fi
        RUN_METRICS+=("$metric")
    done
    if (
        [[ "$synthetic_mcq" == "true" ]] &&
        is_true "$AUTO_ACCURACY_FOR_SYNTHETIC_MCQ" &&
        ! is_true "$METRICS_ONLY" &&
        ! is_true "$REPRESENTATIONS_ONLY" &&
        ! contains_metric "accuracy" "${RUN_METRICS[@]}"
    ); then
        RUN_METRICS+=("accuracy")
    fi
}

print_command() {
    printf '%q ' "$PYTHON_BIN" "$PY_SCRIPT" "$@"
    printf '\n'
}

run_or_print() {
    if is_true "$DRY_RUN"; then
        echo "DRY RUN:"
        print_command "$@"
    else
        "$PYTHON_BIN" "$PY_SCRIPT" "$@"
    fi
}

START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
COMMAND_COUNT=0

echo "====== Representations Classification ======"
echo "Results directory: $RESULTS_DIR"
echo "Representations directory: $REPRESENTATIONS_DIR"
echo "Metrics: ${METRICS[*]}"
echo "Auto accuracy for synthetic MCQ: $AUTO_ACCURACY_FOR_SYNTHETIC_MCQ"
echo "Prompt logs: $SAVE_PROMPTS"
echo "Prompt modes: ${PROMPT_MODES[*]}"
echo "Pooling methods: ${TOKENS_POOLING_METHODS[*]}"
echo "MMMU label granularity: $MMMU_LABEL_GRANULARITY"
echo "ICL shots: $N_SHOTS"
echo "Class fractions: ${CLASS_FRACTION_LIST[*]}"
echo "Max-k values: ${MAXK_LIST[*]}"
echo "MCQ option counts: ${MCQ_OPTION_COUNTS_LIST[*]}"
echo "MCQ filter class counts: ${MCQ_FILTER_CLASS_COUNTS_LIST[*]}"
echo "Subsample sizes: ${SUBSAMPLE_SIZE_LIST[*]}"
echo "Dry run: $DRY_RUN"
echo "Started at: $START_HUMAN"

for MODEL_NAME in "${MODELS[@]}"; do
    echo ""
    echo "Processing model: $MODEL_NAME"

    for DATASET_ENTRY in "${DATASET_EXPERIMENTS[@]}"; do
        IFS='|' read -r DATASET_NAME DATASET_SPLIT EXPERIMENT_FAMILY <<<"$DATASET_ENTRY"
        echo "  Dataset: $DATASET_NAME | split=$DATASET_SPLIT | family=$EXPERIMENT_FAMILY"

        if [[ "$EXPERIMENT_FAMILY" == "fixed" ]]; then
            RUN_PROMPT_MODES=("describe")
        else
            RUN_PROMPT_MODES=("${PROMPT_MODES[@]}")
        fi

        for PROMPT_MODE in "${RUN_PROMPT_MODES[@]}"; do
            if ! prompt_supported_for_dataset "$DATASET_NAME" "$PROMPT_MODE"; then
                echo "    Skip prompt '$PROMPT_MODE': unsupported for $DATASET_NAME"
                continue
            fi
            if ! prompt_matches_family "$EXPERIMENT_FAMILY" "$PROMPT_MODE"; then
                echo "    Skip prompt '$PROMPT_MODE': incompatible with family $EXPERIMENT_FAMILY"
                continue
            fi

            case "$EXPERIMENT_FAMILY" in
                fixed|captioning)
                    N_VALUES=("")
                    ;;
                mcq_option_width)
                    N_VALUES=("${MCQ_OPTION_COUNTS_LIST[@]}")
                    ;;
                mcq_filter_classes)
                    N_VALUES=("${MCQ_FILTER_CLASS_COUNTS_LIST[@]}")
                    ;;
                *)
                    echo "    Unknown experiment family: $EXPERIMENT_FAMILY"
                    exit 1
                    ;;
            esac

            if [[ "$EXPERIMENT_FAMILY" == "fixed" || "$EXPERIMENT_FAMILY" == "mcq_filter_classes" ]]; then
                SUBSAMPLES=("")
            else
                SUBSAMPLES=("${SUBSAMPLE_SIZE_LIST[@]}")
            fi

            for TOKENS_POOLING_METHOD in "${TOKENS_POOLING_METHODS[@]}"; do
                for N_CHOICES in "${N_VALUES[@]}"; do
                    for SUBSAMPLE in "${SUBSAMPLES[@]}"; do
                        for CLASS_FRACTION in "${CLASS_FRACTION_LIST[@]}"; do
                            for MAXK in "${MAXK_LIST[@]}"; do
                                SYNTHETIC_MCQ="false"
                                if is_synthetic_mcq_dataset "$DATASET_NAME" && is_mcq_prompt "$PROMPT_MODE"; then
                                    SYNTHETIC_MCQ="true"
                                fi
                                build_metrics_for_run "$SYNTHETIC_MCQ"
                                if [[ "${#RUN_METRICS[@]}" -eq 0 ]]; then
                                    echo "    Skip run: no metrics remain after family filtering."
                                    continue
                                fi

                                RUN_LABEL_MODE="$LABEL_MODE"
                                if [[ "$SYNTHETIC_MCQ" != "true" ]]; then
                                    RUN_LABEL_MODE="class_name"
                                fi

                                CMD_ARGS=(
                                    --model-name "$MODEL_NAME"
                                    --dataset-name "$DATASET_NAME"
                                    --split "$DATASET_SPLIT"
                                    --guide-text "$GUIDE_TEXT"
                                    --results-dir "$RESULTS_DIR"
                                    --representations-dir "$REPRESENTATIONS_DIR"
                                    --tokens-pooling-method "$TOKENS_POOLING_METHOD"
                                    --metrics "${RUN_METRICS[@]}"
                                    --mmmu-label-granularity "$MMMU_LABEL_GRANULARITY"
                                    --n-shots "$N_SHOTS"
                                    --prompt-mode "$PROMPT_MODE"
                                    --label-mode "$RUN_LABEL_MODE"
                                )

                                if [[ -n "$N_CHOICES" ]]; then
                                    CMD_ARGS+=(--n-choices "$N_CHOICES")
                                fi

                                if is_true "$SAVE_REPRESENTATIONS"; then
                                    CMD_ARGS+=(--save-representations)
                                else
                                    CMD_ARGS+=(--no-save-representations)
                                fi

                                if is_true "$FORCE_RECOMPUTE"; then
                                    CMD_ARGS+=(--force-recompute-representations)
                                fi

                                if is_true "$REPRESENTATIONS_ONLY"; then
                                    CMD_ARGS+=(--representations-only)
                                fi

                                if is_true "$METRICS_ONLY"; then
                                    CMD_ARGS+=(--metrics-only)
                                fi

                                if is_true "$CLUSTER_LABELS"; then
                                    CMD_ARGS+=(--cluster-labels)
                                fi

                                if [[ -n "$MAX_SAMPLES" ]]; then
                                    CMD_ARGS+=(--max-samples "$MAX_SAMPLES")
                                fi

                                if is_true "$SAVE_PROMPTS"; then
                                    CMD_ARGS+=(--save-prompts)
                                else
                                    CMD_ARGS+=(--no-save-prompts)
                                fi

                                if [[ -n "$CLASS_FRACTION" ]]; then
                                    CMD_ARGS+=(--class-fraction "$CLASS_FRACTION")
                                fi

                                if [[ -n "$MAXK" ]]; then
                                    CMD_ARGS+=(--maxk "$MAXK")
                                fi

                                if [[ -n "$SUBSAMPLE" ]]; then
                                    CMD_ARGS+=(--subsample-size "$SUBSAMPLE")
                                fi

                                run_or_print "${CMD_ARGS[@]}"
                                COMMAND_COUNT=$((COMMAND_COUNT + 1))
                                echo "    Done: model=$MODEL_NAME | dataset=$DATASET_NAME | prompt=$PROMPT_MODE | family=$EXPERIMENT_FAMILY | pooling=$TOKENS_POOLING_METHOD | metrics=${RUN_METRICS[*]} | n=${N_CHOICES:-none} | ss=${SUBSAMPLE:-none} | cf=${CLASS_FRACTION:-default} | mk=${MAXK:-default}"

                                if is_true "$DRY_RUN" && [[ "$DRY_RUN_LIMIT" -gt 0 && "$COMMAND_COUNT" -ge "$DRY_RUN_LIMIT" ]]; then
                                    echo "Dry-run limit reached: $DRY_RUN_LIMIT"
                                    exit 0
                                fi
                            done
                        done
                    done
                done
            done
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
echo "Commands prepared: $COMMAND_COUNT"
echo "Finished at: $END_HUMAN"
echo "Elapsed: $ELAPSED_HMS"
