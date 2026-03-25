#!/bin/bash
# Wrapper for evaluating LLaVA-OneVision without modifying the default eval script.

set -e

PROJECT_ROOT="/u/dssc/lbasile/scratch/plm"
VENV="$PROJECT_ROOT/.venv"

if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi

# For LLaVA-OneVision models, use llava_hf with evaluator compatibility patch.
export LMMS_EVAL_MODEL="${LMMS_EVAL_MODEL:-llava_hf}"

MODEL_TO_EVAL="${1:-lbasile/LLaVA-OneVision-1.5-8B-stage1}"
EXTRA_ARGS="${2:-}"

if [[ "$EXTRA_ARGS" != *"--trust-remote-code"* ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --trust-remote-code"
fi

if [[ "$EXTRA_ARGS" != *"--max-new-tokens"* ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --max-new-tokens ${ONEVISION_MAX_NEW_TOKENS:-64}"
fi

bash "$PROJECT_ROOT/shell/eval/run_local_evaluation.sh" "$MODEL_TO_EVAL" "$EXTRA_ARGS"
