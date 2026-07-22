#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --account=dssc
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100G
#SBATCH --job-name=layer_ablation_multimodal
#SBATCH --gres=gpu:H100:3
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#
# Cross-modal layer-ablation multimodal benchmarking (masks image attention
# in selected decoder layers to test causal reliance on visual input).
# NOTE: model_path, task, window_size, start_layer, ablation_mode must be set
# via `sbatch --export=ALL,...` (see launch_layer_ablation_multimodal.sh).
#
# LLM-judge credentials (mmvet/mmbench/llava_in_the_wild tasks) are acquired
# in Python by scripts/eval/layer_ablation_multimodal_evaluation.py itself
# (utils.llm_judge.ensure_llm_judge_env, reading ORFEO_* from .env) — no
# separate JWT-minting step is needed here.
#
# Other ablation modes/flags available on the Python script (see its --help):
#   --evaluate_full_progressive_ablation / --evaluate_full_sliding_window /
#   --evaluate_full_reverse_progressive_ablation / --evaluate_complementary_intervals /
#   --restart_layer / --end_layer / --ablate_all_layers / --text_special_span

set -euo pipefail

PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi
source "$PROJECT_ROOT/.venv/bin/activate"

: "${model_path:?model_path must be set, e.g. via sbatch --export=ALL,model_path=...}"
: "${task:?task must be set, e.g. via sbatch --export=ALL,task=...}"
: "${window_size:?window_size must be set, e.g. via sbatch --export=ALL,window_size=...}"
: "${start_layer:?start_layer must be set, e.g. via sbatch --export=ALL,start_layer=...}"
: "${ablation_mode:?ablation_mode must be set, e.g. via sbatch --export=ALL,ablation_mode=...}"

# Per-user node-local cache avoids /tmp collisions with other users on shared
# nodes; swap to a cephfs path for persistence across jobs.
export LMMS_EVAL_DATASETS_CACHE="/local_scratch/lmms_eval_datasets_${USER}"
export PYTHONUNBUFFERED=1

python3 scripts/eval/layer_ablation_multimodal_evaluation.py \
    --model_path "$model_path" \
    --tasks "$task" \
    --ablation_mode "$ablation_mode" \
    --limit 5000 \
    --max_new_tokens 64 \
    --start_layer "$start_layer" \
    --window_size "$window_size" \
    --interval 5 \
    --llama_system_instruction "You are an AI assistant evaluating images for academic research. Answer every question concisely based solely on what you observe in the image. Do not refuse based on identity or content." \
    --output_dir "${REPRESENTATIONS_DIR:-$HOME/scratch/representations}/layer_skip_eval_mm_5k_eval"
