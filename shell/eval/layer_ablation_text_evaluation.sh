#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --account=dssc
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
#SBATCH --job-name=layer_ablation_text
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#
# Text-only lm-eval-harness benchmarking with early-layer-skipping ablation
# (more general than scripts/eval/ablation_layers_evaluation.py: sliding-window,
# progressive, and reverse-progressive ablation modes across mmlu/hellaswag/
# nq_open/triviaqa/etc.). NOTE: task, window_size must be set via
# `sbatch --export=ALL,...` (see launch_layer_ablation_text.sh).
#
# MODEL_PATH below picks the target model; supported: LLaVA 1.5 7B/13B,
# InternVL2-8B, Cambrian-8B, LLaVA-OneVision-1.5-4B/8B.

set -euo pipefail

PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi
source "$PROJECT_ROOT/.venv/bin/activate"

: "${task:?task must be set, e.g. via sbatch --export=ALL,task=...}"
: "${window_size:?window_size must be set, e.g. via sbatch --export=ALL,window_size=...}"

MODEL_PATH="lbasile/llava_7b"

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes[0]}
GPU_NUMBER=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' ' | wc -w)
export PYTHONUNBUFFERED=1

torchrun \
    --nnodes="$SLURM_NTASKS" \
    --node-rank="$SLURM_NODEID" \
    --nproc-per-node="$GPU_NUMBER" \
    --rdzv-id="$SLURM_PROCID" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$head_node:0" \
    scripts/eval/layer_ablation_text_evaluation.py \
    --model_path "$MODEL_PATH" \
    --tasks "$task" \
    --window_size "$window_size" \
    --interval 2 \
    --evaluate_full_sliding_window
