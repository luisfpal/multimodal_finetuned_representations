#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --account=dssc
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --job-name=retrieval
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#
# Cross-modal retrieval evaluation (recall@k, NDCG, mrr, mean_percentile_rank)
# over embeddings cached by extract_embeddings.sh. NOTE: model must be set via
# `sbatch --export=ALL,model=...` (see launch_retrieval.sh).
#
# Suffixes (image folder): eoi imgavg imgeoi last textavg textspec
# Suffixes (text folder):  last textavg textspec userlast

set -euo pipefail

PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi
source "$PROJECT_ROOT/.venv/bin/activate"

: "${model:?model must be set, e.g. via sbatch --export=ALL,model=...}"

export PYTHONUNBUFFERED=1

# ---------- COCO Karpathy (paired retrieval, 5000 image<->description pairs) ----------
python3 scripts/retrieval/evaluate_retrieval.py \
    --model "$model" \
    --suffix1 imgavg --suffix2 textavg \
    --ks 10 30 100 300 1000 \
    --part_types res2 \
    --folder1_name coco_karpathy_test_image_only \
    --folder2_name coco_karpathy_test_text_only \
    --retrieval_dataset_name coco_karpathy_test

# ---------- Winoground (text/image/group scores, 400 entries x 2 sides) ----------
# python3 scripts/retrieval/evaluate_retrieval.py \
#     --model "$model" \
#     --suffix1 imgavg --suffix2 textavg \
#     --task winoground \
#     --part_types res2 \
#     --folder1_name winoground_test_image_only \
#     --folder2_name winoground_test_text_only \
#     --retrieval_dataset_name winoground_test
