#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --account=dssc
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --job-name=information_imbalance
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#
# Information imbalance / intrinsic dimension between text and image
# representations cached by extract_embeddings.sh. NOTE: model, dataset,
# ft_mode must be set via `sbatch --export=ALL,...` (see
# launch_information_imbalance.sh) — they are not defined in this file.

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
: "${dataset:?dataset must be set, e.g. via sbatch --export=ALL,dataset=...}"
: "${ft_mode:?ft_mode must be set, e.g. via sbatch --export=ALL,ft_mode=...}"

export PYTHONUNBUFFERED=1

python3 scripts/embeddings_extraction/compute_id_imbalance.py \
    --model_name "$model" \
    --ft_mode "$ft_mode" \
    --open_ended \
    --compute_imbalance \
    --dataset_name "$dataset"
