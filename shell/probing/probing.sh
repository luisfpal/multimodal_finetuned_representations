#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --account=dssc
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100G
#SBATCH --job-name=linguistic_probing
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#
# SentEval-style linguistic probing at a fixed decoder layer.
# NOTE: model_path, target_layer must be set via `sbatch --export=ALL,...`
# (see launch_probing.sh). task_name is fixed to "semantic" below.

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
: "${target_layer:?target_layer must be set, e.g. via sbatch --export=ALL,target_layer=...}"

export PYTHONUNBUFFERED=1

python3 scripts/probing/linguistic_probing.py \
    --model_name_or_path "$model_path" \
    --task_name semantic \
    --target_layer "$target_layer" \
    --out_dir "${REPRESENTATIONS_DIR:-$HOME/scratch/representations}/probing_tasks" \
    --data_path ./data/probing \
    --batch_size 16 \
    --n_rep 2 \
    --subset_size 5000
