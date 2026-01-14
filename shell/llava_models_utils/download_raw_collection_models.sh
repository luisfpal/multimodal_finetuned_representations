#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --account=dssc
#SBATCH --partition=EPYC
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G

# Project configuration
PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
cd "$PROJECT_ROOT"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate Python environment
source .venv/bin/activate

# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# Disable tokenizer parallelism to avoid fork warnings
export TOKENIZERS_PARALLELISM=false

# Run the model checking script
# python scripts/llava_models_utils/download_original_lbasile_models.py
python scripts/llava_models_utils/format_all_lbasile_models.py
