#!/bin/bash
#SBATCH --job-name=download_models
#SBATCH --account=dssc
#SBATCH --partition=EPYC
#SBATCH --output=/u/dssc/lpalaciosflores/multimodal_representations_analysis/logs/check_models_%j.out
#SBATCH --error=/u/dssc/lpalaciosflores/multimodal_representations_analysis/logs/check_models_%j.out
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Change to project root directory
cd /u/dssc/lpalaciosflores/multimodal_representations_analysis

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate Python environment
source .venv/bin/activate

# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/u/dssc/lpalaciosflores/multimodal_representations_analysis/"

# Disable tokenizer parallelism to avoid fork warnings
export TOKENIZERS_PARALLELISM=false

# Run the model checking script (checks all collections)
python scripts/download_raw_collection_models.py