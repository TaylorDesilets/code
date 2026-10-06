#!/bin/bash
#SBATCH --account=def-mariecci
#SBATCH --job-name=generate_plots
#SBATCH --output=logs/plots_%j.out
#SBATCH --error=logs/plots_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:30:00
export MPLBACKEND=Agg
# Load Python environment
module load StdEnv/2023 python/3.11.5 scipy-stack

# Ensure log directory exists
mkdir -p logs

# Run Python plotting script
python /home/taylor33/code/Correction_applied_plots.py
