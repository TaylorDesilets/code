#!/bin/bash
#SBATCH --account=def-mariecci
#SBATCH --job-name=UndergroundLabs
#SBATCH --output=logs/plots_%j.out
#SBATCH --error=logs/plots_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00

export MPLBACKEND=Agg
module load StdEnv/2023 python/3.11.5 scipy-stack
source ~/mute_env/bin/activate
mkdir -p logs

# Run MUTE data script
python /home/taylor33/code/UndergroundLabs.py