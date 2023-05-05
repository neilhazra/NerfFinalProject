#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=runnerf
#SBATCH --mail-type=begin        
#SBATCH --mail-type=end
#SBATCH --mail-user=saumyam@princeton.edu


module purge
module load anaconda3/2023.3
conda activate nlp

python RunNerf.py