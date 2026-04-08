#!/bin/bash -l

#SBATCH --job-name=former_infer
#SBATCH --time=03:00:00
#SBATCH --partition=gpu-short
#SBATCH --output=/home/mrajaraman/slurm/inference/mask2former/output-%A.out
#SBATCH --gres=gpu:1

echo "## Starting GPU test on $HOSTNAME"
module purge

echo "## Loading module"
module load ALICE/default
module load Miniconda3
conda init

# Activate Conda environment
source activate /home/mrajaraman/conda/mask2former

# Navigate to project directory
cd /home/mrajaraman/do-not-modify/MassID45/Mask2Former/Mask2Former-MassID45

python inference_pipeline_mask2former.py

echo "Script executed"
