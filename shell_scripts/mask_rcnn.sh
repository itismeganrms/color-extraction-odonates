#!/bin/bash -l

#SBATCH --job-name=maskrcnn
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-long
#SBATCH --output=/home/mrajaraman/slurm/maskrcnn/output-%A.out
#SBATCH --gres=gpu:1

echo "## Starting GPU test on $HOSTNAME"
module purge

echo "## Loading module"
module load ALICE/default
module load Miniconda3
conda init

# Activate Conda environment
source activate /home/mrajaraman/conda/envs/dragonfly

# Navigate to project directory
cd /home/mrajaraman/master-thesis-dragonfly/inference-models

python mask_rcnn_code.py

echo "Script executed"
