#!/bin/bash -l

#SBATCH --job-name=rcnn-inference
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-short
#SBATCH --output=/home/mrajaraman/slurm/maskrcnn/inference/output-%A.out
#SBATCH --gres=gpu:1

echo "## Starting GPU test on $HOSTNAME"
module purge

echo "## Loading module"
module load ALICE/default
module load Miniconda3
conda init

# Activate Conda environment
source activate mask_rcnn

# Navigate to project directory
cd /home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly

# python inference_pipeline_trained.py
python unseen_inference.py

echo "Script executed"
