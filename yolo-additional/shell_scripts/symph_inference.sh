#!/bin/bash -l

#SBATCH --job-name=symph-inf
#SBATCH --time=01:30:00
#SBATCH --partition=gpu-short
#SBATCH --output=/home/mrajaraman/slurm/yolov11/inference/symph/output-%A.out
#SBATCH --gres=gpu:1

echo "## Starting GPU test on $HOSTNAME"
module purge

echo "## Loading module"
module load ALICE/default
module load Miniconda3
conda init

# Activate Conda environment
source activate dragonfly_yolo

# Navigate to project directory
cd /home/mrajaraman/master-thesis-dragonfly/yolo-model

python symph_inference.py

echo "Script executed"
