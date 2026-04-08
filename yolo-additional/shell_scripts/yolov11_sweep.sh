#!/bin/bash -l

#SBATCH --job-name=yolo-sweep
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu-long
#SBATCH --output=//home/mrajaraman/slurm/yolov11/sweep/sweep-output-%A.out
#SBATCH --gres=gpu:1

# Load required modules

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

python yolov11_code_sweep.py

echo "Script executed"