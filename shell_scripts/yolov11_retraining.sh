#!/bin/bash -l

#SBATCH --job-name=yolov11
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-l4-24g
#SBATCH --output=/home/mrajaraman/slurm/yolov11/retrain/output-%A.out
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

python yolov11_code_retrain.py

echo "Script executed"