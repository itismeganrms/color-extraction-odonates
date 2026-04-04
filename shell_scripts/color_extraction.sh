#!/bin/bash -l

#SBATCH --job-name=color-extra
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-mig-40g
#SBATCH --output=/home/mrajaraman/slurm/yolov11/color-extraction/kmeans/output-%A.out
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
cd /home/mrajaraman/master-thesis-dragonfly/yolo-model/color_extraction

python color_extraction_kmeans.py

echo "Script executed"