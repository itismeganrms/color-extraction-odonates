#!/bin/bash -l

#SBATCH --job-name=file-download
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu-2080ti-11g
#SBATCH --output=/home/mrajaraman/slurm/other/file-download/output-%A.out
#SBATCH --gres=gpu:1

echo "## Starting GPU test on $HOSTNAME"
module purge

echo "## Loading module"
module load ALICE/default
module load Miniconda3
conda init

echo "Module Loaded"

# Activate Conda environment
source activate /home/mrajaraman/conda/envs/dragonfly
echo "Activated"

# Navigate to project directory
# cd /home/mrajaraman/master-thesis-dragonfly/additional-code
python /home/mrajaraman/master-thesis-dragonfly/additional-code/file_downloader.py

echo "Script executed"