#!/bin/bash

#SBATCH --job-name=maskdino
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-mig-40g
#SBATCH --output=/home/mrajaraman/slurm/maskdino/merged-train/output-%A.out
#SBATCH --gres=gpu:1

echo "## Starting GPU test on $HOSTNAME"
module purge

echo "## Loading module"
module load ALICE/default
module load Miniconda3
conda init

source activate /home/mrajaraman/conda/maskdino

# Debugging outputs
pwd
which conda
python --version
# pip freeze

# LazyConfig Training Script - pretrained new baseline
TILE_SIZE=512
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# ---- TRAINING COMMANDS FOR ORIGINAL DATASET BELOW ----
# python train_net.py --num-gpus 1 \
# --exp_id ${TILE_SIZE} \
# --train_iter 2500 \
# --config-file /home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/configs/lifeplan/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml \
# --dataset_path /home/mrajaraman/dataset/coco-roboflow/ \
# OUTPUT_DIR output_${TILE_SIZE}_dragonfly_${TIMESTAMP} \
# DATASETS.TRAIN "(\"dragonfly_${TILE_SIZE}_train\",)" \
# DATASETS.TEST "(\"dragonfly_${TILE_SIZE}_valid\",)"  \
# MODEL.WEIGHTS /h/jquinto/MaskDINO/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth \

# python train_net.py --num-gpus 1 \
# --exp_id ${TILE_SIZE} \
# --train_iter 5000 \
# --config-file /home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/configs/lifeplan/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml \
# --dataset_path /home/mrajaraman/dataset/coco-roboflow/ \
# OUTPUT_DIR output_${TILE_SIZE}_dragonfly_${TIMESTAMP} \
# DATASETS.TRAIN "(\"dragonfly_${TILE_SIZE}_train\",)" \
# DATASETS.TEST "(\"dragonfly_${TILE_SIZE}_valid\",)"  \

python train_net.py --num-gpus 1 \
--exp_id ${TILE_SIZE} \
--train_iter 15000 \
--config-file /home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/configs/lifeplan/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml \
--dataset_path /home/mrajaraman/dataset/coco-roboflow/ \
OUTPUT_DIR output_${TILE_SIZE}_dragonfly_${TIMESTAMP} \
DATASETS.TRAIN "(\"dragonfly_${TILE_SIZE}_train\",)" \
DATASETS.TEST "(\"dragonfly_${TILE_SIZE}_valid\",)"  \
# MODEL.WEIGHTS /h/jquinto/MaskDINO/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth \

# # ---- TRAINING COMMANDS FOR MERGED DATASET BELOW ----
python /home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/train_net_merged.py --num-gpus 1 \
--exp_id ${TILE_SIZE} \
--train_iter 14550 \
--config-file /home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/configs/lifeplan/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml \
--dataset_path /home/mrajaraman/dataset/dataset-v2-coco/ \
OUTPUT_DIR output_${TILE_SIZE}_dragonfly_${TIMESTAMP} \
DATASETS.TRAIN "(\"dragonfly_merged_train\",)" \
DATASETS.TEST "(\"dragonfly_merged_valid\",)"  \
# MODEL.WEIGHTS /h/jquinto/MaskDINO/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth \