import logging
from typing import List, Optional
import numpy as np

from matplotlib.patches import Rectangle
from PIL import Image
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import Visualizer

# from sahi.models.base import DetectionModel
# from sahi.prediction import ObjectPrediction
# from sahi.utils.cv import get_bbox_from_bool_mask, get_coco_segmentation_from_bool_mask
# from sahi.utils.import_utils import check_requirements

import detectron2.data.transforms as T

logger = logging.getLogger(__name__)
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from maskdino.modeling import *
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from maskdino import add_maskdino_config
from detectron2.projects.deeplab import add_deeplab_config
import cv2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = get_cfg()
cfg.set_new_allowed(True) 
add_deeplab_config(cfg)
add_maskdino_config(cfg)
cfg.merge_from_file("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/configs/lifeplan/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml")

cfg.MODEL.DEVICE = "cuda"

# set input image size
# NEW TRAINING PIPELINE
# cfg.INPUT.MIN_SIZE_TEST = 1024
# cfg.INPUT.MAX_SIZE_TEST = 1024
# cfg.freeze()

# init predictor
predictor = DefaultPredictor(cfg)
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_lifeplan_b_512_sahi_tiled_v9_R50_1024_one_cycle_lr_5e-5_colour_augs_15k_iters/model_0004884.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_lifeplan_b_512_sahi_tiled_v9_R50_1024_one_cycle_lr_5e-5_colour_augs_15k_iters/model_0009769.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_lifeplan_b_512_sahi_tiled_v9_R50_1024_one_cycle_lr_5e-5_colour_augs_15k_iters/model_0014654.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_lifeplan_b_512_sahi_tiled_v9_R50_1024_one_cycle_lr_5e-5_colour_augs_15k_iters/model_final.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_512_dragonfly_2025-10-02_01-07-38/model_final.pth")

## OLD INFERENCE SNIPPET
dataset_name="dataset_v1_coco"
DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/final_runs_for_consideration/output_512_dragonfly_2025-12-07_15-47-56_clean-plant-171/model_final.pth")
model_name="model_old_2500"

# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/final_runs_for_consideration/output_512_dragonfly_2025-12-07_16-35-15_desert-sun-174/model_final.pth")
# model_name="model_old_5000"

# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/final_runs_for_consideration/output_512_dragonfly_2025-12-15_09-55-04_sandy-wave-198/model_final.pth")
# model_name="model_old_15000"
MetadataCatalog.get(dataset_name).thing_classes = ["dragonfly", "head", "abdomen", "thorax", "wings"]

# ## NEW INFERENCE SNIPPET
# dataset_name="dataset_v2_coco"
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/final_runs_for_consideration/output_512_dragonfly_2025-12-15_15-56-59_lunar-morning-201/model_final.pth")
# MetadataCatalog.get(dataset_name).thing_classes = ["objects","head", "abdomen", "thorax", "wings"]
# model_name="model_final_new_trained"

print("Model loaded successfully.")

# category_mapping={"0": "head", "1": "torso", "2": "tail", "3": "wings"}
# detectron2 category mapping
# category_names = list(category_mapping.values())
image = np.array(cv2.imread("/home/mrajaraman/dataset/originals/img_1458477504.jpg", cv2.IMREAD_COLOR))

print("Inference done on {}", model_name)

prediction_result = predictor(image)
# print(prediction_result)# print(prediction_result)
# original_predictions = prediction_result

# PROOF THAT RESIZING WORKS AS EXPECTED DURING INFERENCE
print(predictor.aug)
print()

# Note that our data loader ONLY USES THE cfg.INPUT.IMAGE_SIZE key, MIN_SIZE_TRAIN and MAX_SIZE_TRAIN are ignored 
# print(predictor.cfg.INPUT.IMAGE_SIZE)
# print()

# Output sample mask predictions:
sample_preds = prediction_result['instances'][0]._fields['pred_masks'].cpu().detach().numpy()
print(sample_preds)
# print(sample_preds.shape)
# print(sample_preds.dtype)

outputs = prediction_result["instances"]

v = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=1.0)
out = v.draw_instance_predictions(outputs.to("cpu"))
plt.figure(figsize=(8, 8))
plt.imshow(out.get_image())
plt.axis("off")
plt.title("Inference Result of Trained MaskDINO on image")
plt.savefig(f"/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/final_runs_for_consideration/all_inference_images/inference_{model_name}.png")
plt.show()