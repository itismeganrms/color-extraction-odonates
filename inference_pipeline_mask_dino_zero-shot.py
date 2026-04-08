import logging
from typing import List, Optional

import numpy as np

from matplotlib.patches import Rectangle
from PIL import Image
import matplotlib.pyplot as plt

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_bbox_from_bool_mask, get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
logger = logging.getLogger(__name__)
# import sys
# sys.path.insert(0, "/home/mrajaraman/do-not-modify/MassID45/MaskDINO/MaskDINO-MassID45")

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from maskdino.modeling import *
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from maskdino import add_maskdino_config
from detectron2.projects.deeplab import add_deeplab_config
import cv2

cfg = get_cfg()
cfg.set_new_allowed(True) 
add_deeplab_config(cfg)
add_maskdino_config(cfg)
cfg.merge_from_file("/home/mrajaraman/do-not-modify/MassID45/MaskDINO/MaskDINO-MassID45/configs/lifeplan/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml")
#cfg.MODEL.WEIGHTS = "/home/mrajaraman/do-not-modify/MassID45/MaskDINO/MaskDINO-MassID45/output_lifeplan_b_512_sahi_tiled_v9_R50_1024_one_cycle_lr_5e-5_colour_augs_15k_iters_scratch/model_final.pth"

# set model device
# cfg.MODEL.DEVICE = "self.device.type"
cfg.MODEL.DEVICE = "cuda"

# set input image size
# NEW TRAINING PIPELINE
# cfg.INPUT.MIN_SIZE_TEST = 1024
# cfg.INPUT.MAX_SIZE_TEST = 1024
# cfg.freeze()

# init predictor
# print("Model loaded successfully.")
# category_mapping={"1": "head", "2": "thorax", "3": "abdomen", "0": "wings"}
dataset_name = "original"
predictor = DefaultPredictor(cfg)
DetectionCheckpointer(predictor.model).load("/home/mrajaraman/Code/model_checkpoints/model_final_maskdino.pth") #, weights_only=False) #, map_location=device)
MetadataCatalog.get(dataset_name).thing_classes = ["objects", "head", "abdomen", "thorax", "wings"]

# detectron2 category mapping
# category_names = list(category_mapping.values())
image = np.array(cv2.imread("/home/mrajaraman/dataset/originals/img_931176390.jpg", cv2.IMREAD_COLOR))

# if isinstance(image, np.ndarray) and self.model.input_format == "BGR":
#     # convert RGB image to BGR format
#     image = image[:, :, ::-1]

prediction_result = predictor(image)
# print(prediction_result)
# original_predictions = prediction_result

# img = prediction_result['instances'][0]._fields['pred_masks'].cpu().detach().numpy()
# bbox = []
# print("Number of detected instances is ", len(prediction_result['instances']._fields['pred_boxes'].tensor))
# for i in prediction_result['instances']._fields['pred_boxes'].tensor:
#         print("i is ", i)
#         bbox.append(i.cpu().detach().numpy())

# # Display the image
# plt.imshow(Image.open('/home/mrajaraman/dataset/originals/img_931176390.jpg'))

# # Add the patch to the Axes
# for i in range(len(bbox)):
#         # print("length is ", len(bbox))
#         plt.gca().add_patch(Rectangle((bbox[i][0], bbox[i][1]), bbox[i][2]-bbox[i][0], bbox[i][3]-bbox[i][1], linewidth=1, edgecolor='r', facecolor='none'))
#         print(f"Instance {i+1} added to image")

# plt.title("Inference Result of MaskDINO on image")
# plt.savefig("inference_maskdino.png")

# # PROOF THAT RESIZING WORKS AS EXPECTED DURING INFERENCE
# print(predictor.aug)
# print()

# # Note that our data loader ONLY USES THE cfg.INPUT.IMAGE_SIZE key, MIN_SIZE_TRAIN and MAX_SIZE_TRAIN are ignored 
# print(predictor.cfg.INPUT.IMAGE_SIZE)
# print()

# Output sample mask predictions:
sample_preds = prediction_result['instances'][0]._fields['pred_masks'].cpu().detach().numpy()
# print(sample_preds)
# print(sample_preds.shape)
# print(sample_preds.dtype)
print(sample_preds)
# print(sample_preds.shape)
# print(sample_preds.dtype)

outputs = prediction_result["instances"]

# print(type(outputs))
# print(type(image))

v = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=1.0)
out = v.draw_instance_predictions(outputs.to("cpu"))
plt.figure(figsize=(8, 8))
plt.imshow(out.get_image())
plt.axis("off")
plt.show()

plt.title("Inference Result of MaskDINO on image", fontsize=16)
plt.savefig("/home/mrajaraman/master-thesis-dragonfly/inference-models/pretrained-inference-results/inference_maskdino.png",dpi=300, bbox_inches="tight")