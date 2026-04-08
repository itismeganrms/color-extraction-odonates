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
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.utils.visualizer import Visualizer

logger = logging.getLogger(__name__)

from detectron2.engine.defaults import DefaultPredictor
from mask2former.modeling import *
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from mask2former import add_maskformer2_config
from detectron2.projects.deeplab import add_deeplab_config
import cv2
import torch, sys

from detectron2.data import MetadataCatalog
from detectron2.config import LazyConfig, instantiate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = get_cfg()
cfg.set_new_allowed(True) 
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("/home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
# cfg.MODEL.WEIGHTS = "/h/jquinto/Mask2Former/model_final_R50.pkl"

# set model device
# cfg.MODEL.DEVICE = "self.device.type"
cfg.MODEL.DEVICE = "cuda"

# set input image size
# NEW TRAINING PIPELINE
# cfg.INPUT.MIN_SIZE_TEST = 1024
# cfg.INPUT.MAX_SIZE_TEST = 1024
# cfg.freeze()

# init predictor
predictor = DefaultPredictor(cfg)
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/Code/model_checkpoints/model_final_mask2former.pth")

# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/output_512_dragonfly_2025-10-02_01-01-28/model_final.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/output_512_dragonfly_2025-10-02_01-01-28/model_0011999.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/output_512_dragonfly_2025-12-07_15-50-37/model_final.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/output_512_dragonfly_2025-12-07_16-21-00/model_final.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/output_512_dragonfly_2025-09-22_01-59-38/model_final.pth")

# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_512_dragonfly_2025-10-02_01-07-38/model_0001499.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_512_dragonfly_2025-10-02_01-07-38/model_0002999.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_512_dragonfly_2025-10-02_01-07-38/model_0004499.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_512_dragonfly_2025-10-02_01-07-38/model_0005999.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_512_dragonfly_2025-10-02_01-07-38/model_0007499.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_512_dragonfly_2025-10-02_01-07-38/model_0008999.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_512_dragonfly_2025-10-02_01-07-38/model_0010499.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_512_dragonfly_2025-10-02_01-07-38/model_0011999.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_512_dragonfly_2025-10-02_01-07-38/model_0013499.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_512_dragonfly_2025-10-02_01-07-38/model_0014999.pth")
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/maskdino-dragonfly/output_512_dragonfly_2025-10-02_01-07-38/model_final.pth")

## OLD INFERENCE SNIPPET
# dataset_name="dataset_v1_coco"
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/final_runs_for_consideration/output_512_dragonfly_2025-12-07_15-50-37_jumping_thunder_172/model_final.pth")
# model_name="model_old_2500"

# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/final_runs_for_consideration/output_512_dragonfly_2025-12-07_16-21-00_fearless_blaze_173/model_final.pth")
# model_name="model_old_5000"

# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/final_runs_for_consideration/output_512_dragonfly_2025-12-15_09-54-23_helpful_mountain_197/model_final.pth")
# model_name="model_old_15000"

# MetadataCatalog.get(dataset_name).thing_classes = ["dragonfly", "head", "abdomen", "thorax", "wings"]
# MetadataCatalog.get(dataset_name).set(thing_classes = ["dragonfly", "head", "abdomen", "thorax", "wings"]) 

# register_coco_instances(
#     "dragonfly_512_train",
#     {},                    
#     "/home/mrajaraman/dataset/coco-roboflow/annotations/instances_train.json",
#     "/home/mrajaraman/dataset/coco-roboflow/"
# )


# ## NEW INFERENCE SNIPPET
dataset_name="dataset_v2_coco"
DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/final_runs_for_consideration/output_512_dragonfly_2025-12-15_15-56-37_wobbly-water-200/model_final.pth")
# MetadataCatalog.get(dataset_name).thing_classes = ["objects", "head", "abdomen", "thorax", "wings"]
model_name="model_final_new_trained"

print("Model loaded successfully.") 
image = np.array(cv2.imread('/home/mrajaraman/dataset/originals/img_1458477504.jpg', cv2.IMREAD_COLOR))
print("Inference done on {}", model_name)

prediction_result = predictor(image)
# print(prediction_result)
# original_predictions = prediction_result

# PROOF THAT RESIZING WORKS AS EXPECTED DURING INFERENCE
print(predictor.aug)
# print()

# # Note that our data loader ONLY USES THE cfg.INPUT.IMAGE_SIZE key, MIN_SIZE_TRAIN and MAX_SIZE_TRAIN are ignored 
# print(predictor.cfg.INPUT.IMAGE_SIZE)
# print()

# Output sample mask predictions:
sample_preds = prediction_result['instances'][0]._fields['pred_masks'].cpu().detach().numpy()
print(sample_preds)
# print(sample_preds.shape)
# print(sample_preds.dtype)

outputs = prediction_result["instances"]
# instances = outputs["instances"].to("cpu")

# print(type(outputs))
# print(type(image))

v = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=1.0)

meta = MetadataCatalog.get(dataset_name)

out = v.draw_instance_predictions(outputs.to("cpu"))

plt.figure(figsize=(8, 8))
plt.imshow(out.get_image())
plt.axis("off")
plt.title("Inference of Trained Mask2Former on Image", fontsize=16)
plt.savefig(f"/home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/final_runs_for_consideration/all_inference_images/inference_{model_name}.png",dpi=300, bbox_inches="tight")
plt.show()