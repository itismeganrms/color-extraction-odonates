import logging
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
import detectron2.data.transforms as T

from detectron2.data import MetadataCatalog

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Config and Model ----
cfg = LazyConfig.load("/home/mrajaraman/do-not-modify/MassID45/Mask-RCNN/Mask-RCNN-MassID45/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py")
cfg.train.init_checkpoint = "/h/jquinto/Mask-RCNN/model_final_14d201.pkl"
cfg.train.device = "cuda"

dataset_name = "original"
model = instantiate(cfg.model)
DetectionCheckpointer(model).load("/home/mrajaraman/Code/model_checkpoints/model_final_mask_rcnn.pth")
MetadataCatalog.get(dataset_name).thing_classes = ["objects", "head", "abdomen", "thorax", "wings"]
# model_name="model_final_new_trained"
model.to(device)
model.eval()

# ---- Load Image ----
image_path = "/home/mrajaraman/dataset/originals/img_931176390.jpg"
orig_image = cv2.imread(image_path)
# orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # convert to RGB
np_image = orig_image.copy()
height, width = orig_image.shape[:2]

# ---- Apply same augmentation used in training ----
mapper = instantiate(cfg.dataloader.test.mapper)
aug = mapper.augmentations
aug_input = T.AugInput(orig_image)
transform = aug(aug_input)
image_resized = aug_input.image

# Convert to tensor
image_tensor = torch.as_tensor(image_resized.astype("float32").transpose(2, 0, 1)).to(device)

# ---- Inference ----
with torch.no_grad():
    prediction_result = model([{"image": image_tensor, "height": height, "width": width}])[0]

# outputs = prediction_result["instances"].to("cpu")
outputs = prediction_result['instances']

# # ---- Resize predicted masks back to original resolution ----
# resized_masks = []
# for m in outputs.pred_masks:
#     m_resized = cv2.resize(
#         m.numpy().astype(np.uint8),
#         (width, height),
#         interpolation=cv2.INTER_NEAREST
#     )
#     resized_masks.append(m_resized)
# outputs.pred_masks = torch.as_tensor(np.stack(resized_masks))

# ---- Visualization ----
v = Visualizer(np_image[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=1.0)
out = v.draw_instance_predictions(outputs.to("cpu"))
plt.figure(figsize=(8, 8))
plt.imshow(out.get_image())
plt.axis("off")
plt.title("Inference Result of Mask R-CNN on image", fontsize=16)
plt.savefig("/home/mrajaraman/master-thesis-dragonfly/inference-models/pretrained-inference-results/inference_rcnn_final.png",dpi=300, bbox_inches="tight")

plt.show()
