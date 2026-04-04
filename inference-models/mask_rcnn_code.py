import wandb
import wandb
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch
import torch
import time
import pylab as pyl
import pandas as pd
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import datetime

from torchvision.models.detection.mask_rcnn import MaskRCNN
# from torchvision.models.detection import maskrcnn_resnet50_fpn
# from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
fasterrcnn_resnet50_fpn
# from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from torch.loss import CrossEntropyLoss
from PIL import Image
# num_epochs = 75
# num_epochs = 150


# run = wandb.init(
#     entity="universiteitleiden",
#     project="master-thesis-dragonfly",
#     tags=["mask-r-cnn", "raw-images", "unsegmented", "unannotated"],
#     config={
#         "learning_rate": 1e-4,
#         "architecture": "Mask R-CNN",
#         "epochs": 100,
#         "optimizer": "Adam"
#     }
# )


class Dragonfly(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def load_nifti(file_path):
        img = nib.load(file_path)        
        return img.get_fdata()

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['image_path']
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image file: {image_path}")
        
        img = self.load_nifti(image_path)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return img_tensor, image_path


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # pth_path = "/home/mrajaraman/Code/model_checkpoints/model_final_mask_rcnn.pth"
    # checkpoint = torch.load(pth_path, weights_only=False, map_location='cpu')
    
    # model = checkpoint.load_state_dict(checkpoint['model'])
    model = torch.load("/home/mrajaraman/Code/model_checkpoints/model_final_mask_rcnn.pth", weights_only=False, map_location=device)
    # model = torch.load("/home/mrajaraman/Code/model_checkpoints/model_final_mask_rcnn.pth",
    #                    weights = "detectron2://ImageNetPretrained/torchvision/R-50.pkl",
    #                    map_location=device)
    
    print("Model loaded successfully.")
    # model.eval()


    #model.eval()

    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_csv = "/home/mrajaraman/train.csv"
    val_csv = "/home/mrajaraman/val.csv"
    test_csv = "/home/mrajaraman/test.csv"

    train_csv = pd.read_csv(train_csv)
    val_csv = pd.read_csv(val_csv)

    dataset = Dragonfly(train_csv)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    val_dataset = Dragonfly(val_csv)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # criterion = nn.CrossEntropyLoss()

    # with torch.no_grad():
    #     for imgs, paths in dataloader:
    #         imgs = [img.to(device) for img in imgs]
    #         predictions = model(imgs)
    #         for img_path, pred in zip(paths, predictions):
    #             print(f"Results for {img_path}:")
    #             print("Boxes:", pred['boxes'])
    #             print("Labels:", pred['labels'])
    #             print("Scores:", pred['scores'])
    config_path = "/home/mrajaraman/do-not-modify/MassID45/Mask-RCNN/Mask-RCNN-MassID45/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py" 
    crop_fac = 16
    postprocess_match_threshold = 0.3

    # Define image shape and cropping dimensions
    scale_factor = 4
    im_shape_0 = 5464 * scale_factor
    im_shape_1 = 8192 * scale_factor 

    if (slice_height is None and slice_width is None) and crop_fac is not None:
        crop_rows = im_shape_0 // crop_fac
        crop_cols = im_shape_1 // crop_fac
    elif slice_height is not None and slice_width is not None:
        if super_resolution:
            crop_rows = slice_height * scale_factor
            crop_cols = slice_width * scale_factor
        else: 
            crop_rows = slice_height
            crop_cols = slice_width
    print(f"crop_rows: {crop_rows}")
    print(f"crop_cols: {crop_cols}")
    print(f"overlap: {overlap}")

    # Set device for model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Load detection model
    detection_model = model.to(device)

    # Perform prediction
    postprocess_match_threshold_id = str(postprocess_match_threshold).replace('.', '')
    model_conf_threshold = str(0.25).replace('.', '')
    model_id = "rcnn"
    exp_name = f"exp_{postprocess_match_threshold_id}_{model_conf_threshold}_conf_{model_id}"

    if os.path.exists(f"/h/jquinto/Mask-RCNN/runs/predict/{exp_name}"):
        exp_name = f"{exp_name}_current" 
    print(exp_name)

    if predict:
        predict(
            detection_model=model,
            model_confidence_threshold=model_confidence_threshold,
            model_device=device,
            model_category_mapping={"1": "b"},
            source=train_csv,
            no_standard_prediction=True,
            # image_size=1024,
            slice_height=crop_rows,
            slice_width=crop_cols,
            overlap_height_ratio=0.6,
            overlap_width_ratio=0.6,
            postprocess_type="GREEDYNMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=postprocess_match_threshold,
            postprocess_class_agnostic=True,
            export_pickle=False,
            export_crop=False, ## SET to FALSE to conserve space
            novisual=True, ## SET to TRUE to MAKE FASTER
            dataset_json_path=test_csv,
            visual_hide_labels=True,
            visual_hide_conf=True,
            name = exp_name,
        )

