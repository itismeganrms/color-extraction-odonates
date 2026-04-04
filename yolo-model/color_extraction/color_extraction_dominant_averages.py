import pandas as pd
import cv2
import numpy as np

from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image

from cv2 import THRESH_BINARY
from cv2 import imread, imwrite, resize, cvtColor, threshold
# from sklearn.cluster import KMeans
from collections import Counter

best_model = YOLO("/home/mrajaraman/master-thesis-dragonfly/yolo-model/runs/segment/train64/weights/best.pt")    
symph_df = pd.read_csv("/home/mrajaraman/master-thesis-dragonfly/geo_statistics/netherlands_symphetrum_striolatum_occurrences_full.csv")
test_images = pd.read_csv("/home/mrajaraman/master-thesis-dragonfly/symph.csv")

dataframe_results = pd.DataFrame(columns=['ID','RGB_Abdomen','RGB_Thorax','RGB_Head','HSV_Abdomen','HSV_Thorax','HSV_Head'])

for test_image_path in test_images['Path']:
    print("Evaluating image" + test_image_path)
    try:
        Image.open(test_image_path).verify()
    except Exception:
        print(f"Corrupted image file: {test_image_path}")
        continue
    
    results_improved = best_model(source=str(test_image_path), classes=[0,1,2], save=True)
    image_id = test_image_path.split("/")[-1].split(".")[0]

    r = results_improved[0]
    boxes = r.boxes
    classes = boxes.cls.cpu().tolist()
    masks = r.masks

    class_names = r.names
    class_ids = r.boxes.cls.cpu().numpy().astype(int)
    counter = Counter(class_ids)
    output = [(class_names[k], v) for k, v in counter.items()]
    dataframe_results.loc[image_id] = [None] * len(dataframe_results.columns)

    # dataframe_results = pd.DataFrame(columns=['ID','RGB_Abdomen','RGB_Thorax','RGB_Head','HSV_Abdomen','HSV_Thorax','HSV_Head'])
    dataframe_results.loc[image_id] = [None] * len(dataframe_results.columns)

    test_image = cv2.imread(test_image_path)
    img = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    for item_name, item_count in output:
        if item_name == 'tail' and item_count == 1:
            corresponding_index = [key for key, value in class_names.items() if value == item_name][0]
            tail_class_id = [k for k, v in class_names.items() if v == 'tail'][0]
            tail_indices = [i for i, c in enumerate(r.boxes.cls) if int(c) == tail_class_id]
            det_idx = tail_indices[0]
            corresponding_mask = r.masks.data[det_idx]

            part_mask = corresponding_mask.cpu().numpy()
            part_resized = cv2.resize(part_mask, (test_image.shape[1], test_image.shape[0])) #, interpolation=cv2.INTER_NEAREST)
            mask_uint = (part_resized * 255).astype(np.uint8)
            _, mask_binary = cv2.threshold(mask_uint, 127, 255, cv2.THRESH_BINARY)
            resultant_part = cv2.bitwise_and(img, img, mask=mask_binary)
            mask_colored = np.any(resultant_part != 0, axis=2)
            colored_pixels = resultant_part[mask_colored]

            mean_rgb = colored_pixels.mean(axis=(0))
            mean_rgb_img = mean_rgb.reshape(1, 1, 3).astype(np.uint8)
            mean_rgb_value = mean_rgb_img.tolist()[0][0]
            dataframe_results.at[image_id,'RGB_Abdomen'] = tuple(mean_rgb_value)
            hsv_frame = cv2.cvtColor(mean_rgb_img, cv2.COLOR_RGB2HSV).tolist()[0][0]
            dataframe_results.at[image_id,'HSV_Abdomen'] = tuple(hsv_frame)

        elif item_name == 'torso' and item_count == 1:
            corresponding_index = [key for key, value in class_names.items() if value == item_name][0]
            torso_class_id = [k for k, v in class_names.items() if v == 'torso'][0]
            torso_index = [i for i, c in enumerate(r.boxes.cls) if int(c) == torso_class_id]
            det_idx = torso_index[0]
            corresponding_mask = r.masks.data[det_idx]

            part_mask = corresponding_mask.cpu().numpy()
            part_resized = cv2.resize(part_mask, (test_image.shape[1], test_image.shape[0])) #, interpolation=cv2.INTER_NEAREST)
            mask_uint = (part_resized * 255).astype(np.uint8)
            _, mask_binary = cv2.threshold(mask_uint, 127, 255, cv2.THRESH_BINARY)
            resultant_part = cv2.bitwise_and(img, img, mask=mask_binary)
            mask_colored = np.any(resultant_part != 0, axis=2)
            colored_pixels = resultant_part[mask_colored]

            mean_rgb = colored_pixels.mean(axis=(0))
            mean_rgb_img = mean_rgb.reshape(1, 1, 3).astype(np.uint8)
            mean_rgb_value = mean_rgb_img.tolist()[0][0]
            dataframe_results.at[image_id,'RGB_Thorax'] = tuple(mean_rgb_value)
            hsv_frame = cv2.cvtColor(mean_rgb_img, cv2.COLOR_RGB2HSV).tolist()[0][0]
            dataframe_results.at[image_id,'HSV_Thorax']  = tuple(hsv_frame)
        
        elif item_name == 'head' and item_count == 1:
            corresponding_index = [key for key, value in class_names.items() if value == item_name][0]
            head_class_id = [k for k, v in class_names.items() if v == 'head'][0]
            head_index = [i for i, c in enumerate(r.boxes.cls) if int(c) == head_class_id]
            det_idx = head_index[0]
            corresponding_mask = r.masks.data[det_idx]

            part_mask = corresponding_mask.cpu().numpy()
            part_resized = cv2.resize(part_mask, (test_image.shape[1], test_image.shape[0])) #, interpolation=cv2.INTER_NEAREST)
            mask_uint = (part_resized * 255).astype(np.uint8)
            _, mask_binary = cv2.threshold(mask_uint, 127, 255, cv2.THRESH_BINARY)
            resultant_part = cv2.bitwise_and(img, img, mask=mask_binary)
            mask_colored = np.any(resultant_part != 0, axis=2)
            colored_pixels = resultant_part[mask_colored]

            mean_rgb = colored_pixels.mean(axis=(0))
            mean_rgb_img = mean_rgb.reshape(1, 1, 3).astype(np.uint8)
            mean_rgb_value = mean_rgb_img.tolist()[0][0]
            dataframe_results.at[image_id,'RGB_Head'] = tuple(mean_rgb_value)
            hsv_frame = cv2.cvtColor(mean_rgb_img, cv2.COLOR_RGB2HSV).tolist()[0][0]
            dataframe_results.at[image_id,'HSV_Head']  = tuple(hsv_frame)
        else:
            print(f"Unexpected item: {image_id} with count {item_count}")

# dataframe_results.reset_index().to_csv("results.csv", index=False)
dataframe_results.reset_index().to_csv("/home/mrajaraman/master-thesis-dragonfly/yolo-model/color_extraction/color_extraction_means.csv", index=False)