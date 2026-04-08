import os
import numpy as np
import cv2


# input_dir = '/home/mrajaraman/dataset/annotated-tif/train'
# output_dir = '/home/mrajaraman/dataset/annotated-tif/train-labels'

input_dir = '/home/mrajaraman/dataset/annotated-tif/test'
output_dir = '/home/mrajaraman/dataset/annotated-tif/test-labels'

for j in os.listdir(input_dir):
    image_path = os.path.join(input_dir, j)
    # load the binary mask and get its contours
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # grayscale: classes are different pixel values

    H, W = mask.shape
    new_j = j[2:]
    # Define mapping from mask values to YOLO IDs
    label_map = {
        21: 0,   # wings
        60: 1,   # head
        113: 2,  # torso
        125: 3   # tail
    }

    with open('{}.txt'.format(os.path.join(output_dir, new_j)[:-4]), 'w') as f:
        for cls in np.unique(mask):
            if cls == 0 or cls == 255:
                continue  # skip background / ignore label

            yolo_cls = label_map[cls]  # map original mask value to YOLO ID

            # Binary mask for this class
            class_mask = np.uint8(mask == cls) * 255
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    polygon = []
                    for point in cnt:
                        x, y = point[0]
                        polygon.append(x / W)
                        polygon.append(y / H)

                    # Write polygon line: YOLO class ID first
                    line = [str(yolo_cls)] + [str(p) for p in polygon]
                    f.write(" ".join(line) + "\n")