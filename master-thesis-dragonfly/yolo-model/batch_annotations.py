from ultralytics import YOLO
import pandas as pd
import cv2
import numpy as np
import os

onnx_model = YOLO("/home/mrajaraman/master-thesis-dragonfly/yolo-model/runs/segment/train60/weights/best.onnx")

test_images = pd.read_csv("/home/mrajaraman/master-thesis-dragonfly/new_batch.csv")
mask_output_path = "/home/mrajaraman/master-thesis-dragonfly/yolo-model/batch_annotations/"

for image_path in test_images['Path']:
    print("Evaluating image")
    results = onnx_model(image_path, save=True, project="batch_annotations",name="first_batch",exist_ok=True)
    r = results[0]
    for i, obj in enumerate(r.boxes.data.tolist()):
        boxes = r.boxes
        cls = int(boxes.cls[i])
        classes = boxes.cls.cpu().tolist()
        masks = r.masks
        h, w = r.orig_shape
        text_file = image_path.split("/")[-1].split(".")[0] + ".txt"
        output_path = os.path.join(mask_output_path, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")

        with open(output_path, "w") as f: 
            if masks:
                # for j, mask in enumerate(masks.data):
                #     mask_np = mask.cpu().numpy().astype(np.uint8)
                #     cls = int(classes[j])
                #     mask_resized = cv2.resize(mask_np, (w,h), interpolation=cv2.INTER_NEAREST)
                #     # mask_resized_np = mask_resized.astype(np.uint8) * 255
                #     mask_resized_np = mask_resized.astype(np.uint8) * 255
                #     contours, _ = cv2.findContours(mask_resized_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #     for cnt in contours:
                #         # poly = cnt.squeeze().tolist()  # fewer points
                #         polygon = []                    
                #         for point in cnt:
                #             print("Point:", point)
                #             x, y = point[0]
                #             polygon.append(x / w)
                #             polygon.append(y / h)
                            
                #         line = [str(cls)] + [str(p) for p in polygon]
                #         f.write(" ".join(line) + "\n")
                for j, polygon in enumerate(r.masks.xy):
                    cls = int(classes[j])
                    norm_poly = [coord / w if i % 2 == 0 else coord / h for i, coord in enumerate(polygon.flatten())]
                    f.write(f"{cls} " + " ".join(map(str, norm_poly)) + "\n")
            else:
                print("No masks detected")
print("Done")