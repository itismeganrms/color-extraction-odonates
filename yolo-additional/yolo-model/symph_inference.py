from ultralytics import YOLO
import pandas as pd
from PIL import Image

best_pt_model = YOLO("/home/mrajaraman/master-thesis-dragonfly/yolo-model/runs/segment/train64/weights/best.pt")
# onnx_model.names.update({0: 'wings', 1: 'head', 2: 'thorax', 3: 'abdomen'})

# test_images = pd.read_csv("/home/mrajaraman/master-thesis-dragonfly/testing-batch.csv")

symph = pd.read_csv("/home/mrajaraman/master-thesis-dragonfly/symph.csv")

for image_path in symph['Path']:
    print("Evaluating image")
    try:
        Image.open(image_path).verify()
    except Exception:
        print(f"Corrupted image file: {image_path}")
        continue

    results = best_pt_model(source=str(image_path), save=True, project="symph_inference",name="symph_batch",exist_ok=True)

print("Done")