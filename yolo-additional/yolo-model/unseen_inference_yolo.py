from ultralytics import YOLO
import pandas as pd

onnx_model = YOLO("/home/mrajaraman/master-thesis-dragonfly/yolo-model/runs/segment/train60/weights/best.onnx")
onnx_model.names.update({0: 'wings', 1: 'head', 2: 'thorax', 3: 'abdomen'})

# test_images = pd.read_csv("/home/mrajaraman/master-thesis-dragonfly/testing-batch.csv")

discarded_images = pd.read_csv("/home/mrajaraman/master-thesis-dragonfly/discarded.csv")

camo_images = pd.read_csv("/home/mrajaraman/master-thesis-dragonfly/camouflaged.csv")


for image_path in discarded_images['Path']:
    print("Evaluating image")
    results = onnx_model(image_path, save=True, project="unseen_inference_testing",name="discarded_batch",exist_ok=True)

for image_path in camo_images['Path']:
    print("Evaluating image")
    results = onnx_model(image_path, save=True, project="unseen_inference_testing",name="camo_batch",exist_ok=True)

print("Done")