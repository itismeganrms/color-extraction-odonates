from ultralytics import YOLO
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from cv2 import cvtColor, COLOR_BGR2RGB

onnx_model_initial = YOLO("/home/mrajaraman/master-thesis-dragonfly/yolo-model/runs/segment/train60/weights/best.onnx")
onnx_model_initial.names.update({0:'wings', 1:'head', 2:'thorax', 3:'abdomen'})

onnx_model_new = YOLO("/home/mrajaraman/master-thesis-dragonfly/yolo-model/runs/segment/train64/weights/best.onnx")
onnx_model_new.names.update({0: 'head', 1: 'abdomen', 2: 'thorax', 3: 'wings'})

test_image = "/home/mrajaraman/dataset/originals/img_1458477504.jpg"

print("Evaluating image with first model")
results_initial = onnx_model_initial(test_image, save=False, project="inference_generation",name="intial_test",exist_ok=False)
initial_image = results_initial[0]
image = initial_image.plot()
final_image = cvtColor(image,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
plt.imshow(final_image)
plt.title("Inference of trained YOLO on Image", fontsize=16)
plt.axis("off")
plt.savefig("/home/mrajaraman/master-thesis-dragonfly/yolo-model/inference_generation/intial_test/initial_image.png", dpi=300, bbox_inches="tight")
plt.close()

print("Evaluating image with second model")
result_final = onnx_model_new(test_image, save=False, project="inference_generation",name="final_test",exist_ok=False)
final_image = result_final[0]
image = final_image.plot()
final_image = cvtColor(image,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
plt.imshow(final_image)
plt.title("Inference of trained YOLO on Image", fontsize=16)
plt.axis("off")
plt.savefig("/home/mrajaraman/master-thesis-dragonfly/yolo-model/inference_generation/final_test/final_image.png", dpi=300, bbox_inches="tight")
plt.close()
print("Done")