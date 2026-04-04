import ultralytics
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("yolo11x-seg.pt")

results = model("/home/mrajaraman/dataset/originals/img_931176390.jpg")

result = results[0]
rendered_img = result.plot() 
plt.figure(figsize=(8, 8))
plt.imshow(rendered_img)
plt.title('Inference of YOLO11x-seg on Image', fontsize=16)
plt.axis("off")
plt.savefig("/home/mrajaraman/master-thesis-dragonfly/inference-models/pretrained-inference-results/inference_yolo.png",dpi=300, bbox_inches="tight")
plt.close()