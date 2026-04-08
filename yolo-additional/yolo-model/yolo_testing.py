import wandb
import ultralytics

from ultralytics import YOLO

run = wandb.init(
    entity="universiteitleiden",
    project="master-thesis-dragonfly",
    config={"architecture": "YOLOv11x-seg"},
    tags=["YOLO","val-true","testing"]
)

# Load a model
model = YOLO("/home/mrajaraman/master-thesis-dragonfly/yolo-model/runs/segment/train60/weights/best.pt")

# Customize validation settings
metrics = model.val(data="/home/mrajaraman/dataset/first_batch_pngs/data/model_status.yaml") #, imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")

summary = metrics.summary()

for i in range(len(summary)):
    summary_value = summary[i]
    # print(summary_value)
    # print(type(summary_value))
    wandb.log(summary_value)

wandb.finish()