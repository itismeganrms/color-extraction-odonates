import ultralytics
import wandb

from ultralytics import YOLO

run = wandb.init(
    entity="universiteitleiden",
    project="master-thesis-dragonfly",
    tags=["YOLO", "annotated-images", "4-part-annotated","val-true"],
    config={
        "learning_rate": 1e-4,
        "architecture": "YOLOv11x-seg",
        "epochs": 250,
        "optimizer": "Adam"
    }
)

# Load a pretrained YOLO11n model
model = YOLO("yolo11x-seg.pt")

# for epoch in range(run.config.epochs):
#     model.train(data="/home/mrajaraman/master-thesis-dragonfly/yolo-model/model_status.yaml", optimizer=run.config['optimizer'], lr0=run.config['learning_rate'])
#     run.log()

train_results = model.train(
    data="/home/mrajaraman/dataset/first_batch_pngs/data/model_status.yaml",  # Path to dataset configuration file
    epochs=run.config["epochs"],
    optimizer=run.config["optimizer"],
    lr0=run.config["learning_rate"],
    val=True
)

metrics = model.val(data="/home/mrajaraman/dataset/first_batch_pngs/data/model_status.yaml",verbose=True)

results = model(
    "/home/mrajaraman/dataset/first_batch_pngs/data/images/test/img_1458467690.png",
    save=True,
    project="predictions",   # folder name
    name="dragonfly_test"    # subfolder name
)


path = model.export(format="onnx")  # Returns the path to the exported model