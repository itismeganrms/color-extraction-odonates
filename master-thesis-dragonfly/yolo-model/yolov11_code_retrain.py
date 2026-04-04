import ultralytics
import wandb

from ultralytics import YOLO

run = wandb.init(
    entity="universiteitleiden",
    project="master-thesis-dragonfly",
    tags=["YOLO", "annotated-images", "4-part-annotated","val-true","retrain"],
    config={
        "learning_rate": 0.0001,
        "architecture": "YOLOv11x-seg",
        "epochs": 150,
        "optimizer": "NAdam",
        "dropout":0.4,
        "mask_ratio":5
    }
)

# Load a pretrained YOLO11n model
model = YOLO("yolo11x-seg.pt")

# train_results = model.train(
#     data="/home/mrajaraman/dataset/annotated-yolo/data/model_status.yaml",  # Path to dataset configuration file
#     epochs=run.config["epochs"],
#     optimizer=run.config["optimizer"],
#     lr0=run.config["learning_rate"],
#     mask_ratio = run.config["mask_ratio"],
#     dropout=run.config["dropout"],
#     val=True
# )

train_results = model.train(
    data="/home/mrajaraman/dataset/dataset-v2-yolo/data.yaml",  # Path to dataset configuration file
    epochs=run.config["epochs"],
    optimizer=run.config["optimizer"],
    lr0=run.config["learning_rate"],
    mask_ratio = run.config["mask_ratio"],
    dropout=run.config["dropout"],
    val=True
)


metrics = model.val(data="/home/mrajaraman/dataset/dataset-v2-yolo/data.yaml",verbose=True)

results = model(
    "/home/mrajaraman/dataset/first_batch_pngs/data/images/test/img_1458467690.png",
    save=True,
    project="predictions",   # folder name
    name="dragonfly_test_ne"    # subfolder name
)

path = model.export(format="onnx")  # Returns the path to the exported model