import ultralytics
import wandb
import torch
from torch.optim import Adam, AdamW, SGD

from ultralytics import YOLO

sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "learning_rate": {"values": [1e-3, 1e-4, 1e-5, 1e-3, 1e-6]},
        "epochs": {"values": [100, 150, 200, 250, 300]},
        "optimizer": {"values": ["Adam", "SGD", "AdamW","NAdam", "RAdam", "RMSProp"]},
        "dropout":{"values":[0.0, 0.1, 0.2, 0.3, 0.4]},
        "mask_ratio":{"values":[1,2,3,4,5]}
    }
}

def main():
    model = YOLO("yolo11x-seg.pt")
    run = wandb.init()
    current_lr0 = run.config.learning_rate
    current_optimizer = run.config.optimizer
    current_epochs = run.config.epochs
    current_dropout = run.config.dropout
    current_mask_ratio = run.config.mask_ratio

    train_results = model.train(
        data="/home/mrajaraman/dataset/first_batch_pngs/data/model_status.yaml",  # Path to dataset configuration file
        epochs=current_epochs,
        optimizer=current_optimizer,
        lr0 = current_lr0,
        dropout = current_dropout,
        mask_ratio = current_mask_ratio, 
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

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="master-thesis-dragonfly", entity="universiteitleiden")
    wandb.agent(sweep_id, function=main, count=30)