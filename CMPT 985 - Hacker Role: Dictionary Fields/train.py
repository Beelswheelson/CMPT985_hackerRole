import os
import numpy as np
import torch.autograd
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
import dataloaders, network


def main():
    config = {
        "batch_size": 1,
        "learning_rate": 0.0002,
        "beta1": 0.5,
        "beta2": 0.999,
        "hidden_layer_sizes": [64, 128],
        "activation": "relu",
        "num_epochs": 100,
        "num_workers": 8,
        "num_volumes": 14,
    }
    wandb.init(project="Hacker Role- Dictionary Fields",
               notes="",
               tags=["Dictionary Fields", "CMPT 985"],
               config=config)

    ckpt_pth = None

    datamodule = dataloaders.SliceDataModule()

    model = network.DictionaryField_2DRegression()

    data = datamodule.train_dataloader()
    wandb_logger = WandbLogger()
    trainer = Trainer(logger=wandb_logger, max_epochs=10000)
    trainer.fit(model=model, train_dataloaders=datamodule)


if __name__ == '__main__':
    main()