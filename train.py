import argparse
import copy
import os
import random
import shutil
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

import wandb
from utils.comm import *


@hydra.main(config_path="configs", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    datamodule = hydra.utils.instantiate(cfg.dataset)
    metrics = hydra.utils.call(cfg.metrics)
    optimizer_factory = hydra.utils.instantiate(cfg.optimizer)
    scheduler_factory = hydra.utils.instantiate(cfg.scheduler)
    criterion = hydra.utils.instantiate(cfg.criterion)
    model = hydra.utils.instantiate(cfg.model, optimizer_factory=optimizer_factory,
                                    scheduler_factory=scheduler_factory, criterion=criterion, metrics=metrics)

    wandb.init(project="hgcal-spvcnn", config=cfg._content)
    wandb_logger = pl.loggers.WandbLogger(save_dir=os.getcwd())
    trainer = pl.Trainer(gpus=1, logger=wandb_logger, weights_save_path=os.getcwd())
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    hydra_main()  # pylint: disable=no-value-for-parameter
