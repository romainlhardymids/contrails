import sys
sys.path.append("../code")

import argparse
import gc
import logging
import numpy as np
import pytorch_lightning as pl
import random
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import wandb
import warnings
import yaml

from dataset import ContrailsPretrainingDataset
from model import Unet
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    GradientAccumulationScheduler,
    ModelCheckpoint, 
    StochasticWeightAveraging
)
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import average_precision, dice
from torch.utils.data import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from utils import data_split, load_synthetic_metadata, FOLDS

warnings.filterwarnings("ignore") 


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_segmentation_model(config):
    # return smp.Unet(
    #     encoder_name=config["encoder"], 
    #     encoder_weights="imagenet",
    #     encoder_depth=5,
    #     decoder_use_batchnorm=True,
    #     decoder_channels=config["decoder_channels"],
    #     decoder_attention_type=config["decoder_attention"],
    #     in_channels=3,
    #     classes=1
    # )
    return Unet(
        encoder_name=config["encoder"],
        decoder_use_batchnorm=True,
        decoder_channels=config["decoder_channels"],
        decoder_attention_type=config["decoder_attention"],
        classes=1,
        activation=None,
        aux_params=None,
        timesteps=1
    )


class SegmentationModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = create_segmentation_model(config)
        # self.criterion = smp.losses.DiceLoss(mode="binary", smooth=config["label_smoothing"])
        self.criterion = smp.losses.SoftBCEWithLogitsLoss(pos_weight=torch.tensor([4.0]), smooth_factor=config["label_smoothing"])

    def forward(self, batch):
        x = batch["image"]
        logits = self.model(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["optimizer"]["learning_rate"],
            betas=(self.config["optimizer"]["adam_beta1"], self.config["optimizer"]["adam_beta2"]),
            weight_decay=self.config["optimizer"]["adam_weight_decay"],
            eps=self.config["optimizer"]["adam_epsilon"],
        )
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            **self.config["scheduler"]["params"],
        )
        lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        logits = self.model(x)
        if self.config["image_size"] != 256:
            logits = torch.nn.functional.interpolate(logits, size=256, mode="bilinear")
        loss = self.criterion(logits, y)
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("lr", lr, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        logits = self.model(x)
        if self.config["image_size"] != 256:
            logits = torch.nn.functional.interpolate(logits, size=256, mode="bilinear")
        loss = self.criterion(logits, y)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, sync_dist=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/segmentation/config.yaml", required=True)
    args = parser.parse_args()
    return args


def train_segmentation_model(fold, config):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if config["seed"] is not None:
        set_seed(config["seed"])

    # Datasets and data loaders
    df = data_split("../data/data_split.csv")
    df_real_train = df[(df.fold != fold) & (df.split != "validation")]
    df_real_valid = df[df.fold == fold]

    df_synthetic_train = None
    df_synthetic_valid = None
    num_synthetic = config["model"]["num_synthetic"]
    if num_synthetic > 0:
        df_synthetic_train = load_synthetic_metadata()
        df_synthetic_train = df_synthetic_train[df_synthetic_train.fold == fold]
        df_synthetic_train = df_synthetic_train.sample(n=num_synthetic, replace=False).reset_index(drop=True)

    train_dataset = ContrailsPretrainingDataset(
        df_real_train, 
        df_synthetic_train,
        config["pseudo_labels_dir"],
        timesteps=8, 
        image_size=config["model"]["image_size"], 
        split="train"
    )
    valid_dataset = ContrailsPretrainingDataset(
        df_real_valid, 
        df_synthetic_valid,
        config["pseudo_labels_dir"],
        timesteps=8, 
        image_size=config["model"]["image_size"], 
        split="validation"
    )
    
    logging.info(f"[FOLD {fold}]")
    logging.info(f"Training images: {len(train_dataset)}")
    logging.info(f"Validation images: {len(valid_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["train_batch_size"], 
        sampler=None,
        shuffle=True,
        num_workers=config["num_workers"]
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        shuffle=False, 
        batch_size=config["valid_batch_size"],
        num_workers=config["num_workers"]
    )

    name = f"pretraining__backbone_{config['model']['encoder']}__fold_{fold}"
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        monitor="valid_loss",
        dirpath=config["output_dir"],
        mode="min",
        filename=name,
        save_top_k=1,
        verbose=1,
    )

    early_stopping_callback = EarlyStopping(monitor="valid_loss", **config["early_stopping"])
    accumulate_callback = GradientAccumulationScheduler(**config["accumulate"])
    swa_callback = StochasticWeightAveraging(**config["swa"])

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback, accumulate_callback, swa_callback],
        logger=WandbLogger(name=name, project=config["wandb_project"], save_dir=f"../logs"),
        **config["trainer"],
    )

    model = SegmentationModule(config["model"])

    trainer.fit(model, train_dataloader, valid_dataloader)

    wandb.finish()
    del trainer, model
    gc.collect()


def main():
    torch.set_float32_matmul_precision("medium")
    args = parse_args()
    with open(args.config, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for fold in range(FOLDS):
        train_segmentation_model(fold, config)
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()