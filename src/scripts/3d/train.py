import argparse
import data.utils as utils
import gc
import logging
import numpy as np
import pytorch_lightning as pl
import random
import torch
import torch.nn.functional as F
import wandb
import warnings
import yaml

from data.dataset import ContrailsDataset
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    GradientAccumulationScheduler,
    ModelCheckpoint, 
    StochasticWeightAveraging
)
from pytorch_lightning.loggers import WandbLogger
from segmentation.model import create_segmentation_model
from torchmetrics.functional import average_precision, dice
from torch.utils.data import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

warnings.filterwarnings("ignore") 


def set_seed(seed: int):
    """Fixes the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SegmentationModule3d(pl.LightningModule):
    """PyTorch Lightning module for training a 3D segmentation model."""
    def __init__(self, config):
        super(SegmentationModule3d, self).__init__()
        self.config = config
        self.model = create_segmentation_model(config["model"])
        self.seg_logit_tracker = []
        self.seg_label_tracker = []

    def configure_optimizers(self):
        lr = self.config["optimizer"]["lr"]
        params = list(self.named_parameters())
        param_groups = [
            {"params": [p for n, p in params if "encoder" in n], "lr": lr / 10.},
            {"params": [p for n, p in params if "encoder" not in n], "lr": lr}
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            **self.config["optimizer"]
        )
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            **self.config["scheduler"],
        )
        lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
    
    def compute_losses(self, logits, y):
        pos_weight = torch.tensor([self.config["losses"]["bce"]["pos_weight"]]).to(self.device)
        bce_loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="mean")
        return bce_loss

    def training_step(self, batch, batch_idx):
        x = batch["frames"]
        y = batch["mask"].float()
        logits = self.model(x)
        if self.config["data"]["image_size"] != 256:
            logits = F.interpolate(logits, size=256, mode="bilinear")
        bce_loss = self.compute_losses(logits, y)
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("train_loss", bce_loss, on_step=True, on_epoch=True)
        self.log("lr", lr, on_step=True, on_epoch=False)

        return bce_loss

    def validation_step(self, batch, batch_idx):
        x = batch["frames"]
        y = batch["mask"].float()
        logits = self.model(x)
        if self.config["data"]["image_size"] != 256:
            logits = F.interpolate(logits, size=256, mode="bilinear")
        bce_loss = self.compute_losses(logits, y)
        self.seg_logit_tracker.append(logits)
        self.seg_label_tracker.append(y)
        self.log("valid_loss", bce_loss, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        seg_logits = torch.cat(self.seg_logit_tracker)
        seg_labels = torch.cat(self.seg_label_tracker)
        valid_auprc = average_precision(seg_logits, seg_labels.long(), task="binary")
        valid_dice  = dice(seg_logits, seg_labels.long())
        self.log("valid_dice", valid_dice, on_step=False, on_epoch=True)
        self.log("valid_auprc", valid_auprc, on_step=False, on_epoch=True)
        self.seg_logit_tracker.clear()
        self.seg_label_tracker.clear()


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/segmentation/config.yaml", required=True)
    args = parser.parse_args()
    return args


def train(fold, config):
    """Trains a 3D segmentation model on a specified fold."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if config["seed"] is not None:
        set_seed(config["seed"])

    df = utils.data_split("../data/data_split.csv")
    df_train = df[(df.fold != fold) & (df.split != "validation")]
    df_valid = df[df.fold == fold]
    train_dataset = ContrailsDataset(
        timesteps=config["model"]["data"]["timesteps"],
        df=df_train, 
        image_size=config["model"]["data"]["image_size"], 
        cutmix_prob=config["model"]["data"]["cutmix"]["prob"],
        cutmix_num_holes=config["model"]["data"]["cutmix"]["num_holes"],
        cutmix_min_size=config["model"]["data"]["cutmix"]["min_size"],
        cutmix_max_size=config["model"]["data"]["cutmix"]["max_size"],
        split="train"
    )
    valid_dataset = ContrailsDataset(
        timesteps=config["model"]["data"]["timesteps"],
        df=df_valid, 
        image_size=config["model"]["data"]["image_size"], 
        cutmix_prob=0.,
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

    name = f"finetuning__family_3d__backbone_{config['model']['model']['encoder_name'].replace('/', '_')}__fold_{fold}"
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        monitor="valid_auprc",
        dirpath=config["output_dir"],
        mode="max",
        filename=name,
        save_top_k=1,
        verbose=1,
    )

    early_stopping_callback = EarlyStopping(monitor="valid_auprc", **config["early_stopping"])
    accumulate_callback = GradientAccumulationScheduler(**config["accumulate"])
    swa_callback = StochasticWeightAveraging(**config["swa"])

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback, accumulate_callback, swa_callback],
        logger=WandbLogger(name=name, project=config["wandb_project"], save_dir=f"../logs"),
        **config["trainer"],
    )
    trainer.logger.log_hyperparams(config)

    model = SegmentationModule3d(config["model"])
    if config["model"]["checkpoint"] is not None:
        model.load_state_dict(torch.load(f"{config['model']['checkpoint']}__fold_{fold}.ckpt")["state_dict"])

    trainer.fit(model, train_dataloader, valid_dataloader)

    wandb.finish()
    del trainer, model
    gc.collect()


def main():
    torch.set_float32_matmul_precision("medium")
    args = parse_args()
    with open(args.config, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for fold in range(utils.FOLDS):
        train(fold, config)
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()