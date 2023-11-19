import argparse
import data
import gc
import logging
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import wandb
import warnings
import yaml

from data.dataset import ContrailsDataset
from segmentation.model import create_model
from scripts.utils import set_seed
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

warnings.filterwarnings("ignore") 


class SegmentationModule2d(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = create_model(config["model"])
        self.logit_tracker = []
        self.label_tracker = []

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
        dice_loss = smp.losses.DiceLoss("binary", from_logits=True, smooth=self.config["losses"]["dice"]["label_smoothing"])
        return bce_loss, dice_loss

    def training_step(self, batch, batch_idx):
        x = batch["frames"]
        y = batch["mask"].float()
        logits = self.model(x)
        if logits.shape[-1] != 256:
            logits = F.interpolate(logits, size=256, mode="bilinear")
        bce_loss, dice_loss = self.compute_losses(logits, y)
        loss = (bce_loss + dice_loss) / 2.
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_bce_loss", bce_loss, on_step=True, on_epoch=True)
        self.log("train_dice_loss", dice_loss, on_step=True, on_epoch=True)
        self.log("lr", lr, on_step=True, on_epoch=False)
        return bce_loss

    def validation_step(self, batch, batch_idx):
        x = batch["frames"]
        y = batch["mask"].float()
        logits = self.model(x)
        if logits.shape[-1] != 256:
            logits = F.interpolate(logits, size=256, mode="bilinear")
        bce_loss, dice_loss = self.compute_losses(logits, y)
        loss = (bce_loss + dice_loss) / 2.
        self.logit_tracker.append(logits)
        self.label_tracker.append(y)
        self.log("valid_loss", loss, on_step=False, on_epoch=True)
        self.log("train_bce_loss", bce_loss, on_step=False, on_epoch=True)
        self.log("train_dice_loss", dice_loss, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        logits = torch.cat(self.logit_tracker)
        labels = torch.cat(self.label_tracker)
        valid_auprc = average_precision(logits, labels.long(), task="binary")
        valid_dice  = dice(logits, labels.long())
        self.log("valid_dice", valid_dice, on_step=False, on_epoch=True)
        self.log("valid_auprc", valid_auprc, on_step=False, on_epoch=True)
        self.logit_tracker.clear()
        self.label_tracker.clear()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/segmentation/config.yaml", required=True)
    args = parser.parse_args()
    return args


def train(fold, config):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if config["seed"] is not None:
        set_seed(config["seed"])

    df = data.utils.data_split("../data/data_split_dedup.csv")
    df_train = df[(df.fold != fold) & (df.split != "validation")]
    df_valid = df[df.fold == fold]
    train_dataset = ContrailsDataset(
        df_train,
        timesteps=config["data"]["timesteps"],
        image_size=config["data"]["image_size"],
        add_temp_diff=config["data"]["add_temp_diff"],
        split="train",
        **config["data"]["cutmix"]
    )
    valid_dataset = ContrailsDataset(
        df=df_valid, 
        timesteps=config["data"]["timesteps"],
        image_size=config["data"]["image_size"],
        add_temp_diff=config["data"]["add_temp_diff"],
        split="validation",
        cutmix_prob=0.,
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

    name = f"{config['model']['model']['encoder_name'].replace('/', '_')}_{config['data']['image_size']}_s{config['seed']}_f{fold}"
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

    model = SegmentationModule2d(config["model"])
    trainer.fit(model, train_dataloader, valid_dataloader)

    wandb.finish()
    del trainer, model
    gc.collect()


def main():
    torch.set_float32_matmul_precision("medium")
    args = parse_args()
    with open(args.config, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for fold in range(data.utils.FOLDS):
        train(fold, config)
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()