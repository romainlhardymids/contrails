import sys
sys.path.append("../../../mids-2023/latent-diffusion")
sys.path.append("../../../mids-2023/taming-transformers")
sys.path.append("../code")

import argparse
import gc
import logging
import numpy as np
import pytorch_lightning as pl
import random
import torch
import wandb
import yaml

from dataset import SemanticSynthesisDataset
from einops import rearrange
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping, 
    GradientAccumulationScheduler,
    ModelCheckpoint, 
    StochasticWeightAveraging
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from utils import data_split, normalize_diffusion, FOLDS, SEG_LABELS


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

 
def inference(model, batch, inference_steps, eta, guidance_scale):
    seg = batch["segmentation"].float()
    bsz = seg.size(0)
    with torch.no_grad():
        seg = rearrange(seg, "b h w c -> b c h w")
        cond = model.get_learned_conditioning(seg)
        uncond = model.get_learned_conditioning(torch.zeros(*seg.shape).to(model.device))
        samples, _ = model.sample_log(
            cond=cond, 
            batch_size=bsz, 
            ddim=True,
            ddim_steps=inference_steps, 
            eta=eta,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uncond
        )
        samples = model.decode_first_stage(samples)
    return seg, samples

class LogImages(Callback):
    def on_validation_batch_end(
        self,
        trainer, 
        pl_module, 
        outputs, 
        batch, 
        batch_idx, 
        dataloader_idx=0
    ):
        if batch_idx == 0:
            conditions, samples = inference(
                pl_module.model,
                batch, 
                inference_steps=pl_module.config["inference_steps"], 
                eta=pl_module.config["eta"], 
                guidance_scale=pl_module.config["guidance_scale"]
            )
            original_images = [[
                wandb.Image(normalize_diffusion(img, False), caption=f"original")
            ] for img in batch["image"]]
            condition_images = [[
                wandb.Image((c.permute(1, 2, 0).to("cpu").numpy() * np.array([i * 255 // SEG_LABELS for i in range(SEG_LABELS)])[None, None, :]).sum(axis=-1), caption=f"condition")
            ] for c in conditions]
            sample_images = [[
                wandb.Image(normalize_diffusion(s), caption=f"sample")
            ] for s in samples]
            images = sum([x + y + z for x, y, z in zip(original_images, condition_images, sample_images)], [])
            pl_module.logger.experiment.log({"image": images})


class DiffusionModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self.instantiate_model()

    def instantiate_model(self):
        arch_config = OmegaConf.load(self.config["arch_config"])
        model = instantiate_from_config(arch_config.model)
        return model

    def get_trainable_params(self):
        params = list(self.model.model.parameters())
        if self.config["train_cond_stage"]:
            params = params + list(self.model.cond_stage_model.parameters())
        if self.config["train_logvar"]:
            params.append(self.model.logvar)
        return params

    def configure_optimizers(self):
        params = self.get_trainable_params()
        optimizer = torch.optim.AdamW(
            params,
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
        loss, _ = self.model.shared_step(batch)
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.model.shared_step(batch)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, sync_dist=True)


def train_diffusion_model(fold, config):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if config["seed"] is not None:
        set_seed(config["seed"])

    # Datasets and data loaders
    df = data_split("../data/data_split.csv")
    df_train = df[(df.fold != fold) & (df.split != "validation")]
    df_valid = df[df.fold == fold]
    train_dataset = SemanticSynthesisDataset(df_train, cond_drop_rate=config["model"]["cond_drop_rate"], split="train")
    valid_dataset = SemanticSynthesisDataset(df_valid, cond_drop_rate=config["model"]["cond_drop_rate"], split="validation")

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

    name = f"semantic_synthesis__fold_{fold}"
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
    log_images = LogImages()

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback, accumulate_callback, swa_callback, log_images],
        logger=WandbLogger(name=name, project=config["wandb_project"], save_dir=f"../logs"),
        **config["trainer"],
    )

    model = DiffusionModule(config["model"])

    trainer.fit(model, train_dataloader, valid_dataloader)

    wandb.finish()
    del trainer, model
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/diffusion/config.yaml", required=True)
    args = parser.parse_args()
    return args


def main():
    torch.set_float32_matmul_precision("medium")
    args = parse_args()
    with open(args.config, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for fold in range(1, 2):
        train_diffusion_model(fold, config)
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()