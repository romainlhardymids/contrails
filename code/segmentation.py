import sys
sys.path.append("../../latent-diffusion")
sys.path.append("../../taming-transformers")
sys.path.append("../../mids-capstone-2023/code")

import albumentations as A
import argparse
import copy
import cv2
import gc
import logging
import numpy as np
import os
import pandas as pd
import scipy.io
import segmentation_models_pytorch as smp
import time
import timm
import torch
import torch.nn as nn
import torchvision.models as models
import utils
import warnings
import yaml

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from albumentations.pytorch import ToTensorV2
from datasets import SegmentationDataset, prepare_segmentation_data
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler


logger = get_logger(__name__, log_level="INFO")
warnings.filterwarnings("ignore") 


def create_segmentation_model(config):
    return smp.UnetPlusPlus(
        encoder_name=config.backbone, 
        encoder_weights="imagenet",
        in_channels=3,
        encoder_depth=config.num_encoding_blocks,
        decoder_channels=config.decoder_channels,
        decoder_attention_type="scse",
        classes=4
    )


def get_train_transform():
    return A.Compose([
        A.GaussianBlur((3, 7), p=0.25),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0, always_apply=True),
        ToTensorV2(always_apply=True)
    ])


def get_valid_transform():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0, always_apply=True),
        ToTensorV2(always_apply=True)
    ])
    

def prepare_dataloaders(train_dataset, valid_dataset, batch_size, num_workers=2):
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, valid_loader


def get_dice_score(output, target, epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1. - p0
    g1 = 1. - g0
    tp = (p0 * g0).sum(dim=(2, 3))
    fp = (p0 * g1).sum(dim=(2, 3))
    fn = (p1 * g0).sum(dim=(2, 3))
    num = 2. * tp
    denom = 2. * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score


def get_dice_loss(output, target):
    return 1. - get_dice_score(output, target)


def valid_step(accelerator, model, batch, criterion):
    with torch.no_grad():
        x = batch["image"]
        y = batch["mask"].type(torch.LongTensor).to(accelerator.device)
        logits = model(x)
        prob = nn.functional.softmax(logits, dim=1)
        targets = torch.permute(nn.functional.one_hot(y, num_classes=4), (0, 3, 1, 2))
        loss = criterion(prob, targets).mean()
        n = prob.size(0)
        loss = loss * n
    return {
        "loss": loss,
        "n": n
    }


def valid_epoch(accelerator, model, loader, criterion):
    model.eval()
    total_loss = 0.
    n = 0
    for batch in loader:
        outputs = valid_step(accelerator, model, batch, criterion)
        total_loss += outputs["loss"]
        n += outputs["n"]
    return {
        "loss": total_loss / n
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Steatosis classification training script.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        required=True,
        help="A path to a YAML config file to load arguments from.",
    )  
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training: local_rank"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def train_segmentation_model(config):
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with="wandb"
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if config.seed is not None:
        set_seed(config.seed)

    if config.prepare_segmentation_data:
        prepare_segmentation_data(config.raw_data_path, config.mask_dir, config.data_dir)

    df_labels = pd.read_csv(config.label_path)
    df_labels = df_labels.rename(columns={"patient_id": "patient"})
    df_labels["patient"] = df_labels["patient"] - 1 # Fix zero-indexing error

    df_raw = pd.read_csv(config.metadata_path)
    df = df_raw.merge(df_labels, on=["patient"], how="inner")
    df = df[df.dataset == "steatosis"]

    df_train = df.copy()
    df_valid = df.sample(n=100, replace=True)
    df_test  = df.sample(n=100, replace=True)

    # df_train = df[df.group == "train"].reset_index(drop=True)
    # df_valid = df[df.group == "val"].reset_index(drop=True)
    # df_test  = df[df.group == "test"].reset_index(drop=True)

    logging.info(f"Training images: {len(df_train)}")
    logging.info(f"Validation images: {len(df_valid)}")
    logging.info(f"Test images: {len(df_test)}")
    
    train_transform = get_train_transform()
    valid_transform = get_valid_transform()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model = create_segmentation_model(config)
    model.to(accelerator.device, dtype=weight_dtype)

    if config.scale_lr:
        config.learning_rate = config.learning_rate * \
            config.train_batch_size * \
            accelerator.num_processes

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    # Criterion
    criterion = get_dice_loss

    # Datasets and data loaders
    train_dataset = SegmentationDataset(df_train, train_transform)
    valid_dataset = SegmentationDataset(df_valid, valid_transform)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        sampler=None,
        shuffle=True,
        num_workers=config.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        shuffle=False, 
        batch_size=config.valid_batch_size,
        num_workers=config.num_workers
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
        num_cycles=config.num_cycles
    )

    # Prepare everything with the accelerator
    outputs = accelerator.prepare(
        model, 
        optimizer, 
        train_dataloader, 
        valid_dataloader, 
        lr_scheduler
    )
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = outputs

    # Initialize the trackers
    if accelerator.is_main_process:
        accelerator.init_trackers(
            config.wandb_project_name, 
            init_kwargs={"wandb":{"name":config.wandb_run_name}}, 
            config=vars(config)
        )

    global_step = 0
    best_valid_loss = float("inf")
    while True:
        model.train()
        train_loss = 0.0
        for train_batch in train_dataloader:
            with accelerator.accumulate(model):
                x = train_batch["image"]
                y = train_batch["mask"].type(torch.LongTensor).to(accelerator.device)

                logits = model(x)
                prob = nn.functional.softmax(logits, dim=1)
                targets = torch.permute(nn.functional.one_hot(y, num_classes=4), (0, 3, 1, 2))
                loss = criterion(prob, targets).mean()

                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({
                    "train_loss": train_loss,
                    "learning_rate": lr_scheduler.get_last_lr()[0]
                }, step=global_step)
                train_loss = 0.0

                if global_step in config.checkpointing_steps:
                    if accelerator.is_main_process:
                        save_path = os.path.join(config.checkpoint_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved accelerator state to {save_path}")

            if global_step % config.valid_delta_steps == 0:
                outputs = valid_epoch(accelerator, model, valid_dataloader, criterion)
                accelerator.log({"valid_loss": outputs["loss"]})
                if outputs["loss"] < best_valid_loss:
                    best_valid_loss = outputs["loss"]
                model.train()

            if global_step >= config.max_train_steps * config.gradient_accumulation_steps:
                break
        else:
            continue
        break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    return


def get_wandb_run_name(
    backbone,
    seed
):
    return "::".join([
        f"backbone_{backbone}",
        f"seed_{seed}"
    ])


def main():
    args = parse_args()
    with open(args.config_path, "rb") as f:
        config = utils.dotdict(yaml.load(f, Loader=yaml.FullLoader))
    config_ = copy.deepcopy(config)
    config_.wandb_run_name = get_wandb_run_name(config_.backbone, config_.seed)
    config_.checkpoint_dir = os.path.join(config_.output_dir, f"{config_.backbone}")
    train_segmentation_model(config_)
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()