import sys
sys.path.append("../code")

import argparse
import gc
import logging
import os
import segmentation_models_pytorch as smp
import time
import torch
import torch.nn as nn
import utils
import warnings
import yaml

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from dataset import ContrailsDataset, get_transform
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from utils import load_metadata

logger = get_logger(__name__, log_level="INFO")
warnings.filterwarnings("ignore") 


def create_segmentation_model(config):
    return smp.UnetPlusPlus(
        encoder_name=config.backbone, 
        encoder_weights=config.encoder_weights,
        encoder_depth=5,
        in_channels=3,
        classes=1
    )


def get_dice_score(prob, target, epsilon=1e-9):
    p0 = prob
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


def get_dice_loss(prob, target):
    return 1. - get_dice_score(prob, target)


def valid_step(model, batch):
    with torch.no_grad():
        x = batch["image"]
        y = batch["mask"]
        logits = model(x)
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()
        dice_loss = get_dice_loss(prob, y).mean()
        bce_loss = nn.BCEWithLogitsLoss()(logits, y.float())
        loss = 0.5 * dice_loss + 0.5 * bce_loss
        n = prob.size(0)
        tp = (pred * y).sum()
    return {
        "loss": loss * n,
        "bce_loss": bce_loss * n,
        "dice_loss": dice_loss * n,
        "n": n,
        "true_positives": tp,
        "cardinality": pred.sum() + y.sum()
    }


def valid_epoch(model, loader):
    model.eval()
    loss = 0.
    bce_loss = 0.
    dice_loss = 0.
    n = 0
    tp = 0.
    cardinality = 0.
    for batch in loader:
        outputs = valid_step(model, batch)
        loss += outputs["loss"]
        bce_loss += outputs["bce_loss"]
        dice_loss += outputs["dice_loss"]
        n += outputs["n"]
        tp += outputs["true_positives"]
        cardinality += outputs["cardinality"]
    return {
        "loss": loss / n,
        "bce_loss": bce_loss / n,
        "dice_loss": dice_loss / n,
        "global_dice_coef": 2. * tp / cardinality
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        required=True
    )
    args = parser.parse_args()
    return args


def train_segmentation_model(config_dict):
    config = utils.dotdict(config_dict)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
        log_with="wandb",
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if config.seed is not None:
        set_seed(config.seed)

    train_meta = load_metadata("train")
    valid_meta = load_metadata("validation")

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

    # Datasets and data loaders
    train_dataset = ContrailsDataset(train_meta, split="train")
    valid_dataset = ContrailsDataset(valid_meta, split="validation")

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
            config=config_dict
        )

    global_step = 0
    while True:
        model.train()
        train_loss = 0.0
        for train_batch in train_dataloader:
            with accelerator.accumulate(model):
                x = train_batch["image"]
                y = train_batch["mask"]

                logits = model(x)
                prob = torch.sigmoid(logits)
                dice_loss = get_dice_loss(prob, y).mean()
                bce_loss = nn.BCEWithLogitsLoss()(logits, y.float())
                loss = 0.5 * dice_loss + 0.5 * bce_loss
                loss = bce_loss

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

                if global_step > 0 and global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(config.checkpoint_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved accelerator state to {save_path}")

            if global_step % config.valid_delta_steps == 0:
                outputs = valid_epoch(model, valid_dataloader)
                accelerator.log({
                    "valid_loss": outputs["loss"],
                    "valid_bce_loss": outputs["bce_loss"],
                    "valid_dice_loss": outputs["dice_loss"],
                    "valid_global_dice_coef": outputs["global_dice_coef"]
                })
                model.train()

            if global_step >= config.max_train_steps * config.gradient_accumulation_steps:
                break
        else:
            continue
        break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    return


def get_wandb_run_name(backbone):
    return "__".join([
        f"backbone_{backbone}",
        f"timestamp_{round(time.time() * 1000)}"
    ])


def main():
    args = parse_args()
    with open(args.config_path, "rb") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config_dict["wandb_run_name"] = get_wandb_run_name(config_dict["backbone"])
    config_dict["checkpoint_dir"] = os.path.join(config_dict["output_dir"], config_dict["wandb_run_name"])
    train_segmentation_model(config_dict)
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()