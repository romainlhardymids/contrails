import sys
sys.path.append("../code")

import argparse
import cv2
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torchmetrics
import yaml

from data.dataset import ContrailsDataset3d
from data.utils import data_split, FOLDS
from train import SegmentationModule3d
from torchmetrics import AveragePrecision, Dice
from torchmetrics.functional import dice
from torch.utils.data import DataLoader


def load_model(module, config, checkpoint_path):
    """Loads a 3D segmentation model checkpoint."""
    model = module(config)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    return model


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/config.yaml", required=True)
    args = parser.parse_args()
    return args


def evaluate(config):
    """Evaluates a set of models on held-out data."""
    df = data_split("../data/data_split.csv")

    device = config["evaluation"]["device"]
    bins = config["evaluation"]["bins"]
    use_validation_split = config["evaluation"]["use_validation_split"]
    thresholds = [i / bins for i in range(1, bins)]

    predictions = {}
    labels = {}
    for fold in range(FOLDS):
        global_dice = [Dice(threshold=t).to(device) for t in thresholds]
        auprc = AveragePrecision(task="binary").to(device)

        if use_validation_split:
            df_valid = df[df.split == "validation"]
        else:
            df_valid = df[df.fold == fold]
        valid_dataset = ContrailsDataset3d(
            timesteps=config["model"]["data"]["timesteps"],
            df=df_valid, 
            image_size=config["model"]["data"]["image_size"], 
            cutmix_prob=0.,
            split="validation"
        )
        print(f"Fold {fold} images: {len(valid_dataset)}")

        valid_dataloader = DataLoader(
            valid_dataset, 
            shuffle=False, 
            batch_size=config["valid_batch_size"],
            num_workers=config["num_workers"]
        )

        checkpoint_path = f"{config['evaluation']['checkpoint']}__fold_{fold}.ckpt"
        model = load_model(SegmentationModule3d, config["model"], checkpoint_path)
        model.to(device)
        model.eval()

        with torch.no_grad():
            for i, batch in enumerate(valid_dataloader):
                x = batch["frames"].to(device)
                y = batch["mask"].to(device)
                logits = model.model(x)
                if config["model"]["data"]["image_size"] != 256:
                    logits = torch.nn.functional.interpolate(logits, size=256, mode="bilinear")
                probs = torch.sigmoid(logits)
                probs = probs.view(probs.size(0), 1, 256, 256) # Output predictions have 1 channel
                for j, (pp, yy) in enumerate(zip(probs, y)):
                    record_id = df_valid.iloc[i * config["valid_batch_size"] + j]["record_id"]
                    if record_id not in predictions:
                        predictions[record_id] = [pp]
                        labels[record_id] = yy
                    else:
                        predictions[record_id].append(pp)
                
                for gd in global_dice:
                    gd.update(probs, y)
                auprc.update(probs, y)

        scores = [gd.compute().cpu().numpy() for gd in global_dice]
        idx = np.argmax(scores)
        print(f"Global DICE coefficient (t = {thresholds[idx]}): {scores[idx]:.04f}")
        print(f"AUPRC: {auprc.compute():.04f}\n")

        del model, global_dice, auprc
        torch.cuda.empty_cache()
        gc.collect()
        
    if use_validation_split:
        ensemble_dice = [Dice(threshold=t).to(device) for t in thresholds]
        for record_id in predictions:
            probs_list, y = predictions[record_id], labels[record_id]
            probs = torch.mean(torch.cat(probs_list, dim=0), dim=0)
            for ed in ensemble_dice:
                ed.update(probs, y)

        scores = [ed.compute().cpu().numpy() for ed in ensemble_dice]
        idx = np.argmax(scores)
        threshold, score = thresholds[idx], scores[idx]
        print(f"Ensemble global DICE coefficient (t = {threshold}): {score:.04f}\n")


def main():
    args = parse_args()
    with open(args.config, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    evaluate(config)
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()