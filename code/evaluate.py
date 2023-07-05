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

from dataset import ContrailsDataset
from segmentation import SegmentationModule
from torchmetrics import AveragePrecision, Dice
from torchmetrics.functional import dice
from torch.utils.data import DataLoader
from utils import data_split, FOLDS, N_TIMES_BEFORE


def load_model(config, checkpoint_path):
    """Loads a model from a checkpoint path."""
    model = SegmentationModule(config)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/config.yaml", required=True)
    args = parser.parse_args()
    return args


def process(probs, threshold, min_size=10):
    "Post-processing logic."
    processed = []
    for pp in probs[:, 0, :]:
        n, img, stats, _ = cv2.connectedComponentsWithStats((pp > threshold).astype(np.uint8))
        sizes = stats[:, -1]
        sizes = sizes[1:]
        n -= 1
        output = np.zeros_like(pp)
        for i in range(n):
            if sizes[i] >= min_size:
                index = img == i + 1
                output[index] = pp[index]
        processed.append(output[None, :])
    processed = torch.from_numpy(np.stack(processed, axis=0))
    return processed


def plot_global_dice(scores, thresholds, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(24, 12))
    ax.plot(thresholds, scores)
    ax.set_xticks([i / 10. for i in range(11)])
    ax.set_yticks([i / 10. for i in range(11)])
    ax.grid(which="both")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Global DICE Coefficient")
    ax.set_title("Global DICE Coefficient vs. Decision Threshold")
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)


def evaluate(config):
    """Evaluates a set of models."""
    df = data_split("../data/data_split.csv")

    device = config["evaluation"]["device"]
    bins = config["evaluation"]["bins"]
    save_predictions = config["evaluation"]["save_predictions"]
    use_validation_split = config["evaluation"]["use_validation_split"]

    thresholds = [i / bins for i in range(1, bins)]
    global_dice = [Dice(threshold=t).to(device) for t in thresholds]
    auprc = AveragePrecision(task="binary").to(device)

    rows = []
    pred_dir = os.path.join(config["evaluation"]["output_dir"], "predictions")
    if os.path.exists(pred_dir):
        os.system(f"rm -rf {pred_dir}")
    os.mkdir(pred_dir)

    for fold in range(1):
        if use_validation_split:
            df_valid = df[df.split == "validation"]
        else:
            df_valid = df[df.fold == fold]
        valid_dataset = ContrailsDataset(df_valid, timesteps=1, image_size=config["model"]["image_size"], split="validation")
        print(f"Fold {fold} images: {len(valid_dataset)}")

        valid_dataloader = DataLoader(
            valid_dataset, 
            shuffle=False, 
            batch_size=config["valid_batch_size"],
            num_workers=config["num_workers"]
        )

        checkpoint_path = f"{config['evaluation']['checkpoint']}__fold_{fold}.ckpt"
        model = load_model(config["model"], checkpoint_path)
        model.to(device)
        model.eval()

        with torch.no_grad():
            for i, batch in enumerate(valid_dataloader):
                x = batch["image"].to(device)
                bsz, t, c, h, w = x.size()
                x = x.view(-1, c, h, w)
                y = batch["mask"].to(device)
                logits = model({"image": x})
                if config["model"]["image_size"] != 256:
                    logits = torch.nn.functional.interpolate(logits, size=256, mode="bilinear")
                    h, w = 256, 256
                probs = torch.sigmoid(logits)
                probs = probs.view(bsz, t, 1, h, w) # Output predictions have 1 channel
                for gd in global_dice:
                    gd.update(probs[:, 0, :], y)
                auprc.update(probs[:, 0, :], y)
                if save_predictions:
                    for j, (pp, yy) in enumerate(zip(probs, y)):
                        local_dice = dice(pp[0], yy)
                        record_id = df_valid.iloc[i * config["valid_batch_size"] + j]["record_id"]
                        save_path = os.path.join(pred_dir, f"{record_id}.npy")
                        np.save(save_path, pp.cpu().numpy())
                        rows.append({
                            "record_id": record_id,
                            "path": save_path,
                            "local_dice": local_dice.cpu().numpy()
                        })

        if use_validation_split: # Stop the loop if we are using the validation folder
            break

    scores = [gd.compute().cpu().numpy() for gd in global_dice]
    idx = np.argmax(scores)
    print(f"Best DICE coefficient (t = {thresholds[idx]}): {scores[idx]:.04f}")
    print(f"AUPRC: {auprc.compute():.04f}")

    if save_predictions:
        df_meta = pd.DataFrame(rows)
        df_meta.to_csv(os.path.join(config["evaluation"]["output_dir"], "metadata.csv"), index=None)
        plot_global_dice(scores, thresholds, os.path.join(config["evaluation"]["output_dir"], f"plots/{config['model']['encoder']}.png"))


def main():
    args = parse_args()
    with open(args.config, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    evaluate(config)
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()