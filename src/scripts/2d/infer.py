import sys
sys.path.append("../code")

import argparse
import data
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import yaml

from data.dataset import ContrailsDataset
from train import SegmentationModule2d
from torch.utils.data import DataLoader

from tqdm.auto import tqdm


def load_model(module, config, checkpoint_path):
    """Loads a 2D segmentation model checkpoint."""
    model = module(config)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    return model


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/config.yaml", required=True)
    args = parser.parse_args()
    return args


def infer(config):
    """Performs inference on held-out data."""
    df = data.utils.data_split("../data/data_split.csv")
    print(f"Images: {df.shape[0]}")

    seg_models = config["seg_models"]
    device = config["device"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    threshold = config["threshold"]
    output_dir = config["pseudo_label_dir"]
    total_seg_weight = sum([seg_models[k]["weight"] * len(seg_models[k]["checkpoint_paths"]) for k in seg_models])

    predictions = {}
    for name, model in seg_models.items():
        weight = model["weight"]
        config_path = model["config_path"]
        checkpoint_paths = model["checkpoint_paths"]

        print(f"Loading segmentation model `{name}` with configuration: {config_path}")
        with open(config_path, "rb") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        for checkpoint_path in checkpoint_paths:
            print(f"Checkpoint: {checkpoint_path}")
            model = load_model(SegmentationModule2d, config["model"], checkpoint_path)
            model.to(device)
            model.eval()
            
            dataset = ContrailsDataset(
                df, 
                timesteps=[i for i in range(8)], 
                image_size=config["model"]["data"]["image_size"],
                use_pseudo_labels=False,
                cutmix_prob=0.,
                split="validation"
            )
            dataloader = DataLoader(
                dataset, 
                shuffle=False, 
                batch_size=batch_size // 8,
                num_workers=num_workers
            )

            with torch.no_grad():
                for i, batch in tqdm(enumerate(dataloader)):
                    x = batch["frames"].to(device)
                    n, c, t, h, w = x.shape
                    x = torch.permute(x, (0, 2, 1, 3, 4)).contiguous()
                    x = x.view(-1, c, h, w)
                    logits = model.model(x)
                    if logits.shape[-1] != 256:
                        logits = torch.nn.functional.interpolate(logits, size=256, mode="bilinear")
                    probs = torch.sigmoid(logits).view(n, t, 1, 256, 256).cpu().numpy()
                    for j, pp in enumerate(probs):
                        row = df.iloc[i * batch_size // 8 + j]
                        record_id = row["record_id"]
                        predictions.setdefault(record_id, np.zeros(pp.shape, dtype=np.float32))
                        predictions[record_id] += pp * weight / total_seg_weight

            del model, dataset, dataloader
            torch.cuda.empty_cache()
            gc.collect()

    rows = []
    for record_id in predictions:
        preds = (predictions[record_id] > threshold).astype(np.uint8)
        save_path = os.path.join(output_dir, "masks", f"{record_id}.npy")
        np.save(save_path, preds)
        rows.append({
            "record_id": record_id,
            "mask_path": save_path
        })
    df_meta = pd.DataFrame(rows)
    df_meta.to_csv(os.path.join(output_dir, "metadata.csv"), index=None)
        

def main():
    args = parse_args()
    with open(args.config, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    infer(config)
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()