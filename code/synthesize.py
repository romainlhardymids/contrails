import sys
sys.path.append("../../../mids-2023/latent-diffusion")
sys.path.append("../../../mids-2023/taming-transformers")
sys.path.append("../code")

import argparse
import cv2
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import torch
import yaml

from dataset import SemanticSynthesisDataset
from diffusion import DiffusionModule, inference
from einops import rearrange
from functools import reduce
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils import normalize_diffusion


def inference_batch(model, loader, inference_steps, eta, guidance_scale):
    conditions = []
    outputs = []
    for batch in loader:
        batch = {"segmentation": batch["segmentation"].to(model.device)}
        _, samples = inference(
            model, 
            batch, 
            inference_steps=inference_steps, 
            eta=eta,
            guidance_scale=guidance_scale
        )
        for i, s in enumerate(samples):
            outputs.append(normalize_diffusion(s))
            conditions.append(batch["segmentation"][i].cpu().numpy())
    outputs = np.stack(outputs, axis=0)
    conditions = np.stack(conditions, axis=0)
    return outputs, conditions


def save_outputs(inference_outputs, save_dir, append=True):
    """Saves synthetic images to an output directory."""
    if not append:
        os.system(f"rm -r {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "conditions"), exist_ok=True)
    outputs, conditions = inference_outputs
    rows = []
    for i, (output, condition) in enumerate(zip(outputs, conditions)):
        condition = cv2.resize(condition, (256, 256), interpolation=cv2.INTER_NEAREST)[..., -1]
        hash = random.getrandbits(128)
        image_path = os.path.join(save_dir, "images", f"{hash}.png")
        condition_path = os.path.join(save_dir, "conditions", f"{hash}.png")
        row = {"condition_path": condition_path, "image_path": image_path}
        rows.append(row)
        cv2.imwrite(image_path, output)
        cv2.imwrite(condition_path, condition)
    metadata_path = os.path.join(save_dir, "metadata.csv")
    try:
        df = pd.concat([
            pd.read_csv(metadata_path),
            pd.DataFrame(rows)
        ], axis=0)
    except:
        df = pd.DataFrame(rows)
    df.to_csv(metadata_path, index=False)


def synthesize_images(fold, config):
    checkpoint_path = os.path.join(config["output_dir"], f"semantic_synthesis__fold_{fold}.ckpt")
    model = DiffusionModule(config["model"])
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    m = model.model.to(config["synthesize"]["device"])
    m.eval()

    df = pd.read_csv("../data/data_split.csv")
    df_train = df[(df.fold != fold) & (df.split != "validation")]
    dataset = SemanticSynthesisDataset(df_train, cond_drop_rate=0., split="train")
    dataloader = DataLoader(dataset, batch_size=config["synthesize"]["batch_size"], shuffle=True)

    inference_outputs = inference_batch(
        m,
        dataloader,
        inference_steps=config["synthesize"]["inference_steps"],
        eta=config["synthesize"]["eta"],
        guidance_scale=config["synthesize"]["guidance_scale"]
    )

    save_dir = os.path.join(config["synthesize"]["output_dir"], f"fold-{fold}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "conditions"), exist_ok=True)

    save_outputs(inference_outputs, save_dir, append=config["synthesize"]["append"])

    del model, dataset, dataloader, inference_outputs
    torch.cuda.empty_cache()
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
    for fold in range(1):
        synthesize_images(fold, config)
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()