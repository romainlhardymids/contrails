# Imports
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from einops import rearrange
from sklearn.model_selection import KFold

# Constants
DATA_DIR = "../data/"
FOLDS = 5
N_TIMES_BEFORE = 4
SEG_LABELS = 5
_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)
_CLOUD_BOUNDS = (323, 203)


def load_metadata(split="train"):
    """Loads metatdata."""
    with open(os.path.join(DATA_DIR, f"{split}_metadata.json"), "r") as f:
        meta = json.load(f)
    for record in meta:
        record["split"] = split
        record["record_path"] = os.path.join(DATA_DIR, split, record["record_id"])
    return meta


def load_synthetic_metadata():
    meta = None
    for fold in range(1): # Change later
        meta_ = pd.read_csv(os.path.join(DATA_DIR, "synthetic", f"fold-{fold}", "metadata.csv"))
        meta_["fold"] = fold
        if meta is None:
            meta = meta_
        else:
            meta = pd.concat([meta, meta_], axis=0)
    meta = meta.reset_index(drop=True)
    return meta


def data_split(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df_train = pd.DataFrame(load_metadata("train"))
        df_valid = pd.DataFrame(load_metadata("validation"))
        kf = KFold(shuffle=True, n_splits=FOLDS)
        for n, (_, valid_idx) in enumerate(kf.split(df_train)):
            df_train.loc[valid_idx, "fold"] = int(n)
        df_train["fold"] = df_train["fold"].astype(int)
        df_valid["fold"] = -1
        df = pd.concat([df_train, df_valid], axis=0).reset_index(drop=True)
        df.to_csv(path, index=None)
    return df


def load_band(record_path, band):
    """Loads a specific band array."""
    return np.load(os.path.join(record_path, f"band_{band:02d}.npy"))


def load_mask(record_path):
    """Loads the ground-truth mask."""
    return np.load(os.path.join(record_path, "human_pixel_masks.npy"))[..., 0]


def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def normalize_min_max(data):
    """Min-max normalization."""
    return (data - data.min()) / (data.max() - data.min())


def normalize_diffusion(img, permute=True):
    """Normalization function for diffusion models."""
    x = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
    if permute:
        x = 255. * rearrange(x.cpu().numpy(), "c h w -> h w c")
    else:
        x = 255. * x.cpu().numpy()
    return x.astype(np.uint8)


def ash_color(bands):
    """Returns a 3-channel false color image."""
    r = normalize_range(bands[2] - bands[1], _TDIFF_BOUNDS)
    g = normalize_range(bands[2] - bands[0], _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(bands[1], _T11_BOUNDS)
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    return false_color


def cloud_color(bands):
    """Returns a 3-channel false color image."""
    r = normalize_min_max(bands[0])
    g = normalize_min_max(bands[1])
    b = normalize_range(bands[2].max() - bands[2], _T11_BOUNDS)
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    return false_color


def load_record(record_path, timesteps):
    """Returns a false color image and its ground-truth mask."""
    bands = [load_band(record_path, band) for band in [11, 14, 15]]
    stacked = ash_color(bands)
    a = N_TIMES_BEFORE - timesteps // 2
    b = a + timesteps
    img = np.transpose(stacked[..., a:b], (3, 0, 1, 2))
    mask = load_mask(record_path)
    return img, mask


def plot_image_and_mask(img, mask):
    _, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[1].imshow(mask)
    plt.show()


class dotdict(dict):
    """Dot notation to access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__