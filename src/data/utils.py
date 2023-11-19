import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from einops import rearrange
from scipy import signal
from sklearn.model_selection import KFold


# Global variables
DATA_DIR = "/home/romainlhardy/kaggle/contrails/data"
FOLDS = 5
N_TIMES_BEFORE = 4
SEG_LABELS = 5
PATCHES = 4
_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)


def normalize_range(data, bounds):
    """Applies min-max normalization to an input image."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def load_metadata(split="train"):
    """Loads metadata for a given split."""
    with open(os.path.join(DATA_DIR, f"{split}_metadata.json"), "r") as f:
        meta = json.load(f)
    for record in meta:
        record["split"] = split
        record["record_path"] = os.path.join(DATA_DIR, split, record["record_id"])
    return meta


def data_split(path):
    """Creates a cross-validation split for the segmentation task."""
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
        df = dedup_records(df)
        df.to_csv(path, index=None)
    return df


def dedup_records(df):
    """Deduplicates records by timestamp."""
    rows = []
    for _, group in df.groupby(["row_min", "row_size", "col_min", "col_size"]):
        prev_timestamp = None
        for _, row in group.sort_values(by="timestamp", ascending=True).iterrows():
            if prev_timestamp is not None and (row.timestamp - prev_timestamp) / 3600 < 1.:
                pass
            else:
                rows.append(row)
            prev_timestamp = row.timestamp
    return pd.DataFrame(rows).reset_index(drop=True)


def ash_color(bands):
    """Ash color preprocessing."""
    r = normalize_range(bands[2] - bands[1], _TDIFF_BOUNDS)
    g = normalize_range(bands[2] - bands[0], _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(bands[1], _T11_BOUNDS)
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    return false_color


def normalize_temperature_signal(t):
    """Temperature normalization."""
    t_smooth = cv2.GaussianBlur(t, (5, 5), 0)
    t_signal = np.clip(t - t_smooth, 0, 1)
    t_std = np.sqrt(cv2.GaussianBlur(t_signal ** 2, (5, 5), 0))
    t_norm = t_signal / (t_std + 0.1)
    return t_norm


def load_band(record_path, band):
    """Loads a specific band for a given record."""
    return np.load(os.path.join(record_path, f"band_{band:02d}.npy"))


def load_mask(record_path):
    """Loads a segmentation mask for a given record."""
    mask = np.load(os.path.join(record_path, "human_pixel_masks.npy"))[..., 0]
    return mask


def load_pseudo_label(pseudo_label_path, timestep):
    """Loads a pseudo label at a given timestep."""
    mask = np.load(pseudo_label_path)[timestep, 0]
    return mask


def load_frames(record_path, timesteps, add_temp_diff=False):
    """Loads and preprocesses images at given timesteps."""
    bands = [load_band(record_path, band) for band in [11, 14, 15]]
    frames = ash_color(bands)
    frames = np.transpose(frames[..., timesteps], (3, 0, 1, 2))
    if add_temp_diff:
        temp_diff = []
        for t_diff in frames[..., 0]:
            temp_diff.append(normalize_temperature_signal(t_diff))
        temp_diff = np.stack(temp_diff, axis=0)
        frames = np.concatenate([frames, temp_diff[..., None]], axis=-1)
    return frames


def load_record(record_path, timesteps=[N_TIMES_BEFORE], add_temp_diff=False, use_pseudo_labels=False):
    """Loads a full record (input frames and label)."""
    frames = load_frames(record_path, timesteps, add_temp_diff)
    mask = load_mask(record_path)
    return frames, mask


def cut_hole(h, w, min_size, max_size):
    """Returns random rectangular boundaries for a hole to be cut out of an image."""
    s = np.random.choice(min_size) + (max_size - min_size)
    i = np.random.choice(h - s)
    j = np.random.choice(w - s)
    y0, y1 = i, i + s
    x0, x1 = j, j + s
    return x0, x1, y0, y1


def cutmix(
    frames, 
    mask, 
    timesteps, 
    cutmix_path, 
    num_holes=2,
    min_size=16, 
    max_size=48, 
    add_temp_diff=False
):
    """Applies the CutMix augmentation to a set of input frames."""
    _, h, w, _ = frames.shape
    cutmix_frames = load_frames(cutmix_path, timesteps, add_temp_diff)
    cutmix_mask = None if mask is None else load_mask(cutmix_path)
    for _ in range(num_holes):
        x0, x1, y0, y1 = cut_hole(h, w, min_size, max_size)
        frames[:, y0 : y1, x0 : x1, :] = cutmix_frames[:, y0 : y1, x0 : x1]
        if mask is not None:
            mask[y0 : y1, x0 : x1] = cutmix_mask[y0 : y1, x0 : x1]
    return frames, mask