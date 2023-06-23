# Imports
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
DATA_DIR = "../data/"
N_TIMES_BEFORE = 4
_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)

def load_metadata(split="train"):
    """Loads metatdata."""
    with open(os.path.join(DATA_DIR, f"{split}_metadata.json"), "r") as f:
        meta = json.load(f)
    return meta


def load_band(record_path, band):
    """Loads a specific band array."""
    return np.load(os.path.join(record_path, f"band_{band:02d}.npy"))


def load_mask(record_path):
    """Loads the ground-truth mask."""
    return np.load(os.path.join(record_path, "human_pixel_masks.npy"))[None, ..., 0]


def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def false_color(bands):
    """Returns a 3-channel false color image."""
    r = normalize_range(bands[2] - bands[1], _TDIFF_BOUNDS)
    g = normalize_range(bands[2] - bands[0], _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(bands[1], _T11_BOUNDS)
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    return false_color


def load_record(record_id, split="train"):
    """Returns a false color image and its ground-truth mask."""
    record_path = os.path.join(DATA_DIR, split, record_id)
    bands = [load_band(record_path, band) for band in [11, 14, 15]]
    stacked = false_color(bands)
    img = stacked[..., N_TIMES_BEFORE]
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