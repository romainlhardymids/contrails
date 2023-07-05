import sys
sys.path.append("../code")

import albumentations as A
import cv2
import numpy as np
import os
import random
import torch

from albumentations.pytorch import ToTensorV2
from skimage.filters import threshold_multiotsu
from torch.utils.data import Dataset
from utils import load_record, normalize_min_max, SEG_LABELS


class ContrailsDataset(Dataset):
    """Dataset class for contrail images."""
    def __init__(self, df, timesteps=5, image_size=384, split="train"):
        self.df = df
        self.timesteps = timesteps
        self.image_size = image_size
        self.split = split
        self.transform = get_transform(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        record_path = self.df.iloc[i]["record_path"]
        frames, mask = load_record(record_path, self.timesteps)
        stacked = []
        for step, frame in enumerate(frames):
            if step == 0:
                t = self.transform(image=frame, mask=mask)
                frame, mask = t["image"], t["mask"]
            else:
                t = self.transform(image=frame)
                frame = t["image"]
            frame = A.Compose([
                A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_LINEAR, always_apply=True),
                ToTensorV2(always_apply=True)
            ])(image=frame)["image"]
            stacked.append(frame)
        img = torch.stack(stacked, dim=0)
        sample = {
            "image": img,
            "mask": mask[None, :]
        }
        return sample
    

class ContrailsPretrainingDataset(Dataset):
    """Pretraining dataset class for contrail images."""
    def __init__(
        self, 
        df_real,
        df_synthetic, 
        pseudo_label_dir,
        timesteps=5, 
        image_size=384, 
        split="train"
    ):
        self.df_real = df_real
        self.df_synthetic = df_synthetic
        self.pseudo_label_dir = pseudo_label_dir
        self.timesteps = timesteps
        self.image_size = image_size
        self.split = split
        self.transform = get_transform(split)
        self._r = len(df_real)
        self._s = len(df_synthetic) if df_synthetic is not None else 0

    def __len__(self):
        return self._r + self._s

    def __getitem__(self, i):
        if i >= self._r:
            row = self.df_synthetic.iloc[i - self._r]
            frame = cv2.imread(row.image_path)
            mask = cv2.imread(row.condition_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            t = A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=1.0, always_apply=True)(image=frame, mask=mask)
            frame, mask = t["image"], t["mask"]
        else:
            timestep = np.random.choice(self.timesteps) # Choose a random timestep
            row = self.df_real.iloc[i]
            record_id = row["record_id"]
            record_path = row["record_path"]
            pseudo_label_path = os.path.join(self.pseudo_label_dir, f"{record_id}.npy")
            frames, _ = load_record(record_path, self.timesteps)
            masks = np.load(pseudo_label_path)
            frame, mask = frames[timestep], masks[timestep]
            t = self.transform(image=frame, mask=mask[0])
            frame, mask = t["image"], t["mask"]
        frame = A.Compose([
            A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_LINEAR, always_apply=True),
            ToTensorV2(always_apply=True)
        ])(image=frame)["image"]
        sample = {
            "image": frame,
            "mask": mask[None, :]
        }
        return sample
    

class SemanticSynthesisDataset(Dataset):
    def __init__(self, df, cond_drop_rate=0.1, split="train"):
        self.df = df
        self.cond_drop_rate = cond_drop_rate
        self.split = split
        self.transform = get_transform(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        record_path = self.df.iloc[i]["record_path"]
        frames, mask = load_record(record_path, timesteps=1)
        img = frames[0]
        t = self.transform(image=frames[0], mask=mask)
        img, mask = t["image"], t["mask"]
        if random.random() < self.cond_drop_rate:
            seg = np.zeros(shape=(128, 128)).astype(np.uint8)
        else:
            seg = get_segmentation(img, mask)
        sample = {
            "image": img,
            "segmentation": np.eye(SEG_LABELS)[seg].astype(np.uint8)
        }
        return sample


def repeated_blur(img, n=1, k=9):
    prev = img
    for i in range(n):
        next = cv2.GaussianBlur(prev, (k, k), -1)
        prev = next
    return prev


def get_segmentation(img, mask):
    seg = np.zeros(shape=(128, 128), dtype=np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (128, 128), cv2.INTER_CUBIC)
    mask_resized = cv2.resize(mask.astype(np.uint8), (128, 128), cv2.INTER_NEAREST)
    img_blurred = repeated_blur(img_resized, 1, 9)
    thresholds = threshold_multiotsu(img_blurred, classes=4)
    seg[img_blurred < thresholds[0]] = 0
    seg[(img_blurred > thresholds[0]) & (img_blurred < thresholds[1])] = 1
    seg[(img_blurred > thresholds[1]) & (img_blurred < thresholds[2])] = 2
    seg[(img_blurred > thresholds[2])] = 3
    seg[mask_resized > 0] = 4
    return seg
    

def get_transform(split="train"):
    if split == "train":
        return A.Compose([
            A.OneOf([
                A.Flip(p=0.5),
                A.Rotate(limit=90, interpolation=cv2.INTER_LINEAR, p=0.25),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, interpolation=cv2.INTER_LINEAR, p=0.25),
                A.RandomResizedCrop(height=256, width=256, scale=(0.6, 1.4), ratio=(0.8, 1.2), interpolation=cv2.INTER_LINEAR, p=0.5),
            ], p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=1.0, always_apply=True)
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=1.0, always_apply=True)
        ])