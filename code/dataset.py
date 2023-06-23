import sys
sys.path.append("../code")

import albumentations as A
import numpy as np
import pandas as pd
import random

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from utils import load_record


class ContrailsDataset(Dataset):
    """Dataset class for contrail images."""
    def __init__(self, meta, split="train"):
        self.meta = meta
        self.split = split
        self.transform = get_transform(split)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        record_id = self.meta[i]["record_id"]
        img, mask = load_record(record_id, self.split)
        if self.transform is not None:
            t = self.transform(image=img, mask=mask)
            img, mask = t["image"], t["mask"]
        sample = {
            "image": img,
            "mask": mask
        }
        return sample
    

def get_transform(split="train"):
    if split == "train":
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0, always_apply=True),
            ToTensorV2(always_apply=True)
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0, always_apply=True),
            ToTensorV2(always_apply=True)
        ])