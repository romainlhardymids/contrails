import albumentations as A
import cv2
import numpy as np
import os
import random
import torch
import data.utils as utils

from albumentations.pytorch import ToTensorV2
from skimage.filters import threshold_multiotsu
from torch.utils.data import Dataset


PSEUDO_LABEL_DIR = "/home/romainlhardy/kaggle/contrails/data/pseudo-labels/masks"


class BaseDataset(Dataset):
    def __init__(
        self, 
        df, 
        image_size=384, 
        cutmix_prob=0.5, 
        cutmix_num_holes=2,
        cutmix_min_size=16,
        cutmix_max_size=48,
        split="train"
    ):
        self.df = df
        self.image_size = image_size
        self.cutmix_prob = cutmix_prob
        self.cutmix_num_holes = cutmix_num_holes
        self.cutmix_min_size = cutmix_min_size
        self.cutmix_max_size = cutmix_max_size
        self.split = split

    def __len__(self):
        return len(self.df)

    def choose_random_row(self):
        return self.df.iloc[np.random.choice(len(self.df))]

    def apply_cutmix(self, frames, mask, timesteps):
        if random.random() < self.cutmix_prob:
            cutmix_path = self.choose_random_row()["record_path"]
            return utils.cutmix(
                frames, 
                mask, 
                timesteps, 
                cutmix_path, 
                self.cutmix_num_holes, 
                self.cutmix_min_size,
                self.cutmix_max_size,
                False
            )
        else:
            return frames, mask
        
    def resize_to_tensor(self, image):
        image = A.Compose([
            A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_LINEAR, always_apply=True),
            ToTensorV2(always_apply=True)
        ])(image=image)["image"]
        return image
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        return None
    

class ContrailsDataset(Dataset):
    def __init__(
        self,
        df, 
        timesteps=[4],
        image_size=512, 
        cutmix_prob=0.5, 
        cutmix_num_holes=2,
        cutmix_min_size=16,
        cutmix_max_size=48,
        use_pseudo_labels=False,
        split="train"
    ):
        self.df = df
        self.timesteps = timesteps
        self.image_size = image_size
        self.cutmix_prob = cutmix_prob
        self.cutmix_num_holes = cutmix_num_holes
        self.cutmix_min_size = cutmix_min_size
        self.cutmix_max_size = cutmix_max_size
        self.use_pseudo_labels = use_pseudo_labels
        self.split = split
        self.transform = get_transform(self.split)
        if use_pseudo_labels:
            print(f"Setting `use_pseudo_labels = True` will override the `timesteps` argument.")

    def __len__(self):
        return len(self.df)

    def choose_random_row(self):
        return self.df.iloc[np.random.choice(len(self.df))]

    def resize_to_tensor(self, image):
        image = A.Compose([
            A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_LINEAR, always_apply=True),
            ToTensorV2(always_apply=True)
        ])(image=image)["image"]
        return image
    
    def apply_transform(self, frames, mask):
        tf = self.transform(image=frames[0], mask=mask)
        image, mask = tf["image"], tf["mask"]
        processed = [self.resize_to_tensor(image)]
        if len(frames) > 1:
            for frame in frames[1:]:
                frame = A.ReplayCompose.replay(tf["replay"], image=frame)["image"]
                processed.append(self.resize_to_tensor(frame))
        frames = torch.stack(processed, dim=1)
        return frames, mask

    def apply_cutmix(self, frames, mask, timesteps):
        if random.random() < self.cutmix_prob:
            cutmix_path = self.choose_random_row()["record_path"]
            return utils.cutmix(
                frames, 
                mask, 
                timesteps, 
                cutmix_path, 
                self.cutmix_num_holes, 
                self.cutmix_min_size,
                self.cutmix_max_size,
                False
            )
        else:
            return frames, mask

    def __getitem__(self, i):
        row = self.df.iloc[i]
        if self.use_pseudo_labels:
            timesteps = [np.random.choice(8)]
            frames = utils.load_frames(row["record_path"], timesteps, False)
            pseudo_label_path = os.path.join(PSEUDO_LABEL_DIR, f"{row['record_id']}.npy")
            mask = utils.load_pseudo_label(pseudo_label_path, timesteps[0])
        else:
            timesteps = self.timesteps
            frames, mask = utils.load_record(row["record_path"], timesteps, False)
        frames, mask = self.apply_cutmix(frames, mask, timesteps)
        frames, mask = self.apply_transform(frames, mask)
        sample = {
            "frames": frames,
            "mask": mask[None, :],
            "label": float(mask.sum() > 0)
        }
        return sample
    

# def get_transform(split="train"):
#     if split == "train":
#         augments = [
#             A.OneOf([
#                 A.Flip(p=0.5),
#                 A.Rotate(limit=90, interpolation=cv2.INTER_LINEAR, p=0.25),
#                 A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, interpolation=cv2.INTER_LINEAR, p=0.25),
#                 A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.2), ratio=(0.8, 1.2), interpolation=cv2.INTER_LINEAR, p=0.5),
#             ], p=0.5),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0, always_apply=True)
#         ]
#     else:
#         augments = [
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0, always_apply=True)
#         ]
#     return A.ReplayCompose(augments)


def get_transform(split="train"):
    if split == "train":
        augments = [
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.2), ratio=(0.8, 1.2), interpolation=cv2.INTER_LINEAR, p=0.8),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0, always_apply=True)
        ]
    else:
        augments = [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0, always_apply=True)
        ]
    return A.ReplayCompose(augments)