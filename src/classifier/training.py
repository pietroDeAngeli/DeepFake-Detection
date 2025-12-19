import os
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A

import classifier.network as nw


# ===================== CONFIG =====================
SEED = 42

EPOCHS = 50
WARMUP_EPOCHS = 5
LR_HEAD_PHASE1 = 1e-3
LR_ALL_PHASE2  = 1e-4

EARLY_PATIENCE = 5
EARLY_MIN_DELTA = 1e-4

BATCH_SIZE = 1          # 1 video per batch (video length variabile)
NUM_WORKERS = 0

AUGMENT_TRAIN = True
OUT_DIR_NAME = "run_single_split_ft"

# ImageNet normalization (standard)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# ==================================================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def project_root() -> Path:
    # .../DeepFake-Detection/src/classifier/training_final.py -> parents[2] = project root
    return Path(__file__).resolve().parents[2]


def is_valid_sample(video_dir: Path) -> bool:
    return (video_dir / "tensors.pt").is_file() and (video_dir / "faces").is_dir()


def load_video_data_fixed(video_dir: Path, label: int, augment_fn=None):
    """
    Carica un sample pre-processato:
      - tensors.pt: {"features": [N,H,W,3]} (MVx, MVy, IM)
      - faces/: frame_XXXX.jpg

    IMPORTANT: augmentation su RGB in [0,1] -> poi ImageNet normalization.
    """
    # MV
    feat = torch.load(video_dir / "tensors.pt", map_location="cpu")["features"]  # [N,H,W,3]
    mv_tensor = feat.permute(0, 3, 1, 2).float()  # [N,3,H,W]

    # RGB faces
    faces_dir = video_dir / "faces"
    img_files = sorted([p for p in faces_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if len(img_files) == 0:
        return None

    imgs = []
    for p in img_files:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(im).permute(2, 0, 1).float().div(255.0)  # [3,H,W] in [0,1]
        imgs.append(im)

    if len(imgs) == 0:
        return None

    imgs_tensor = torch.stack(imgs, dim=0)  # [N,3,H,W] in [0,1]

    # Safety: se per qualche motivo N non coincide, tronca
    n = min(imgs_tensor.size(0), mv_tensor.size(0))
    if n <= 0:
        return None
    imgs_tensor = imgs_tensor[:n]
    mv_tensor   = mv_tensor[:n]

    # Augment (su RGB in [0,1] e MV raw)
    if augment_fn is not None:
        aug_imgs, aug_mvs = [], []
        for i in range(n):
            img_i, mv_i = augment_fn(imgs_tensor[i], mv_tensor[i])
            aug_imgs.append(img_i)
            aug_mvs.append(mv_i)
        imgs_tensor = torch.stack(aug_imgs, dim=0)
        mv_tensor   = torch.stack(aug_mvs,  dim=0)

    # NORMALIZZAZIONE DOPO AUG (fix rispetto al bug precedente)
    imgs_tensor = (imgs_tensor - IMAGENET_MEAN) / IMAGENET_STD

    label_tensor = torch.tensor([int(label)], dtype=torch.float32)
    return (imgs_tensor, mv_tensor), label_tensor


class VideoDataset(Dataset):
    def __init__(self, entries, dataset_path: Path, augment: bool = False):
        self.entries = entries
        self.dataset_path = Path(dataset_path)
        self.augment = augment
        self.augment_fn = self._apply_augs_one if self.augment else None

        self.color_aug = None
        self.geom_aug  = None

        if self.augment:
            # RGB-only
            self.color_aug = A.Compose([
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.20),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.20),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.25),
            ])

            # Shared geometry on both RGB and MV
            self.geom_aug = A.ReplayCompose(
                [A.HorizontalFlip(p=0.5)],
                additional_targets={"mv": "image"},
            )

    def __len__(self):
        return len(self.entries)

    def _apply_augs_one(self, img_t: torch.Tensor, mv_t: torch.Tensor):
        """
        img_t: (3,H,W) in [0,1]
        mv_t:  (3,H,W)
        """
        device = img_t.device

        img_hwc = img_t.permute(1, 2, 0).cpu().numpy()
        mv_hwc  = mv_t.permute(1, 2, 0).cpu().numpy()

        # geometry
        if self.geom_aug is not None:
            out = self.geom_aug(image=img_hwc, mv=mv_hwc)
            img_hwc = out["image"]
            mv_hwc  = out["mv"]
            replay  = out["replay"]

            # if horizontal flip applied -> invert mv_x
            for t in replay["transforms"]:
                if t.get("applied", False):
                    name = t.get("__class_fullname__", "")
                    if name.endswith("HorizontalFlip"):
                        mv_hwc[:, :,]()_
