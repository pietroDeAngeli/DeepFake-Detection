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
LR_ALL_PHASE2 = 1e-4

EARLY_PATIENCE = 5
EARLY_MIN_DELTA = 1e-4

BATCH_SIZE = 1          # 1 video per batch (lunghezze variabili)
NUM_WORKERS = 0

AUGMENT_TRAIN = True
OUT_DIR_NAME = "run_single_split_ft"

# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# ==================================================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def project_root() -> Path:
    # .../DeepFake-Detection/src/classifier/training.py -> parents[2] = project root
    return Path(__file__).resolve().parents[2]


def is_valid_sample(video_dir: Path) -> bool:
    return (video_dir / "tensors.pt").is_file() and (video_dir / "faces").is_dir()


def load_video_data(video_dir: Path, label: int, augment_fn=None):
    """
    Load one preprocessed sample:
      - tensors.pt: {"features": [N,H,W,3]} (mvx,mvy,im)
      - faces/: frame_XXXX.jpg

    IMPORTANT: RGB augmentation on [0,1], then ImageNet normalization AFTER augs.
    """
    # Load MV features: [N,H,W,3] -> [N,3,H,W]
    feat = torch.load(video_dir / "tensors.pt", map_location="cpu")["features"]
    mv_tensor = feat.permute(0, 3, 1, 2).float()

    # Load face images: [N,3,H,W] in [0,1]
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
        im = torch.from_numpy(im).permute(2, 0, 1).float().div(255.0)
        imgs.append(im)

    if len(imgs) == 0:
        return None

    imgs_tensor = torch.stack(imgs, dim=0)

    # Align lengths (safety)
    n = min(imgs_tensor.size(0), mv_tensor.size(0))
    if n <= 0:
        return None
    imgs_tensor = imgs_tensor[:n]
    mv_tensor = mv_tensor[:n]

    # Apply augmentation per-frame (RGB in [0,1])
    if augment_fn is not None:
        aug_imgs, aug_mvs = [], []
        for i in range(n):
            img_i, mv_i = augment_fn(imgs_tensor[i], mv_tensor[i])
            aug_imgs.append(img_i)
            aug_mvs.append(mv_i)
        imgs_tensor = torch.stack(aug_imgs, dim=0)
        mv_tensor = torch.stack(aug_mvs, dim=0)

    # Normalize RGB AFTER augmentation
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
        self.geom_aug = None

        if self.augment:
            # RGB-only
            self.color_aug = A.Compose([
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.20),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.20),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.25),
            ])

            # Shared geometry for RGB + MV
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

        img_hwc = img_t.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        mv_hwc = mv_t.permute(1, 2, 0).cpu().numpy().astype(np.float32)

        # Geometry on both
        if self.geom_aug is not None:
            out = self.geom_aug(image=img_hwc, mv=mv_hwc)
            img_hwc = out["image"]
            mv_hwc = out["mv"]
            replay = out["replay"]

            # If HorizontalFlip applied, invert mv_x (channel 0)
            for t in replay["transforms"]:
                if t.get("applied", False):
                    name = t.get("__class_fullname__", "")
                    if name.endswith("HorizontalFlip"):
                        mv_hwc[:, :, 0] = -mv_hwc[:, :, 0]

        # Color on RGB only
        if self.color_aug is not None:
            img_hwc = self.color_aug(image=img_hwc)["image"]

        img_out = torch.from_numpy(img_hwc).permute(2, 0, 1).float().clamp(0.0, 1.0).to(device)
        mv_out = torch.from_numpy(mv_hwc).permute(2, 0, 1).float().to(device)
        return img_out, mv_out

    def __getitem__(self, idx):
        e = self.entries[idx]
        video_dir = self.dataset_path / e["video"]
        label = int(e["label"])

        if not is_valid_sample(video_dir):
            return None

        return load_video_data(video_dir, label, augment_fn=self.augment_fn)


def collate_single(batch):
    # batch_size=1: return the single element
    return batch[0]


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for item in loader:
            if item is None:
                continue
            (imgs, mvs), label = item
            imgs = imgs.to(device, non_blocking=True)
            mvs = mvs.to(device, non_blocking=True)
            label = label.to(device).view(-1)
            logits = model((imgs, mvs)).view(-1)
            loss = criterion(logits, label)
            total_loss += loss.item()
            n += 1
    return total_loss / max(1, n)


def train_one_epoch(model, loader, device, criterion, optimizer, desc="Train"):
    model.train()
    running = 0.0
    n = 0
    for item in tqdm(loader, desc=desc, leave=False):
        if item is None:
            continue
        (imgs, mvs), label = item
        imgs = imgs.to(device, non_blocking=True)
        mvs = mvs.to(device, non_blocking=True)
        label = label.to(device).view(-1)

        optimizer.zero_grad(set_to_none=True)
        logits = model((imgs, mvs)).view(-1)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        running += loss.item()
        n += 1

    return running / max(1, n)


if __name__ == "__main__":
    set_seed(SEED)

    root = project_root()
    dataset_path = root / "dataset" / "preprocessed"
    manifest_path = dataset_path / "manifest.json"

    if not manifest_path.is_file():
        raise RuntimeError(f"manifest.json not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_entries = manifest.get("train", [])
    val_entries = manifest.get("val", [])

    if len(train_entries) == 0 or len(val_entries) == 0:
        raise RuntimeError("manifest.json must contain non-empty 'train' and 'val' splits.")

    # Filter entries that exist on disk
    train_entries = [e for e in train_entries if is_valid_sample(dataset_path / e["video"])]
    val_entries = [e for e in val_entries if is_valid_sample(dataset_path / e["video"])]

    if len(train_entries) == 0 or len(val_entries) == 0:
        raise RuntimeError("No valid samples after filtering. Check preprocessing output.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = root / "src" / "classifier" / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = VideoDataset(train_entries, dataset_path, augment=AUGMENT_TRAIN)
    val_ds = VideoDataset(val_entries, dataset_path, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_single,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_single,
    )

    model = nw.MLP(in_channels=3).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Phase 1: head only
    model.freeze_backbones(bn_eval=True)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR_HEAD_PHASE1)

    best_val = math.inf
    best_ep = -1
    patience = 0
    unfrozen = False

    train_losses, val_losses = [], []

    for ep in range(1, EPOCHS + 1):
        if (not unfrozen) and (ep > WARMUP_EPOCHS):
            model.unfreeze_backbones()
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR_ALL_PHASE2)
            unfrozen = True
            print(f"[Info] Unfreeze backbone at epoch {ep} (lr={LR_ALL_PHASE2:g})")

        avg_tr = train_one_epoch(
            model, train_loader, device, criterion, optimizer,
            desc=f"Epoch {ep}/{EPOCHS}{' (frozen)' if not unfrozen else ''}"
        )
        avg_va = evaluate(model, val_loader, device, criterion)

        train_losses.append(avg_tr)
        val_losses.append(avg_va)

        print(f"Epoch {ep:03d} | TrainLoss: {avg_tr:.4f} | ValLoss: {avg_va:.4f}")

        if avg_va < best_val - EARLY_MIN_DELTA:
            best_val = avg_va
            best_ep = ep
            patience = 0
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            with open(out_dir / "best_summary.json", "w", encoding="utf-8") as f:
                json.dump({"best_epoch": best_ep, "best_val_loss": float(best_val)}, f, indent=2)
        else:
            patience += 1
            if patience >= EARLY_PATIENCE:
                print(f"Early stopping at epoch {ep} (best epoch {best_ep})")
                break

    with open(out_dir / "losses.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_loss": train_losses,
                "val_loss": val_losses,
                "best_epoch": best_ep,
                "best_val_loss": float(best_val),
            },
            f,
            indent=2,
        )

    print("Done. Artifacts in:", str(out_dir))
