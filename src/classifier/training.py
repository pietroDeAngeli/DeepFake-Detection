import os
import json
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import cv2
import numpy as np

# Try to import Albumentations; if missing, we will skip augmentations gracefully
try:
    import albumentations as A
    _HAS_ALBU = True
except Exception:
    _HAS_ALBU = False

import classifier.network as nw  # MLP wrapping two MobileNetV3_adaption branches

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# Dataset with joint RGB + MV augmentations
# -----------------------------
class VideoDataset(Dataset):
    def __init__(self, entries, dataset_path, augment: bool = False):
        """
        Args:
            entries (list[dict]): items with keys {"video","label"}
            dataset_path (str): root path that contains one subdir per video
            augment (bool): enable paper-style augmentations only for training
        """
        self.entries = entries
        self.dataset_path = dataset_path
        self.augment = augment and _HAS_ALBU

        # Build augmentation pipelines (paper-style). If Albumentations is missing, keep None.
        self.color_aug = None
        self.geom_aug  = None
        if self.augment:
            # Color/noise/compression augmentations applied to RGB ONLY
            # Ref: paper Section "Data Augmentation" (image compression, Gaussian noise/blur,
            # RGB & HSV shifts, FancyPCA, RandomBrightnessContrast, Gray-scale). 
            self.color_aug = A.Compose([
                A.ImageCompression(quality_lower=35, quality_upper=95, p=0.25),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.25),
                A.GaussianBlur(blur_limit=(3, 5), p=0.20),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.20),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.20),
                A.FancyPCA(alpha=0.1, p=0.15),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.25),
                A.ToGray(p=0.10),
            ])

            # Geometry/GridMask applied to BOTH RGB and MV/IM (must be identical & synchronized)
            # We use ReplayCompose so we can read which flips were applied to mirror MV signs.
            self.geom_aug = A.ReplayCompose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                    # Use GridDropout as GridMask proxy: zero-out patches on a grid over the input.
                    A.GridDropout(
                        ratio=0.5,         # fraction of dropped area
                        random_offset=True,
                        holes_number_x=None,
                        holes_number_y=None,
                        unit_size_min=32,  # choose sensible cell size for 224x224
                        unit_size_max=64,
                        p=0.20
                    ),
                ],
                additional_targets={'mv': 'image'}  # ensure identical spatial ops for MV/IM
            )
        elif augment and not _HAS_ALBU:
            print("[Warn] Albumentations not installed. Augmentations are disabled.")

    def __len__(self):
        return len(self.entries)

    def _apply_augs_one(self, img_t: torch.Tensor, mv_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply joint augmentations to a single frame.
        img_t:  (3,H,W) float in [0,1]
        mv_t:   (3,H,W) float where channels = [mvx, mvy, im] (already standardized/scaled)
        Returns augmented (img_t, mv_t).
        """
        if (self.color_aug is None and self.geom_aug is None):
            return img_t, mv_t

        # Convert to HWC numpy (Albumentations expects uint8 or float HWC)
        img_hwc = img_t.permute(1, 2, 0).cpu().numpy()
        mv_hwc  = mv_t.permute(1, 2, 0).cpu().numpy()

        # 1) Geometry & GridMask on BOTH (keep transforms replay to infer flips)
        if self.geom_aug is not None:
            out = self.geom_aug(image=img_hwc, mv=mv_hwc)
            img_hwc = out["image"]
            mv_hwc  = out["mv"]
            replay  = out["replay"]

            # Mirror MV components according to flips applied:
            # - Horizontal flip: invert x-component
            # - Vertical flip:   invert y-component
            for t in replay["transforms"]:
                if not t.get("applied", False):
                    continue
                name = t.get("__class_fullname__", "")
                if name.endswith("HorizontalFlip"):
                    mv_hwc[:, :, 0] = -mv_hwc[:, :, 0]  # mv_x -> -mv_x
                if name.endswith("VerticalFlip"):
                    mv_hwc[:, :, 1] = -mv_hwc[:, :, 1]  # mv_y -> -mv_y

        # 2) Color/noise/compression on RGB ONLY
        if self.color_aug is not None:
            img_hwc = self.color_aug(image=img_hwc)["image"]

        # Ensure shape consistency (ToGray may produce single channel → expand back to 3)
        if img_hwc.ndim == 2:
            img_hwc = np.expand_dims(img_hwc, 2)
        if img_hwc.shape[2] == 1:
            img_hwc = np.repeat(img_hwc, 3, axis=2)

        # Back to CHW tensors
        img_out = torch.from_numpy(img_hwc).permute(2, 0, 1).float().clamp(0.0, 1.0)
        mv_out  = torch.from_numpy(mv_hwc).permute(2, 0, 1).float()

        return img_out, mv_out

    def __getitem__(self, idx):
        entry     = self.entries[idx]
        video_dir = os.path.join(self.dataset_path, entry["video"])
        label     = entry["label"]

        # 1) Load MV+IM feature tensor: [N, H, W, 3] with channels (mvx, mvy, im)
        tensor_path = os.path.join(video_dir, "tensors.pt")
        data        = torch.load(tensor_path)
        features    = data["features"]                          # [N, H, W, 3], float32
        mv_tensor   = features.permute(0, 3, 1, 2).float()      # [N, 3, H, W]

        # 2) Load face RGB frames as float in [0,1] → [N, 3, H, W]
        faces_dir = os.path.join(video_dir, "faces")
        img_files = sorted(os.listdir(faces_dir))               # expected N=100
        imgs = []
        for fname in img_files:
            img = cv2.imread(os.path.join(faces_dir, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            imgs.append(img)
        imgs_tensor = torch.stack(imgs, dim=0)                  # [N, 3, H, W]

        # 3) Apply augmentations frame-by-frame (train only)
        if self.augment:
            aug_imgs = []
            aug_mvs  = []
            for i in range(imgs_tensor.size(0)):
                img_i, mv_i = self._apply_augs_one(imgs_tensor[i], mv_tensor[i])
                aug_imgs.append(img_i)
                aug_mvs.append(mv_i)
            imgs_tensor = torch.stack(aug_imgs, dim=0)
            mv_tensor   = torch.stack(aug_mvs,  dim=0)

        # BCEWithLogits expects float target; model forward expects ((imgs, mvs))
        return (imgs_tensor, mv_tensor), torch.tensor([[label]], dtype=torch.float32)

# -----------------------------
# Train / Eval
# -----------------------------
def evaluate(model, loader, device, criterion):
    """Evaluate average BCE loss on the given loader (no augmentation)."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (imgs, mvs), label in loader:
            imgs  = imgs[0].to(device)        # [N, 3, H, W]
            mvs   = mvs[0].to(device)         # [N, 3, H, W]
            label = label.to(device).view(-1)  # [1]
            logits = model((imgs, mvs))        # [1] (logits)
            loss   = criterion(logits, label)
            total_loss += loss.item()
    return total_loss / max(1, len(loader))

def train_one_epoch(model, loader, device, criterion, optimizer, desc="Train"):
    """Train for one epoch and return average loss."""
    model.train()
    running = 0.0
    for (imgs, mvs), label in tqdm(loader, desc=desc, leave=False):
        imgs  = imgs[0].to(device)
        mvs   = mvs[0].to(device)
        label = label.to(device).view(-1)

        optimizer.zero_grad(set_to_none=True)
        logits = model((imgs, mvs))    # [1] (logits)
        loss   = criterion(logits, label)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))

# -----------------------------
# Main: single split (fixed val = 10%) + 2 phases (freeze -> unfreeze)
# -----------------------------
if __name__ == "__main__":
    set_seed(42)

    dataset_path  = "../../dataset"
    manifest_path = os.path.join(dataset_path, "manifest.json")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    all_entries = manifest["train"]

    # Fixed 90/10 split
    rng = np.random.default_rng(42)
    idx_all = np.arange(len(all_entries))
    rng.shuffle(idx_all)
    val_size = max(1, int(0.10 * len(idx_all)))
    val_idx  = idx_all[:val_size]
    train_idx= idx_all[val_size:]

    # Datasets: train with augmentations, val without
    full_train = VideoDataset(all_entries, dataset_path, augment=True)
    full_val   = VideoDataset(all_entries, dataset_path, augment=False)
    train_ds   = Subset(full_train, train_idx)
    val_ds     = Subset(full_val,   val_idx)

    # Config
    epochs          = 30
    warmup_epochs   = 5          # phase 1: freeze backbone
    lr_head_phase1  = 1e-3
    lr_all_phase2   = 1e-4
    num_workers     = 0
    batch_size      = 1
    out_dir         = "run_single_split_ft"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type=="cuda"))

    # Model returns logits; your implementation separates backbone/head
    model     = nw.MLP(in_channels=3).to(device)  # two MobileNetV3_adaption branches
    criterion = nn.BCEWithLogitsLoss()

    # ---- Phase 1: freeze backbones, train adapters+head+alpha
    model.freeze_backbones(bn_eval=True)  # set both backbones to eval and no-grad
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr_head_phase1)

    train_losses, val_losses = [], []
    best_val = math.inf
    best_ep  = -1
    early_patience  = 5
    early_min_delta = 1e-4
    patience = 0
    unfrozen = False

    for ep in range(1, epochs + 1):
        # Switch to phase 2: unfreeze backbones and use lower LR
        if (not unfrozen) and (ep > warmup_epochs):
            model.unfreeze_backbones()
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr_all_phase2)
            unfrozen = True
            print(f"[Info] Unfreeze backbone at epoch {ep} with lr={lr_all_phase2:g}")

        avg_tr = train_one_epoch(model, train_loader, device, criterion, optimizer,
                                 desc=f"Epoch {ep}/{epochs}{' (frozen)' if not unfrozen else ''}")
        avg_va = evaluate(model, val_loader, device, criterion)

        train_losses.append(avg_tr)
        val_losses.append(avg_va)
        print(f"Epoch {ep:03d} | TrainLoss: {avg_tr:.4f} | ValLoss: {avg_va:.4f}")

        # Early stopping on validation loss
        if avg_va < best_val - early_min_delta:
            best_val = avg_va
            best_ep  = ep
            patience = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
            with open(os.path.join(out_dir, "best_summary.json"), "w") as f:
                json.dump({"best_epoch": best_ep, "best_val_loss": float(best_val)}, f, indent=2)
        else:
            patience += 1
            if patience >= early_patience:
                print(f"Early stopping at epoch {ep} (best epoch {best_ep})")
                break

    with open(os.path.join(out_dir, "losses.json"), "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses,
                   "best_epoch": best_ep, "best_val_loss": float(best_val)}, f, indent=2)

    print("\nDone. Artifacts in:", os.path.abspath(out_dir))
