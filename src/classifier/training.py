import os
import json
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import numpy as np
import albumentations as A

import tools.tools as tools
import classifier.network as nw



# Dataset with joint RGB + MV augmentations
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
        self.augment = augment
        self.augment_fn = self._apply_augs_one if self.augment else None

        # Augmentation pipelines
        self.color_aug = None
        self.geom_aug  = None
        if self.augment:
            # Color/Noise/Compression applied to RGB only
            self.color_aug = A.Compose([
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.20),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.20),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.25),
            ])

            # Geometric Augmentations applied to BOTH RGB and MV
            self.geom_aug = A.ReplayCompose(
                [
                    A.HorizontalFlip(p=0.5),
                ],
                additional_targets={'mv': 'image'}  # ensure identical spatial ops for MV/IM
            )

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
        
        device = img_t.device

        # Convert to CPU numpy (For Albumentations)
        img_hwc = img_t.permute(1, 2, 0).cpu().numpy()
        mv_hwc  = mv_t.permute(1, 2, 0).cpu().numpy()

        # Geometry augmentations
        if self.geom_aug is not None:
            out = self.geom_aug(image=img_hwc, mv=mv_hwc)
            img_hwc = out["image"]
            mv_hwc  = out["mv"]
            replay  = out["replay"]

            # Manual MV adjustments for HorizontalFlip
            for t in replay["transforms"]:
                if t.get("applied", False):
                    name = t.get("__class_fullname__", "")
                    if name.endswith("HorizontalFlip"):
                        mv_hwc[:, :, 0] = -mv_hwc[:, :, 0]

        # RGB augmentations
        if self.color_aug is not None:
            img_hwc = self.color_aug(image=img_hwc)["image"]

        # Back to CHW tensors
        img_out = torch.from_numpy(img_hwc).permute(2, 0, 1).float().clamp(0.0, 1.0).to(device)
        mv_out  = torch.from_numpy(mv_hwc).permute(2, 0, 1).float().to(device)

        return img_out, mv_out

    def __getitem__(self, idx):
        entry     = self.entries[idx]
        video_dir = os.path.join(self.dataset_path, entry["video"])
        label     = entry["label"]

        return tools.load_video_data(video_dir, label, augment_fn=self.augment_fn)


def evaluate(model, loader, device, criterion):
    """Evaluate average BCE loss on the given loader"""
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
    """Train for one epoch and return average loss"""
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


if __name__ == "__main__":

    dataset_path  = "../../../dataset"
    manifest_path = os.path.join(dataset_path, "manifest.json")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    all_entries = manifest["train"]

    # 90/10 split
    rng = np.random.default_rng(42)
    idx_all = np.arange(len(all_entries))
    rng.shuffle(idx_all)
    val_size = max(1, int(0.10 * len(idx_all)))
    val_idx  = idx_all[:val_size]
    train_idx= idx_all[val_size:]

    # Datasets creation
    full_train = VideoDataset(all_entries, dataset_path, augment=True)
    full_val   = VideoDataset(all_entries, dataset_path, augment=False)
    train_ds   = Subset(full_train, train_idx)
    val_ds     = Subset(full_val,   val_idx)

    # Config
    epochs          = 30
    warmup_epochs   = 5 # Backbone frozen for first N epochs
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

    # Model, criterion, optimizer
    model     = nw.MLP(in_channels=3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr_head_phase1)

    # Frozen backbones, train head only
    model.freeze_backbones(bn_eval=True)  # set both backbones to eval and no-grad
    
    train_losses, val_losses = [], []
    best_val = math.inf
    best_ep  = -1
    early_patience  = 5
    early_min_delta = 1e-4
    patience = 0
    unfrozen = False

    for ep in range(1, epochs + 1):
        # Unfreeze backbones after warmup and change LR
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
