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

import classifier.network as nw  # your MLP(two-stream) lives here


# -----------------------------
# Reproducibility helpers
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset
# -----------------------------
class VideoDataset(Dataset):
    def __init__(self, entries, dataset_path):
        """
        entries: list of dicts with keys "video", "label"
        dataset_path: root folder containing one subfolder per video
        """
        self.entries = entries
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry     = self.entries[idx]
        video_dir = os.path.join(self.dataset_path, entry["video"])
        label     = entry["label"]

        # 1) Load MV+IM features saved as tensors.pt (float32)
        tensor_path = os.path.join(video_dir, "tensors.pt")
        data        = torch.load(tensor_path)
        features    = data["features"]                        # [N, H, W, 3], float32
        mv_tensor   = features.permute(0, 3, 1, 2).float()    # [N, 3, H, W]

        # 2) Load cropped face images (RGB uint8) and convert to float32
        faces_dir = os.path.join(video_dir, "faces")
        img_files = sorted(os.listdir(faces_dir))             # N=100 frame per video (fissi)
        imgs = []
        for fname in img_files:
            img = cv2.imread(os.path.join(faces_dir, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).float()    # [3, H, W]
            img = img.div(255.0)                                    # normalize to [0,1]
            imgs.append(img)
        imgs_tensor = torch.stack(imgs, dim=0)                      # [N, 3, H, W]

        # Return ((rgb_frames, mv_frames), label)
        return (imgs_tensor, mv_tensor), torch.tensor([[label]], dtype=torch.float32)


# -----------------------------
# Training utilities
# -----------------------------
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (imgs, mvs), label in loader:
            imgs  = imgs[0].to(device)            # [N, 3, H, W]
            mvs   = mvs[0].to(device)             # [N, 3, H, W]
            label = label.to(device).view(-1)     # [1]
            out   = model((imgs, mvs))            # [1]
            loss  = criterion(out, label)
            total_loss += loss.item()
    return total_loss / max(1, len(loader))


def train_one_epoch(model, loader, device, criterion, optimizer, desc="Train"):
    model.train()
    running = 0.0
    for (imgs, mvs), label in tqdm(loader, desc=desc, leave=False):
        imgs  = imgs[0].to(device)
        mvs   = mvs[0].to(device)
        label = label.to(device).view(-1)

        optimizer.zero_grad(set_to_none=True)
        out  = model((imgs, mvs))
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        running += loss.item()
    return running / max(1, len(loader))


# -----------------------------
# Main: single split (fixed val = 10%)
# -----------------------------
if __name__ == "__main__":
    set_seed(42)

    isOnColab = True

    # Paths
    dataset_path  = "../../dataset"
    if isOnColab:
        dataset_path = "/content/drive/MyDrive/DeepFake - Detection/dataset"
    manifest_path = os.path.join(dataset_path, "manifest.json")

    # Read train split from manifest.json
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    all_entries = manifest["train"]   # list of dicts: {"video": "...", "label": 0/1}

    # Fixed validation split = 10% (deterministico)
    rng = np.random.default_rng(42)
    idx_all = np.arange(len(all_entries))
    rng.shuffle(idx_all)
    val_size = max(1, int(0.10 * len(idx_all)))
    val_idx  = idx_all[:val_size]
    train_idx= idx_all[val_size:]

    # Datasets
    full_ds   = VideoDataset(all_entries, dataset_path)
    train_ds  = Subset(full_ds, train_idx)
    val_ds    = Subset(full_ds, val_idx)

    # Config
    epochs             = 30
    lr                 = 1e-4
    num_workers        = 2 
    batch_size         = 1
    out_dir            = "run_single_split"
    os.makedirs(out_dir, exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type=="cuda"))

    # Model / Optim
    model     = nw.MLP(in_channels=3).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train loop with early stopping on fixed val
    train_losses, val_losses = [], []
    best_val = math.inf
    best_ep  = -1
    early_patience  = 5
    early_min_delta = 1e-4
    patience = 0

    for ep in range(1, epochs + 1):
        avg_tr = train_one_epoch(model, train_loader, device, criterion, optimizer,
                                 desc=f"Epoch {ep}/{epochs}")
        avg_va = evaluate(model, val_loader, device, criterion)

        train_losses.append(avg_tr)
        val_losses.append(avg_va)

        print(f"Epoch {ep:03d} | TrainLoss: {avg_tr:.4f} | ValLoss: {avg_va:.4f}")

        # Early stopping
        if avg_va < best_val - early_min_delta:
            best_val = avg_va
            best_ep  = ep
            patience = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
            with open(os.path.join(out_dir, "best_summary.json"), "w") as f:
                json.dump({"best_epoch": best_ep, "best_val_loss": best_val}, f, indent=2)
        else:
            patience += 1
            if patience >= early_patience:
                print(f"Early stopping at epoch {ep} (best epoch {best_ep})")
                break

    # Save losses
    log = {"train_loss": train_losses, "val_loss": val_losses,
           "best_epoch": best_ep, "best_val_loss": best_val}
    with open(os.path.join(out_dir, "losses.json"), "w") as f:
        json.dump(log, f, indent=2)

    print("\nDone. Artifacts in:", os.path.abspath(out_dir))
