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

import classifier.network as nw  # MLP che wrappa due MobileNetV3_adaption

# -----------------------------
# Reproducibility
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
        self.entries = entries
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry     = self.entries[idx]
        video_dir = os.path.join(self.dataset_path, entry["video"])
        label     = entry["label"]

        # 1) MV+IM features (float32)
        tensor_path = os.path.join(video_dir, "tensors.pt")
        data        = torch.load(tensor_path)
        features    = data["features"]                     # [N, H, W, 3]
        mv_tensor   = features.permute(0, 3, 1, 2).float() # [N, 3, H, W]

        # 2) Face frames RGB -> float32 in [0,1]
        faces_dir = os.path.join(video_dir, "faces")
        img_files = sorted(os.listdir(faces_dir))          # N=100 fissi
        imgs = []
        for fname in img_files:
            img = cv2.imread(os.path.join(faces_dir, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            imgs.append(img)
        imgs_tensor = torch.stack(imgs, dim=0)             # [N, 3, H, W]

        return (imgs_tensor, mv_tensor), torch.tensor([[label]], dtype=torch.float32)

# -----------------------------
# Train / Eval
# -----------------------------
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (imgs, mvs), label in loader:
            imgs  = imgs[0].to(device)
            mvs   = mvs[0].to(device)
            label = label.to(device).view(-1)            # [1]
            logits = model((imgs, mvs))                  # [1] (logits)
            loss   = criterion(logits, label)
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
        logits = model((imgs, mvs))                      # [1] (logits)
        loss   = criterion(logits, label)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))

# -----------------------------
# Main: single split (fixed val = 10%) + 2 fasi (freeze -> unfreeze)
# -----------------------------
if __name__ == "__main__":
    set_seed(42)

    isOnColab = True
    dataset_path  = "/content/drive/MyDrive/DeepFake - Detection/dataset" if isOnColab else "../../dataset"
    manifest_path = os.path.join(dataset_path, "manifest.json")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    all_entries = manifest["train"]

    # split fisso 90/10
    rng = np.random.default_rng(42)
    idx_all = np.arange(len(all_entries))
    rng.shuffle(idx_all)
    val_size = max(1, int(0.10 * len(idx_all)))
    val_idx  = idx_all[:val_size]
    train_idx= idx_all[val_size:]

    full_ds   = VideoDataset(all_entries, dataset_path)
    train_ds  = Subset(full_ds, train_idx)
    val_ds    = Subset(full_ds, val_idx)

    # config
    epochs          = 30
    warmup_epochs   = 5         # fase 1: backbone frozen
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

    # modello: restituisce logits; backbone/head separati nei tuoi file
    model     = nw.MLP(in_channels=3).to(device)  # due rami MobileNetV3_adaption
    criterion = nn.BCEWithLogitsLoss()

    # ---- Fase 1: freeze backbone, allena adapter+head+alpha ----
    model.freeze_backbones(bn_eval=True)  # mette i due backbone in eval e no-grad
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr_head_phase1)

    train_losses, val_losses = [], []
    best_val = math.inf
    best_ep  = -1
    early_patience  = 5
    early_min_delta = 1e-4
    patience = 0
    unfrozen = False

    for ep in range(1, epochs + 1):
        # switch a fase 2: sblocca backbone e usa LR più basso su tutto
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

        # early stopping
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
