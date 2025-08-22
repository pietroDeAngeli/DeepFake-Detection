# evaluation.py
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import classifier.network as nw  # your two-stream MLP
import tools.tools as tools  # for loading video data

class VideoDataset(Dataset):
    def __init__(self, entries, dataset_path):
        """
        entries: list of dicts with keys {"video": <id>, "label": 0/1}
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

        return tools.load_video_data(video_dir, label, augment_fn=None)

@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Minimal evaluation: returns accuracy, avg BCE loss, and confusion matrix counts.
    """
    model.eval()
    criterion = nn.BCELoss()

    total = 0
    correct = 0
    loss_sum = 0.0

    tp = tn = fp = fn = 0

    for (imgs, mvs), label in tqdm(dataloader, desc="Evaluating"):
        # unpack batch size 1: [1,N,3,H,W] -> [N,3,H,W]
        imgs  = imgs[0].to(device)
        mvs   = mvs[0].to(device)
        label = label.to(device).view(-1)   # [1], float32 0/1

        prob = model((imgs, mvs))           # [1], sigmoid already in your model
        loss = criterion(prob, label)
        loss_sum += loss.item()

        pred = (prob >= 0.5).float()        # threshold @ 0.5
        correct += (pred == label).sum().item()
        total += 1

        # confusion matrix counts
        y = int(label.item())
        p = int(pred.item())
        if y == 1 and p == 1: tp += 1
        elif y == 0 and p == 0: tn += 1
        elif y == 0 and p == 1: fp += 1
        elif y == 1 and p == 0: fn += 1

    acc = correct / total if total > 0 else 0.0
    avg_loss = loss_sum / total if total > 0 else 0.0
    return acc, avg_loss, {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total}

if __name__ == "__main__":
    # Paths (adjust as needed)
    dataset_path  = "../../../dataset"
    manifest_path = os.path.join(dataset_path, "manifest.json")
    checkpoint    = "mlp_best.pth"  # path to your trained model

    # Load test split
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    test_entries = manifest["test"]

    # DataLoader
    test_ds     = VideoDataset(test_entries, dataset_path)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = nw.MLP(in_channels=3).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    # Run eval
    acc, avg_loss, cm = evaluate(model, test_loader, device)

    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print(f"Avg BCE Loss:  {avg_loss:.4f}")
    print(f"Confusion Matrix counts: {cm}")
