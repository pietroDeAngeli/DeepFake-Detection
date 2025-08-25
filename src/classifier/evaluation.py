import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import classifier.network as nw
import tools.tools as tools


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

        return tools.load_video_data(video_dir, label, augment_fn=None)


@torch.no_grad()
def evaluate(model, dataloader, device, out_dir="eval_results"):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total = 0
    correct = 0
    loss_sum = 0.0

    tp = tn = fp = fn = 0

    all_labels = []
    all_preds = []

    for (imgs, mvs), label in tqdm(dataloader, desc="Evaluating"):
        imgs  = imgs[0].to(device)
        mvs   = mvs[0].to(device)
        label = label.to(device).view(-1)   # [1]

        logits = model((imgs, mvs))         # [1]
        loss   = criterion(logits, label)
        loss_sum += loss.item()

        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()

        correct += (pred == label).sum().item()
        total += 1

        # confusion matrix counts
        y = int(label.item())
        p = int(pred.item())
        if y == 1 and p == 1: tp += 1
        elif y == 0 and p == 0: tn += 1
        elif y == 0 and p == 1: fp += 1
        elif y == 1 and p == 0: fn += 1

        all_labels.append(y)
        all_preds.append(p)

    acc = correct / total if total > 0 else 0.0
    avg_loss = loss_sum / total if total > 0 else 0.0

    metrics = {
        "accuracy": acc,
        "loss": avg_loss,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total
    }

    # Save results
    os.makedirs(out_dir, exist_ok=True)

    # Confusion matrix as heatmap
    cm = [[tn, fp],
          [fn, tp]]
    df_cm = pd.DataFrame(cm, index=["Real (0)", "Fake (1)"], columns=["Pred Real", "Pred Fake"])
    plt.figure(figsize=(5,4))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    # Save raw confusion matrix
    df_cm.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

    # Save metrics
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    dataset_path  = "../../../dataset"
    manifest_path = os.path.join(dataset_path, "manifest.json")
    checkpoint    = "best_model.pth"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    test_entries = manifest["test"]

    test_ds     = VideoDataset(test_entries, dataset_path)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = nw.MLP(in_channels=3).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    metrics = evaluate(model, test_loader, device)

    print("\nEvaluation Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
