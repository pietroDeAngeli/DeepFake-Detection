import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

import classifier.network as nw
import tools.tools as tools


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class VideoDataset(Dataset):
    def __init__(self, entries, dataset_path: Path):
        self.entries = entries
        self.dataset_path = Path(dataset_path)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        video_dir = self.dataset_path / entry["video"]
        label = int(entry["label"])
        return tools.load_video_data(str(video_dir), label, augment_fn=None)


def collate_keep_list(batch):
    batch = [b for b in batch if b is not None]
    return batch if len(batch) > 0 else None


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, out_dir="eval_results"):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    loss_sum = 0.0
    total = 0
    correct = 0

    tp = tn = fp = fn = 0

    for batch in tqdm(loader, desc="Evaluating"):
        if batch is None:
            continue

        for (imgs, mvs), label in batch:
            imgs = imgs.to(device, non_blocking=True)   # [N,3,H,W]
            mvs  = mvs.to(device, non_blocking=True)    # [N,3,H,W]
            y    = label.to(device).view(-1)            # [1]

            logits = model((imgs, mvs)).view(-1)        # [1]
            loss = criterion(logits, y)

            prob = torch.sigmoid(logits)
            pred = (prob >= threshold).float()

            correct += (pred == y).sum().item()
            total += 1
            loss_sum += loss.item()

            yi = int(y.item())
            pi = int(pred.item())
            if yi == 1 and pi == 1:
                tp += 1
            elif yi == 0 and pi == 0:
                tn += 1
            elif yi == 0 and pi == 1:
                fp += 1
            elif yi == 1 and pi == 0:
                fn += 1

    acc = correct / total if total else 0.0
    avg_loss = loss_sum / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "loss": float(avg_loss),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total": int(total),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = [[tn, fp],
          [fn, tp]]
    df_cm = pd.DataFrame(cm, index=["Fake (0)", "Real (1)"], columns=["Pred Fake", "Pred Real"])
    df_cm.to_csv(out_dir / "confusion_matrix.csv", index=True)

    plt.figure(figsize=(5, 4))
    plt.imshow(df_cm.values)
    plt.xticks([0, 1], df_cm.columns, rotation=15)
    plt.yticks([0, 1], df_cm.index)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(df_cm.values[i, j]), ha="center", va="center")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    root = project_root()

    dataset_path = root / "dataset" / "preprocessed"
    manifest_path = dataset_path / "manifest.json"

    checkpoint = root / "src" / "classifier" / "run_single_split_ft" / "best_model.pth"

    out_dir = root / "src" / "classifier" / "eval_results"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    test_entries = manifest["test"]

    test_ds = VideoDataset(test_entries, dataset_path)
    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_keep_list,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nw.MLP(in_channels=3).to(device)

    if not checkpoint.is_file():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)

    metrics = evaluate(model, test_loader, device, threshold=0.5, out_dir=str(out_dir))

    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
