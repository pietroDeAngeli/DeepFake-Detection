# evaluation.py

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm

import classifier.network as nw

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

        # Load MV+IM features saved as tensors.pt (float32)
        tensor_path = os.path.join(video_dir, "tensors.pt")
        features    = torch.load(tensor_path)["features"]      # [N, H, W, 3], float32
        mv_tensor   = features.permute(0, 3, 1, 2).float()     # [N, 3, H, W]

        # Load cropped face images (RGB uint8) and convert to float32
        faces_dir = os.path.join(video_dir, "faces")
        img_files = sorted(os.listdir(faces_dir))
        imgs = []
        for fname in img_files:
            img = cv2.imread(os.path.join(faces_dir, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).float()  # [3, H, W]
            img = img.div(255.0)                                  # normalize to [0,1]
            imgs.append(img)
        imgs_tensor = torch.stack(imgs, dim=0)                   # [N, 3, H, W]

        # Return ((rgb_frames, mv_frames), label)
        return (imgs_tensor, mv_tensor), torch.tensor([label], dtype=torch.float32)

def evaluate(model, dataloader, device):
    """
    Run evaluation on the test set and return accuracy.
    """
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for (imgs, mvs), label in tqdm(dataloader, desc="Evaluating"):
            # remove leading batch dimension ([1, N, C, H, W] -> [N, C, H, W])
            imgs  = imgs[0].to(device)
            mvs   = mvs[0].to(device)
            label = label.to(device).view(-1)  # shape [1]

            output = model((imgs, mvs))        # [1]
            pred   = (output >= 0.5).float()   # threshold at 0.5
            correct += (pred == label).sum().item()
            total   += 1

    return correct / total if total > 0 else 0.0

if __name__ == "__main__":
    # Paths
    dataset_path  = "../../dataset"
    manifest_path = os.path.join(dataset_path, "manifest.json")
    checkpoint    = "mlp_best.pth"  # or your desired checkpoint

    # Load train/test split
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    test_entries = manifest["test"]

    # Create test DataLoader
    test_ds     = VideoDataset(test_entries, dataset_path)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    # Initialize and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = nw.MLP(in_channels=3).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    # Run evaluation
    accuracy = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
