import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

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

        # 1) Load MV+IM features saved as tensors.pt (float32)
        tensor_path = os.path.join(video_dir, "tensors.pt")
        features    = torch.load(tensor_path)["features"]       # [N, H, W, 3], float32
        mv_tensor   = features.permute(0, 3, 1, 2).float()      # [N, 3, H, W]

        # 2) Load cropped face images (RGB uint8) and convert to float32
        faces_dir = os.path.join(video_dir, "faces")
        img_files = sorted(os.listdir(faces_dir))
        imgs = []
        for fname in img_files:
            img = cv2.imread(os.path.join(faces_dir, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).float()    # [3, H, W]
            img = img.div(255.0)                                   # normalize to [0,1]
            imgs.append(img)
        imgs_tensor = torch.stack(imgs, dim=0)                    # [N, 3, H, W]

        # Return ((rgb_frames, mv_frames), label)
        return (imgs_tensor, mv_tensor), torch.tensor([[label]], dtype=torch.float32)

if __name__ == "__main__":
    # Paths
    dataset_path  = "../../dataset"
    manifest_path = os.path.join(dataset_path, "manifest.json")

    # Read train/test split from manifest.json
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    train_entries = manifest["train"]
    test_entries  = manifest["test"]

    # Create datasets and dataloaders
    train_ds     = VideoDataset(train_entries, dataset_path)
    test_ds      = VideoDataset(test_entries,  dataset_path)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=4)

    # Initialize model, loss, optimizer
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = nw.MLP(in_channels=3).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for (imgs, mvs), label in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            # imgs & mvs: [1, N, 3, H, W] → remove leading batch dimension
            imgs  = imgs[0].to(device)   # [N, 3, H, W]
            mvs   = mvs[0].to(device)    # [N, 3, H, W]

            # label: originally shape [1,1], squeeze to [1]
            label = label.to(device).view(-1)  # now [1]

            optimizer.zero_grad()
            output = model((imgs, mvs))       # output: [1]
            loss   = criterion(output, label) # both are [1] now
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs}  Loss: {avg_loss:.4f}")

    # Save the best model
    torch.save(model.state_dict(), "mlp_best.pth")
