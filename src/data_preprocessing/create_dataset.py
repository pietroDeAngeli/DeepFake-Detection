import os
import json
import cv2
from pathlib import Path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tqdm import tqdm
import torch

import tools.face_detection as faceDetection
import tools.tools as tools
import tools.motion_vectors as motionVectors
import tools.feature_computation as featureComputation


# ================= CONFIG =================
REAL_DIR = Path("../../dataset/real/videos")
FAKE_DIR = Path("../../dataset/fake/videos")
MODEL_PATH = Path("../../models/face_detection_yunet_2023mar.onnx")
OUT_DIR = Path("../../dataset/preprocessed")
# =========================================


def process_video(video_path, detector):
    """Process a single video and return extracted data or None."""

    feature_matrix, video_faces = tools.feature_computation(detector, video_path)

    return feature_matrix, video_faces


def save_video_output(video_path, features, faces, label):
    video_name = video_path.stem
    save_dir = OUT_DIR / video_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save tensor
    torch.save(
        {"features": features},
        save_dir / "tensors.pt"
    )

    # Save faces
    faces_dir = save_dir / "faces"
    faces_dir.mkdir(exist_ok=True)

    for idx, face in enumerate(faces):
        if face is not None:
            img_path = faces_dir / f"frame_{idx:04d}.jpg"
            cv2.imwrite(str(img_path), face.image)

    # Save metadata
    meta = {
        "video_path": str(video_path),
        "label": label,
        "n_frames": features.shape[0]
    }

    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "real": {
            "path": REAL_DIR,
            "label": True
        },
        "fake": {
            "path": FAKE_DIR,
            "label": False
        }
    }

    detector = faceDetection.initialize_detector(str(MODEL_PATH))

    for split_name, cfg in datasets.items():
        videos = tools.get_dir_videos(cfg["path"])
        label = cfg["label"]

        print(f"\nProcessing {split_name.upper()} videos ({len(videos)})")

        for video_path in tqdm(videos, desc=f"{split_name}"):
            # Skip if already processed
            save_dir = OUT_DIR / video_path.stem
            if save_dir.exists():
                continue

            video_path = Path(video_path)

            result = process_video(video_path, detector)

            if result is None:
                print(f"\nSkipping {video_path.name} (no valid data)")
                continue

            features, faces = result
            save_video_output(video_path, features, faces, label)

    print("\n Preprocessing completed for REAL and FAKE datasets")


if __name__ == "__main__":
    main()
