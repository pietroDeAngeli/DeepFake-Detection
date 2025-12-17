import os
import json
from sklearn.model_selection import train_test_split

import tools.tools as tools


if __name__ == "__main__":

    # ===== PATHS =====
    dataset_root = "../../dataset/preprocessed"
    real_dir = os.path.join(dataset_root, "real", "videos")
    fake_dir = os.path.join(dataset_root, "fake", "videos")

    # Output JSON
    json_filepath = os.path.join(dataset_root, "manifest.json")
    # =================

    # Get video paths
    reals = tools.get_dir_videos(real_dir)
    fakes = tools.get_dir_videos(fake_dir)

    # Build X (video names) and y (labels)
    X = []
    y = []

    for p in reals:
        X.append(os.path.splitext(os.path.basename(p))[0])
        y.append(1)

    for p in fakes:
        X.append(os.path.splitext(os.path.basename(p))[0])
        y.append(0)

    # First split: train + temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,          # 70% train, 30% temp
        random_state=42,
        stratify=y
    )

    # Second split: validation + test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,          # 15% val, 15% test
        random_state=42,
        stratify=y_temp
    )

    train_data = [{"video": n, "label": l} for n, l in zip(X_train, y_train)]
    val_data   = [{"video": n, "label": l} for n, l in zip(X_val,   y_val)]
    test_data  = [{"video": n, "label": l} for n, l in zip(X_test,  y_test)]

    splits = {
        "train": train_data,
        "val":   val_data,
        "test":  test_data
    }

    # Save JSON
    with open(json_filepath, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved dataset splits to {json_filepath}")
