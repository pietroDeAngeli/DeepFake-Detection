import os
import json
from sklearn.model_selection import train_test_split

import tools.tools as tools

if __name__ == "__main__":
    # Original data path
    data_path = "../../FF++"

    # Dataset path
    dataset_path = "../../dataset"

    # JSON file path
    json_filepath = dataset_path + "/manifest.json"

    # Get the videos
    fakes = tools.get_dir_videos(data_path + "/fake")
    reals = tools.get_dir_videos(data_path + "/real")

    # Get the names of the videos
    X = []
    y = []
    for p in fakes:
        X.append(os.path.splitext(os.path.basename(p))[0])
        y.append(0)
    for p in reals:
        X.append(os.path.splitext(os.path.basename(p))[0])
        y.append(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    train_data = [{"video": n, "label": l} for n, l in zip(X_train, y_train)]
    test_data  = [{"video": n, "label": l} for n, l in zip(X_test,  y_test)]

    splits = {
        "train": train_data,
        "test":  test_data
    }

    # 4) Salva su file JSON
    with open(json_filepath, "w") as f:
        json.dump(splits, f, indent=2)

    print("Saved split to splits.json")











