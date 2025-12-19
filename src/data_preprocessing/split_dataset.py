import json
from pathlib import Path
from sklearn.model_selection import train_test_split


SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def project_root() -> Path:
    # .../DeepFake-Detection/src/data_preprocessing/split_dataset.py
    return Path(__file__).resolve().parents[2]


def list_preprocessed_flat(preprocessed_root: Path):
    """
    Reads flat preprocessed structure:
    dataset/preprocessed/<video>/{tensors.pt, meta.json}
    """
    X, y = [], []

    for d in preprocessed_root.iterdir():
        if not d.is_dir():
            continue

        tensor_path = d / "tensors.pt"
        meta_path = d / "meta.json"

        if not tensor_path.is_file() or not meta_path.is_file():
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)

        X.append(d.name)
        y.append(int(meta["label"]))

    return X, y


def main():
    root = project_root()
    preprocessed_root = root / "dataset" / "preprocessed"
    manifest_path = preprocessed_root / "manifest.json"

    X, y = list_preprocessed_flat(preprocessed_root)

    if len(X) == 0:
        raise RuntimeError(
            f"No valid samples found in {preprocessed_root}. "
            f"Expected <video>/tensors.pt and meta.json"
        )

    # Sanity check
    if abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    # Train / temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1.0 - TRAIN_FRAC),
        random_state=SEED,
        stratify=y
    )

    # Val / test
    val_frac_of_temp = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - val_frac_of_temp),
        random_state=SEED,
        stratify=y_temp
    )

    splits = {
        "train": [{"video": v, "label": int(l)} for v, l in zip(X_train, y_train)],
        "val":   [{"video": v, "label": int(l)} for v, l in zip(X_val, y_val)],
        "test":  [{"video": v, "label": int(l)} for v, l in zip(X_test, y_test)],
    }

    with open(manifest_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved manifest to: {manifest_path}")
    print(f"Counts: train={len(X_train)} val={len(X_val)} test={len(X_test)}")


if __name__ == "__main__":
    main()
