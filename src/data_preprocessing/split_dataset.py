import json
from pathlib import Path
from sklearn.model_selection import train_test_split


SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def project_root() -> Path:
    # file: .../src/data_preprocessing/split_dataset.py
    return Path(__file__).resolve().parents[2]


def list_preprocessed_video_ids(class_dir: Path, class_name: str) -> list[str]:
    """
    Returns relative ids like: "real/<video_name>" or "fake/<video_name>"
    Only keeps dirs containing tensors.pt
    """
    if not class_dir.exists():
        return []

    ids = []
    for d in class_dir.iterdir():
        if not d.is_dir():
            continue
        if (d / "tensors.pt").is_file():
            ids.append(f"{class_name}/{d.name}")
    return sorted(ids)


def main():
    root = project_root()
    preprocessed_root = root / "dataset" / "preprocessed"

    real_dir = preprocessed_root / "real"
    fake_dir = preprocessed_root / "fake"

    manifest_path = preprocessed_root / "manifest.json"

    # Collect preprocessed samples
    real_ids = list_preprocessed_video_ids(real_dir, "real")
    fake_ids = list_preprocessed_video_ids(fake_dir, "fake")

    if len(real_ids) == 0 and len(fake_ids) == 0:
        raise RuntimeError(
            f"No preprocessed samples found in {preprocessed_root}. "
            f"Expected structure: dataset/preprocessed/real/<video>/tensors.pt and fake/<video>/tensors.pt"
        )

    X = real_ids + fake_ids
    y = [1] * len(real_ids) + [0] * len(fake_ids)

    # Sanity on fractions
    if abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) > 1e-6:
        raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must sum to 1.0")

    # Split: train + temp
    temp_frac = 1.0 - TRAIN_FRAC
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_frac,
        random_state=SEED,
        stratify=y
    )

    # Split: val + test from temp
    # val_frac_of_temp = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
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

    preprocessed_root.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved manifest to: {manifest_path}")
    print(f"Counts: real={len(real_ids)} fake={len(fake_ids)} | "
          f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


if __name__ == "__main__":
    main()
