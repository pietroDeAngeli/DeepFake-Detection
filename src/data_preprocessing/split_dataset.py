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


def safe_read_json(path: Path):
    """Return dict if JSON is valid, else None."""
    try:
        txt = path.read_text(encoding="utf-8").strip()
        if not txt:
            return None
        return json.loads(txt)
    except Exception:
        return None


def list_preprocessed_flat(preprocessed_root: Path):
    """
    Expects:
      dataset/preprocessed/<video_id>/tensors.pt
      dataset/preprocessed/<video_id>/meta.json  (contains 'label')
    Returns:
      X: list[str] video_id
      y: list[int] labels (0/1)
      bad: list[str] skipped video_id
    """
    X, y = [], []
    bad = []

    for d in sorted(preprocessed_root.iterdir()):
        if not d.is_dir():
            continue

        tensor_path = d / "tensors.pt"
        meta_path = d / "meta.json"

        if not tensor_path.is_file():
            continue

        meta = safe_read_json(meta_path)
        if meta is None or "label" not in meta:
            bad.append(d.name)
            continue

        try:
            label = int(meta["label"])
        except Exception:
            bad.append(d.name)
            continue

        X.append(d.name)
        y.append(label)

    return X, y, bad


def main():
    root = project_root()
    preprocessed_root = root / "dataset" / "preprocessed"
    manifest_path = preprocessed_root / "manifest.json"

    if abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) > 1e-6:
        raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must sum to 1.0")

    X, y, bad = list_preprocessed_flat(preprocessed_root)

    if len(X) == 0:
        raise RuntimeError(
            f"No valid samples found in {preprocessed_root}. "
            f"Need <video_id>/tensors.pt and valid <video_id>/meta.json with 'label'."
        )

    # Check class counts for stratify
    n0 = sum(1 for v in y if v == 0)
    n1 = sum(1 for v in y if v == 1)
    if n0 < 2 or n1 < 2:
        raise RuntimeError(
            f"Not enough samples per class for stratified split. "
            f"Counts: label0={n0}, label1={n1}. "
            f"Fix preprocessing / labels first."
        )

    # train + temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1.0 - TRAIN_FRAC),
        random_state=SEED,
        stratify=y
    )

    # val + test from temp
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
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved manifest to: {manifest_path}")
    print(f"Usable samples: {len(X)} (label0={n0}, label1={n1})")
    print(f"Split sizes: train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    if bad:
        print(f"Skipped {len(bad)} samples due to invalid meta.json (first 20): {bad[:20]}")


if __name__ == "__main__":
    main()
