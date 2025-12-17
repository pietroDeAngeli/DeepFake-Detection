import subprocess
from pathlib import Path
import shutil
import sys

# ================= CONFIG =================
OUT_PATH = Path("../../dataset")
COMPRESSION = "c23"
TYPE = "videos"
DOWNLOAD_SCRIPT = "download.py"

REAL_NAME = "real"
FAKE_NAME = "fake"
# =========================================


def run(cmd):
    print(f"\n▶ Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ Command failed")
        sys.exit(result.returncode)


def move_all(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        shutil.move(str(item), dst / item.name)


def main():
    # 1. Create output directory
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Download ORIGINAL (real)
    run([
        "python", DOWNLOAD_SCRIPT,
        str(OUT_PATH),
        "-d", "original",
        "-c", COMPRESSION,
        "-t", TYPE
    ])

    # 3. Download DEEPFAKES (fake)
    run([
        "python", DOWNLOAD_SCRIPT,
        str(OUT_PATH),
        "-d", "Deepfakes",
        "-c", COMPRESSION,
        "-t", TYPE
    ])

    # 4. Restructure directories
    print("\n▶ Restructuring dataset...")

    src_real = OUT_PATH / "original_sequences" / "youtube" / COMPRESSION / TYPE
    src_fake = OUT_PATH / "manipulated_sequences" / "Deepfakes" / COMPRESSION / TYPE

    dst_real = OUT_PATH / REAL_NAME / TYPE
    dst_fake = OUT_PATH / FAKE_NAME / TYPE

    move_all(src_real, dst_real)
    move_all(src_fake, dst_fake)

    print("\n✅ FaceForensics++ download + restructure completed")
    print(f"📁 Real videos → {dst_real}")
    print(f"📁 Fake videos → {dst_fake}")


if __name__ == "__main__":
    main()
