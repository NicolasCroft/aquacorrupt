"""
Step 0: get the MVP dataset in place.

Default MVP dataset: Kaggle "Healthy and Bleached Corals Image Classification"
  https://www.kaggle.com/datasets/vencerlanz09/healthy-and-bleached-corals-image-classification

Option A (manual): download + unzip from the page above so you end up with class
subfolders, then point config.TRAIN_DIR / TEST_DIR at the split folders. The exact folder
names in the zip vary, so run this script afterwards to sanity-check what was found.

Option B (Kaggle API): put your kaggle.json in ~/.kaggle/, then run with --download.
    pip install kaggle
    python scripts/00_download_data.py --download

Full-study alternative (documented in README): the NOAA PIFSC ESD Coral Bleaching Dataset
on HuggingFace (NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset). That one is point
annotations on photoquadrats -> crop patches around points and remap label codes.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.data import discover_classes

KAGGLE_SLUG = "vencerlanz09/healthy-and-bleached-corals-image-classification"


def try_download():
    import kaggle  # requires ~/.kaggle/kaggle.json
    dest = config.DATA_DIR / "corals"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_SLUG} -> {dest}")
    kaggle.api.dataset_download_files(KAGGLE_SLUG, path=str(dest), unzip=True)
    print("Done. Inspect the extracted folder layout below.")


def sanity_check():
    print(f"\nDATA_DIR = {config.DATA_DIR}")
    for name, d in [("TRAIN_DIR", config.TRAIN_DIR), ("TEST_DIR", config.TEST_DIR)]:
        d = Path(d)
        if d.exists():
            try:
                classes = discover_classes(d)
                print(f"  {name}: {d}  ->  classes = {classes}")
            except FileNotFoundError as e:
                print(f"  {name}: {d}  ->  no class subfolders yet ({e})")
        else:
            print(f"  {name}: {d}  ->  MISSING. Set config.TRAIN_DIR/TEST_DIR to the "
                  f"split folders inside the unzipped dataset.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="use Kaggle API to fetch")
    args = ap.parse_args()
    if args.download:
        try_download()
    sanity_check()
