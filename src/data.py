"""
Dataset loading for the MVP.

Default: an ImageFolder-style layout (one subfolder per class), which is how the Kaggle
"Healthy and Bleached Corals" set unzips. We return images as float32 RGB arrays in
[0,1] at IMG_SIZE, plus an integer label, so the corruption functions can act on them
*before* the backbone-specific normalization is applied.

Swap in the NOAA PIFSC ESD Coral Bleaching Dataset for the full study: it is point
annotations on photoquadrats, so you would crop fixed-size patches around each annotated
point and map the label codes to {healthy, bleached, ...}. That is more work than the MVP
needs, hence the Kaggle default.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def discover_classes(root: Path):
    """Return sorted class names = immediate subdirectories that contain images."""
    root = Path(root)
    classes = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(f.suffix.lower() in IMG_EXTS for f in d.rglob("*")):
            classes.append(d.name)
    if not classes:
        raise FileNotFoundError(
            f"No class subfolders with images under {root}. "
            f"Expected e.g. {root}/healthy_corals/*.jpg (see scripts/00_download_data.py)."
        )
    return classes


def index_split(root: Path, class_to_idx: dict, max_per_class: int | None = None):
    """List (path, label) pairs for a split root, capped at max_per_class."""
    root = Path(root)
    items = []
    for cls, idx in class_to_idx.items():
        files = [f for f in sorted((root / cls).rglob("*")) if f.suffix.lower() in IMG_EXTS]
        if max_per_class:
            files = files[:max_per_class]
        items += [(f, idx) for f in files]
    return items


def load_image(path: Path, size: int) -> np.ndarray:
    """Load -> RGB -> resize -> float32 [0,1], shape (size, size, 3)."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


class ArrayImageFolder:
    """Tiny iterable over (float_rgb_array, label). No torch dependency here so the
    corruption/probe steps stay light; backbones.py handles tensor conversion.
    Construct from an explicit (path, label) list, or via .from_dir()."""

    def __init__(self, items, size):
        self.items = items
        self.size = size

    @classmethod
    def from_dir(cls, root, class_to_idx, size, max_per_class=None):
        return cls(index_split(root, class_to_idx, max_per_class), size)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for path, label in self.items:
            yield load_image(path, self.size), label


# --- layout handling: Kaggle zips vary, so figure out train/test ourselves -----

_TRAIN_NAMES = {"train", "training"}
_TEST_NAMES = {"valid", "val", "test", "testing", "validation"}


def _dirs(p: Path):
    return [d for d in p.iterdir() if d.is_dir()] if p.exists() else []


def find_split_dirs(base: Path):
    """Search base and one level down for train/ and valid|test/ folders.
    Returns (train_dir | None, test_dir | None)."""
    base = Path(base)
    train = test = None
    for c in [base, *_dirs(base)]:
        for d in [c, *_dirs(c)]:
            n = d.name.lower()
            if n in _TRAIN_NAMES and train is None:
                train = d
            if n in _TEST_NAMES and test is None:
                test = d
    return train, test


def _has_class_dirs(p: Path):
    try:
        discover_classes(p)
        return True
    except FileNotFoundError:
        return False


def _find_single_root(base: Path):
    """Find a directory that directly holds class subfolders (no train/test split)."""
    base = Path(base)
    for c in [base, *_dirs(base)]:
        if _has_class_dirs(c):
            return c
    return None


def stratified_split(items, frac=0.8, seed=0):
    """Deterministic per-class split of an (path,label) list into (train, test)."""
    import random
    by_label = {}
    for it in items:
        by_label.setdefault(it[1], []).append(it)
    train, test = [], []
    for label, group in sorted(by_label.items()):
        rng = random.Random(seed + label)
        group = sorted(group)
        rng.shuffle(group)
        k = max(1, int(len(group) * frac))
        train += group[:k]
        test += group[k:]
    return train, test


def prepare_datasets(data_dir, configured_train, configured_test, size,
                     max_per_class=None, split_frac=0.8, seed=0):
    """Return (train_ds, test_ds, class_to_idx, note) handling three layouts:
      1. configured_train / configured_test both valid  -> use as-is
      2. a train/ + valid|test/ split found under data_dir -> use those
      3. a single folder of class subdirs -> deterministic stratified split
    Raises FileNotFoundError with guidance if none apply."""
    ct, cte = Path(configured_train), Path(configured_test)
    if _has_class_dirs(ct) and _has_class_dirs(cte):
        train_dir, test_dir, note = ct, cte, "using configured TRAIN_DIR / TEST_DIR"
    else:
        train_dir, test_dir = find_split_dirs(data_dir)
        if train_dir and test_dir:
            note = f"auto-detected split: {train_dir}  |  {test_dir}"
        else:
            single = _find_single_root(data_dir) or (ct if _has_class_dirs(ct) else None)
            if single is None:
                raise FileNotFoundError(
                    f"Could not find a usable dataset under {data_dir}. Expected either "
                    f"train/ + valid|test/ folders, or one folder of class subdirs. "
                    f"Run scripts/00_download_data.py and check the reported layout, then "
                    f"set config.TRAIN_DIR / TEST_DIR."
                )
            classes = discover_classes(single)
            class_to_idx = {c: i for i, c in enumerate(classes)}
            items = index_split(single, class_to_idx, max_per_class)
            tr, te = stratified_split(items, split_frac, seed)
            note = (f"single folder {single}: deterministic {int(split_frac*100)}/"
                    f"{int((1-split_frac)*100)} split ({len(tr)} train / {len(te)} test)")
            return (ArrayImageFolder(tr, size), ArrayImageFolder(te, size),
                    class_to_idx, note)

    classes = discover_classes(train_dir)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_ds = ArrayImageFolder.from_dir(train_dir, class_to_idx, size, max_per_class)
    test_ds = ArrayImageFolder.from_dir(test_dir, class_to_idx, size, max_per_class)
    return train_ds, test_ds, class_to_idx, note
