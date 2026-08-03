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
    corruption/probe steps stay light; backbones.py handles tensor conversion."""

    def __init__(self, root, class_to_idx, size, max_per_class=None):
        self.items = index_split(root, class_to_idx, max_per_class)
        self.size = size
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for path, label in self.items:
            yield load_image(path, self.size), label
