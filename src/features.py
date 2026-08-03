"""
Extract and cache frozen features.

For a linear-probe robustness study we need:
  - clean TRAIN features   (to fit the probe)
  - clean TEST features     (baseline accuracy)
  - corrupted TEST features (accuracy vs severity)

We do NOT corrupt train: the probe learns on clean features and we measure how well those
frozen features survive corrupted inputs. This is the ImageNet-C style protocol.

Cache layout:  embeddings/{backbone}__{split}__{corruption}__s{severity}.npz  with X, y.
Clean is stored as corruption='clean', severity=0.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from .corruptions import apply_corruption
from .cache import cache_path, save, load  # re-exported for convenience


def _to_pil(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"))


def _batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def extract(dataset, model, tf, embed, device="cpu", batch_size=64,
            corruption=None, severity=0, seed=0):
    """Run the backbone over the dataset, optionally corrupting each image first.
    Returns (X [N,D] float32, y [N] int64)."""
    feats, labels = [], []
    for i, batch in enumerate(_batched(iter(dataset), batch_size)):
        tensors = []
        for j, (img, label) in enumerate(batch):
            if corruption is not None:
                img = apply_corruption(img, corruption, severity, seed=seed + i * 100000 + j)
            tensors.append(tf(_to_pil(img)))
            labels.append(label)
        x = torch.stack(tensors)
        feats.append(embed(model, x))
    X = np.concatenate(feats).astype("float32")
    y = np.asarray(labels, dtype="int64")
    return X, y
