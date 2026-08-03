"""Tiny npz cache for feature tensors. No torch, so the probe/plot steps stay light.

Layout: embeddings/{backbone}__{split}__{corruption}__s{severity}.npz  (arrays X, y).
Clean is stored as corruption='clean', severity=0."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def cache_path(emb_dir, backbone, split, corruption, severity):
    return Path(emb_dir) / f"{backbone}__{split}__{corruption}__s{severity}.npz"


def save(emb_dir, backbone, split, corruption, severity, X, y):
    p = cache_path(emb_dir, backbone, split, corruption, severity)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, X=X, y=y)
    return p


def load(emb_dir, backbone, split, corruption, severity):
    d = np.load(cache_path(emb_dir, backbone, split, corruption, severity))
    return d["X"], d["y"]


def exists(emb_dir, backbone, split, corruption, severity):
    return cache_path(emb_dir, backbone, split, corruption, severity).exists()
