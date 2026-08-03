"""
Step 1 (the only GPU-worthy step): extract frozen features.

For each backbone we cache:
  clean train, clean test, and corrupted test at each severity.
Resumable: anything already cached is skipped. Run this on GPU (Colab notebook provided)
or on CPU if you are patient and MAX_PER_CLASS is small.

    python scripts/01_extract_features.py
    python scripts/01_extract_features.py --device cuda --backbones dinov2 mae
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.data import discover_classes, ArrayImageFolder
from src.backbones import load_backbone
from src.features import extract
from src import cache


def main(device, backbones, corruption, severities):
    classes = discover_classes(config.TRAIN_DIR)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes: {class_to_idx}")

    train = ArrayImageFolder(config.TRAIN_DIR, class_to_idx, config.IMG_SIZE, config.MAX_PER_CLASS)
    test = ArrayImageFolder(config.TEST_DIR, class_to_idx, config.IMG_SIZE, config.MAX_PER_CLASS)
    print(f"train={len(train)} test={len(test)} images")

    for bb in backbones:
        print(f"\n=== backbone: {bb} (device={device}) ===")
        t0 = time.time()
        model, tf, embed = load_backbone(bb, img_size=config.IMG_SIZE, device=device)

        # clean train
        if not cache.exists(config.EMB_DIR, bb, "train", "clean", 0):
            X, y = extract(train, model, tf, embed, device, config.BATCH_SIZE)
            cache.save(config.EMB_DIR, bb, "train", "clean", 0, X, y)
            print(f"  clean train cached: {X.shape}")

        # clean test
        if not cache.exists(config.EMB_DIR, bb, "test", "clean", 0):
            X, y = extract(test, model, tf, embed, device, config.BATCH_SIZE)
            cache.save(config.EMB_DIR, bb, "test", "clean", 0, X, y)
            print(f"  clean test cached: {X.shape}")

        # corrupted test
        for s in severities:
            if cache.exists(config.EMB_DIR, bb, "test", corruption, s):
                continue
            X, y = extract(test, model, tf, embed, device, config.BATCH_SIZE,
                           corruption=corruption, severity=s, seed=config.SEED)
            cache.save(config.EMB_DIR, bb, "test", corruption, s, X, y)
            print(f"  {corruption} s{s} test cached: {X.shape}")

        del model
        print(f"  backbone done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--backbones", nargs="*", default=config.MVP_BACKBONES)
    ap.add_argument("--corruption", default=config.CORRUPTION)
    ap.add_argument("--severities", nargs="*", type=int, default=config.SEVERITIES)
    args = ap.parse_args()
    main(args.device, args.backbones, args.corruption, args.severities)
