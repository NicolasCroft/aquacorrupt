"""
Step 4 (full grid): extract frozen features for EVERY image under EVERY condition.

Why this exists separately from 01_extract_features.py: that script caches a fixed
738/185 split, which leaves only 185 held-out predictions to judge "do the curves
separate" on -- far too few (bootstrap CIs came out ~5x wider than the effect). Here we
embed all N images under clean + each (corruption, severity), so 05_analyze_grid.py can
run k-fold CV and get a held-out prediction for every image, tightening CIs ~sqrt(5)x.

Cache split name is 'all'. Resumable: existing conditions are skipped.

    python scripts/04_extract_full_grid.py --device mps
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.data import prepare_datasets, ArrayImageFolder
from src.backbones import load_backbone
from src.features import extract
from src import cache
from src.corruptions import CORRUPTIONS, FULL_SEVERITIES


def all_items_dataset():
    """The same images 01 uses, as one deterministic ordered set (train + test)."""
    train, test, class_to_idx, note = prepare_datasets(
        config.DATA_DIR, config.TRAIN_DIR, config.TEST_DIR,
        config.IMG_SIZE, config.MAX_PER_CLASS, seed=config.SEED)
    items = sorted(train.items + test.items, key=lambda t: str(t[0]))
    return ArrayImageFolder(items, config.IMG_SIZE), class_to_idx, note


def main(device, backbones, corruptions, severities):
    ds, class_to_idx, note = all_items_dataset()
    print(f"dataset: {note}")
    print(f"classes: {class_to_idx}")
    print(f"all images: {len(ds)}")
    conditions = [("clean", 0)] + [(c, s) for c in corruptions for s in severities]
    print(f"conditions per backbone: {len(conditions)}  "
          f"({len(corruptions)} corruptions x {len(severities)} severities + clean)")

    for bb in backbones:
        todo = [(c, s) for c, s in conditions
                if not cache.exists(config.EMB_DIR, bb, "all", c, s)]
        if not todo:
            print(f"\n=== {bb}: all {len(conditions)} conditions cached, skipping ===")
            continue
        print(f"\n=== backbone: {bb} (device={device}) -- {len(todo)} to do ===")
        t0 = time.time()
        model, tf, embed = load_backbone(bb, img_size=config.IMG_SIZE, device=device)
        for c, s in todo:
            ts = time.time()
            X, y = extract(ds, model, tf, embed, device, config.BATCH_SIZE,
                           corruption=None if c == "clean" else c,
                           severity=s, seed=config.SEED)
            cache.save(config.EMB_DIR, bb, "all", c, s, X, y)
            print(f"  {c} s{s}: {X.shape} in {time.time()-ts:.0f}s", flush=True)
        del model
        print(f"  {bb} done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backbones", nargs="*", default=config.MVP_BACKBONES)
    ap.add_argument("--corruptions", nargs="*", default=list(CORRUPTIONS))
    ap.add_argument("--severities", nargs="*", type=int, default=FULL_SEVERITIES)
    args = ap.parse_args()
    main(args.device, args.backbones, args.corruptions, args.severities)
