"""
Regression tests for dataset-layout detection.

These exist because of a bug that produced a CONVINCING WRONG RESULT rather than a crash.
discover_classes() used rglob to decide whether a subdirectory was a class, so any
ancestor of the real class folders matched. For the Kaggle corals layout --
    data/corals/{bleached_corals,healthy_corals}/*.jpg
-- passing data/ resolved to classes == ['corals'], every image got label 0, and the
linear probe scored ~100% on a one-class problem while exiting 0.

smoke_test.py cannot catch this: it builds a synthetic tree already at the correct depth.
The failure only appears when a dataset root sits one level above the class folders,
which is exactly how the real dataset unzips.

    python -m pytest tests/ -q       (or: python tests/test_data_layout.py)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import discover_classes, prepare_datasets  # noqa: E402


def _img(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(p)


def _corals_layout(root: Path, n=3):
    """Mimic the real unzip: root/corals/<class>/*.jpg, class dirs one level down."""
    for cls in ("bleached_corals", "healthy_corals"):
        for i in range(n):
            _img(root / "corals" / cls / f"{i}.jpg")


def test_nested_root_does_not_collapse_to_one_class(tmp_path):
    """THE regression: data/ must not report ['corals'] as a single class."""
    _corals_layout(tmp_path)
    assert discover_classes(tmp_path / "corals") == ["bleached_corals", "healthy_corals"]
    # data/corals holds no images DIRECTLY, so it is not itself a class dir.
    try:
        found = discover_classes(tmp_path)
    except FileNotFoundError:
        found = []
    assert found != ["corals"], "regressed: ancestor dir detected as a single class"


def test_prepare_datasets_finds_two_classes_from_parent(tmp_path):
    """End-to-end: pointing at the parent still yields both classes and a clean split."""
    _corals_layout(tmp_path, n=10)
    train, test, class_to_idx, note = prepare_datasets(
        tmp_path, tmp_path / "corals" / "train", tmp_path / "corals" / "valid",
        size=8, max_per_class=None, seed=0)
    assert set(class_to_idx) == {"bleached_corals", "healthy_corals"}, class_to_idx
    assert len(class_to_idx) == 2

    labels = {l for _, l in train.items} | {l for _, l in test.items}
    assert labels == {0, 1}, f"probe would be degenerate: labels={labels}"

    overlap = {p for p, _ in train.items} & {p for p, _ in test.items}
    assert not overlap, f"train/test leak: {len(overlap)} shared images"
    assert len(train) + len(test) == 20


def test_explicit_split_dirs_still_win(tmp_path):
    """A real pre-split dataset must still use its own train/ and valid/ folders."""
    for split in ("train", "valid"):
        for cls in ("a", "b"):
            for i in range(4):
                _img(tmp_path / "ds" / split / cls / f"{i}.jpg")
    train, test, class_to_idx, note = prepare_datasets(
        tmp_path / "ds", tmp_path / "ds" / "train", tmp_path / "ds" / "valid",
        size=8, max_per_class=None, seed=0)
    assert set(class_to_idx) == {"a", "b"}
    assert len(train) == 8 and len(test) == 8
    assert "configured" in note or "auto-detected" in note


if __name__ == "__main__":
    import tempfile
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(Path(d))
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print("ALL TESTS PASSED" if not fails else f"{fails} TEST(S) FAILED")
    sys.exit(1 if fails else 0)
