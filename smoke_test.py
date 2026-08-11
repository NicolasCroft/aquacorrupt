"""
Offline smoke test: proves the pipeline runs end-to-end with REAL torch, without any
downloads. It uses a tiny randomly-initialized model and synthetic images, so the numbers
are meaningless -- the point is that features.extract -> probe -> plotting all work in your
environment before you spend time pulling the dataset and the real backbones.

    python smoke_test.py

Passing means torch, the corruption path, the probe, and plotting are wired correctly.
It says NOTHING about the research question.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.features import extract
from src import cache, probe
from src.plotting import plot_curves

SMOKE_DIR = Path("results/_smoke")
BB = "smoke_tiny_net"
CORR, SEVS = "sun_glint", [1, 3, 5]


class TinyNet(nn.Module):
    """A few conv layers + global pool -> a feature vector. Random weights."""
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, d, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class SyntheticImages:
    """Iterable of (float_rgb_array[H,W,3] in [0,1], label). Class 0 leans red,
    class 1 leans blue, plus noise -> a linear probe should beat chance."""
    def __init__(self, n, size=64, seed=0):
        self.n, self.size, self.seed = n, size, seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n):
            label = int(rng.integers(0, 2))
            img = rng.uniform(0, 0.4, (self.size, self.size, 3)).astype("float32")
            img[..., 0 if label == 0 else 2] += 0.5  # class signal in R or B
            yield np.clip(img, 0, 1), label


def light_tf(pil):
    a = np.asarray(pil).astype("float32") / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


@torch.no_grad()
def embed(model, x):
    out = model(x)
    if out.ndim == 3:
        out = out.mean(dim=1)
    return out.float().cpu().numpy()


def main():
    torch.manual_seed(0)
    model = TinyNet().eval()
    train = SyntheticImages(80, seed=1)
    test = SyntheticImages(60, seed=2)

    Xtr, ytr = extract(train, model, light_tf, embed, batch_size=16)
    cache.save(SMOKE_DIR, BB, "train", "clean", 0, Xtr, ytr)
    Xte, yte = extract(test, model, light_tf, embed, batch_size=16)
    cache.save(SMOKE_DIR, BB, "test", "clean", 0, Xte, yte)
    for s in SEVS:
        Xs, ys = extract(test, model, light_tf, embed, batch_size=16,
                         corruption=CORR, severity=s, seed=0)
        cache.save(SMOKE_DIR, BB, "test", CORR, s, Xs, ys)

    r = probe.evaluate_backbone(SMOKE_DIR, BB, CORR, SEVS)
    fig = plot_curves([r], CORR, SMOKE_DIR / "smoke_curve.png")

    print(f"clean_acc={r['clean_acc']:.3f}  acc={ {s: round(a,3) for s,a in r['acc'].items()} }")
    print(f"feature dim = {Xtr.shape[1]}, train/test = {len(ytr)}/{len(yte)}")
    print(f"SMOKE TEST PASSED. figure -> {fig}")
    print("(numbers are meaningless: random model + synthetic data. plumbing works.)")


if __name__ == "__main__":
    main()
