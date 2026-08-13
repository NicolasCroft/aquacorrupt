"""
Augmentation probes -- diagnostics, NOT part of the AquaCorrupt physical corruption suite.

Why these exist. The full grid found exactly one surviving effect: MAE is less robust to
water_attenuation than DINOv2 and the supervised ViT. That is confounded, because
water_attenuation is photometric and the two winners both pretrain with heavy photometric
augmentation (DINOv2: color jitter, blur, solarization; AugReg supervised: RandAugment,
which includes color/brightness/contrast/solarize) while MAE pretrains with
random-resized-crop and horizontal flip essentially alone.

So the question is whether robustness tracks the PRETRAINING OBJECTIVE or merely
AUGMENTATION EXPOSURE. These three transforms are chosen to sit squarely inside the
DINOv2 / RandAugment augmentation distributions and outside MAE's:

  solarize   - literally a DINOv2 (and BYOL) augmentation
  grayscale  - random grayscale is a DINO/DINOv2 augmentation
  hue_shift  - the hue axis of standard color jitter

Pre-registered predictions:
  augmentation hypothesis -> MAE's deficit here is LARGER than its 0.056 deficit on
                             water_attenuation; dinov2 - supervised stays ~0.
  objective hypothesis    -> no reason for a reconstructive objective to be specially
                             fragile to solarization; deficits look like the physical grid.

Keeping these out of corruptions.CORRUPTIONS matters: that dict is the paper's artifact and
should stay physically motivated. They are merged only into the dispatch registry so the
existing extraction pipeline can run them.

Same interface as the corruptions: float HxWx3 in [0,1], severity 1..5, seed, returns the
same shape. All three are deterministic; `seed` is accepted for interface parity.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# Rec.709 luminance, used for the grayscale probe.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

_SOLARIZE_THRESH = {1: 0.85, 2: 0.70, 3: 0.55, 4: 0.42, 5: 0.30}  # lower = more inverted
_GRAY_BLEND = {1: 0.20, 2: 0.40, 3: 0.60, 4: 0.80, 5: 1.00}       # 1.0 = fully gray
_HUE_DEG = {1: 20.0, 2: 45.0, 3: 75.0, 4: 110.0, 5: 150.0}        # degrees of rotation


def solarize(img, severity=3, seed=0):
    """Invert every pixel above a threshold. A DINOv2/BYOL augmentation."""
    t = _SOLARIZE_THRESH[int(severity)]
    out = np.where(img >= t, 1.0 - img, img)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def grayscale(img, severity=3, seed=0):
    """Blend toward Rec.709 luminance. Random grayscale is a DINO/DINOv2 augmentation."""
    a = _GRAY_BLEND[int(severity)]
    gray = (img * _LUMA[None, None, :]).sum(axis=2, keepdims=True)
    return np.clip(img * (1.0 - a) + gray * a, 0.0, 1.0).astype(np.float32)


def hue_shift(img, severity=3, seed=0):
    """Rotate hue, leaving saturation and value alone. The hue axis of color jitter."""
    deg = _HUE_DEG[int(severity)]
    hsv = rgb_to_hsv(np.clip(img, 0.0, 1.0))
    hsv[..., 0] = np.mod(hsv[..., 0] + deg / 360.0, 1.0)
    return np.clip(hsv_to_rgb(hsv), 0.0, 1.0).astype(np.float32)


AUG_PROBES = {
    "solarize": solarize,
    "grayscale": grayscale,
    "hue_shift": hue_shift,
}
