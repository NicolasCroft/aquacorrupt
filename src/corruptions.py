"""
AquaCorrupt: physically-motivated air-water-interface corruptions for benthic imagery.

The idea (see README): rather than test robustness to generic noise, we simulate the
three dominant ways looking at a reef *through the air-water interface from above*
degrades an image:

  1. sun_glint        - specular highlights off wave facets (additive, wave-driven)
  2. refractive_warp  - spatially-varying "wave lens" displacement of the seafloor
  3. water_attenuation- wavelength-dependent absorption + a blue-green scattering veil

Each corruption takes a float image in [0,1], HxWx3 (RGB), a severity in 1..5, and a
seed. Given the same seed + severity the corruption is deterministic, so you get paired
clean/corrupted samples for free.

Grounding: the glint + refractive-lens picture follows the fluid-lensing literature
(Chirayath & Earle 2016; Chirayath & Instrella 2019), where surface waves act as a
time-varying refractive lens and specular glint is the dominant additive artifact.
These are *plausible approximations*, not a radiometric simulator. See README "Honesty".
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# Severity -> parameters. Tuned by eye to go from "barely visible" (1) to "brutal" (5).
_GLINT = {1: 0.12, 2: 0.22, 3: 0.35, 4: 0.50, 5: 0.68}      # peak additive brightness
_WARP  = {1: 1.5,  2: 3.0,  3: 5.5,  4: 8.5,  5: 12.0}      # max pixel displacement
_ATTEN = {1: 0.15, 2: 0.30, 3: 0.45, 4: 0.62, 5: 0.80}      # attenuation strength


def _wave_field(shape, seed, n_waves=6, kmin=2.0, kmax=9.0):
    """A smooth pseudo-random height field = sum of sinusoids with random directions.
    Returns array in roughly [-1, 1]. Used both for glint crests and warp displacement."""
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= max(w, 1); yy /= max(h, 1)
    field = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_waves):
        k = rng.uniform(kmin, kmax)
        theta = rng.uniform(0, 2 * np.pi)
        phase = rng.uniform(0, 2 * np.pi)
        kx, ky = np.cos(theta) * k, np.sin(theta) * k
        field += np.sin(2 * np.pi * (kx * xx + ky * yy) + phase)
    field /= n_waves
    return field  # ~[-1, 1]


def sun_glint(img, severity=3, seed=0):
    """Additive specular highlights: bright wave crests, blurred, added to the image."""
    peak = _GLINT[int(severity)]
    h, w, _ = img.shape
    field = _wave_field((h, w), seed)
    # Keep only the sharp crests (top of the wave field) -> specular streaks.
    crests = np.clip(field, 0.0, None) ** 3
    crests = gaussian_filter(crests, sigma=max(h, w) / 220.0)
    crests /= (crests.max() + 1e-8)
    glint = (crests * peak)[..., None]
    out = img + glint          # additive specular light
    return np.clip(out, 0.0, 1.0)


def refractive_warp(img, severity=3, seed=1):
    """Spatially-varying displacement = the wave acting as a lens over the seafloor."""
    amp = _WARP[int(severity)]
    h, w, _ = img.shape
    dx = _wave_field((h, w), seed) * amp
    dy = _wave_field((h, w), seed + 999) * amp
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords_x = np.clip(xx + dx, 0, w - 1)
    coords_y = np.clip(yy + dy, 0, h - 1)
    coords = np.stack([coords_y.ravel(), coords_x.ravel()])
    out = np.empty_like(img)
    for c in range(3):
        out[..., c] = map_coordinates(
            img[..., c], coords, order=1, mode="reflect"
        ).reshape(h, w)
    return np.clip(out, 0.0, 1.0)


def water_attenuation(img, severity=3, seed=2):
    """Wavelength-dependent absorption (red dies fastest) + blue-green scattering veil."""
    s = _ATTEN[int(severity)]
    # Per-channel transmittance: red attenuates most, blue least.
    trans = np.array([1.0 - 0.9 * s, 1.0 - 0.5 * s, 1.0 - 0.2 * s], dtype=np.float32)
    veil = np.array([0.0, 0.10 * s, 0.18 * s], dtype=np.float32)  # additive scatter
    out = img * trans[None, None, :] + veil[None, None, :]
    # Slight contrast wash from scattering.
    out = 0.5 + (out - 0.5) * (1.0 - 0.25 * s)
    return np.clip(out, 0.0, 1.0)


CORRUPTIONS = {
    "sun_glint": sun_glint,
    "refractive_warp": refractive_warp,
    "water_attenuation": water_attenuation,
}

# The MVP headline: glint only, three severities. Flip to the full grid for the paper.
MVP_CORRUPTION = "sun_glint"
MVP_SEVERITIES = [1, 3, 5]
FULL_SEVERITIES = [1, 2, 3, 4, 5]


def apply_corruption(img, name, severity, seed=0):
    """Dispatch. img: float HxWx3 in [0,1]. Returns same shape/type."""
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption {name!r}; have {list(CORRUPTIONS)}")
    return CORRUPTIONS[name](np.asarray(img, dtype=np.float32), severity, seed)
