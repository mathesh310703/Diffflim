"""Binned MLE: sum the histograms of the k x k neighbourhood of every pixel, then run the per-pixel
Poisson MLE (src/baselines/mle.py) on the summed histograms.

This is the usual way to trade spatial resolution for photons in FLIM software. With k = 3 the
fit sees about 9 times more photons (k = 5: 25 times). The background per bin and the fitted
intensity are scaled accordingly; the returned intensity is per original pixel (divided by k^2).
Image borders are handled by reflection, so every pixel sums exactly k^2 histograms.
"""
import numpy as np
from scipy.ndimage import uniform_filter

from src.baselines.mle import fit_mle


def bin_spatial(counts, k):
    """(T, H, W) counts -> (T, H, W) sums over the k x k neighbourhood (float32)."""
    summed = uniform_filter(counts.astype(np.float32), size=(1, k, k), mode="reflect") * (k * k)
    return np.round(summed)                       # uniform_filter works in float; sums are integers


def fit_binned_mle(counts, cfg, bg_per_bin, k=3, **mle_kwargs):
    tau, inten = fit_mle(bin_spatial(counts, k), cfg, bg_per_bin * k * k, **mle_kwargs)
    return tau, inten / (k * k)
