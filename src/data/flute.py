"""Loader for the FLUTE example data (Gottlieb et al. 2023, Zenodo record 8046636, CC BY 4.0).

The files in data/raw/flute/ are multi-page TIFFs with one page per time bin (56 bins of 0.223 ns,
80 MHz repetition rate, i.e. a 12.5 ns period). `load_flute(name)` returns the TCSPC stack as an
integer array (T, H, W) of photon counts.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from src.paths import RAW

FLUTE_DIR = RAW / "flute"
FLUTE_FILES = {
    "fluorescein_embryo": "Fluorescein_Embryo.tif",
    "fluorescein_hmsc": "Fluorescein_hMSC.tif",
    "embryo": "Embryo.tif",
    "hmsc_control": "hMSC control.tif",
    "hmsc_zoom": "hMSC-ZOOM.tif",
    "hmsc_rotenone": "hMSC_rotenone.tif",
}
FLUTE_N_BINS = 56
FLUTE_BIN_NS = 0.223          # stated in the Zenodo record
FLUTE_PERIOD_NS = 12.5        # 80 MHz
FLUORESCEIN_TAU_NS = 4.0      # stated reference lifetime of fluorescein
# Gaussian IRF width estimated from the summed decay of Fluorescein_Embryo.tif with tau fixed at 4.0 ns
# (scripts/flute_fluorescein.py, results/tables/real_flute_irf.csv); used by the FLUTE simulator.
FLUTE_EMBRYO_IRF_SIGMA_NS = 0.150


def load_flute(name, flute_dir=FLUTE_DIR):
    """(T, H, W) int64 photon counts of one FLUTE file (`name` is a key of FLUTE_FILES or a file name)."""
    fname = FLUTE_FILES.get(name, name)
    with Image.open(Path(flute_dir) / fname) as im:
        pages = []
        for i in range(im.n_frames):
            im.seek(i)
            pages.append(np.asarray(im, dtype=np.int64))
    return np.stack(pages, 0)


# ----------------------------------------------------------------------------------------
# Forward model for the FLUTE instrument: periodic (the tail of earlier pulses wraps into the
# window, which covers exactly one period), IRF convolved, integrated over each bin.
# ----------------------------------------------------------------------------------------
SUB = 10   # sub-samples per bin for the bin integration


class FluteConfig:
    """Minimal time-axis description used by the decay model and the phasor."""

    def __init__(self, n_bins=FLUTE_N_BINS, bin_width_ns=FLUTE_BIN_NS):
        self.n_bins = n_bins
        self.bin_width_ns = bin_width_ns
        self.period_ns = n_bins * bin_width_ns          # 12.48 ns: the window is one period
        self.time_starts = np.arange(n_bins) * bin_width_ns
        self.time_centers = self.time_starts + bin_width_ns / 2


def fine_time(cfg):
    return (np.arange(cfg.n_bins * SUB) + 0.5) * cfg.bin_width_ns / SUB


def gaussian_irf_fine(cfg, t0, sigma):
    """Periodically wrapped Gaussian IRF on the fine grid, normalised to sum 1."""
    t = fine_time(cfg)
    p = cfg.period_ns
    irf = sum(np.exp(-0.5 * ((t - t0 + k * p) / sigma) ** 2) for k in (-2, -1, 0, 1, 2))
    return irf / irf.sum()


def bin_fine(x, cfg):
    """(..., n_bins * SUB) -> (..., n_bins) by summing the sub-samples of each bin."""
    return x.reshape(*x.shape[:-1], cfg.n_bins, SUB).sum(-1)


def periodic_decays(tau, irf_fine, cfg):
    """Normalised periodic decays for lifetimes tau (1-D array): (len(tau), n_bins), each row sums to 1.

    The decay exp(-t / tau) of every earlier pulse is included: on one period the sum of all pulses
    is exp(-(t mod P) / tau) / (1 - exp(-P / tau)). Circular convolution with the IRF (FFT), then
    integration over each bin."""
    t = fine_time(cfg)
    tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    e = np.exp(-t[None, :] / tau[:, None]) / (1 - np.exp(-cfg.period_ns / tau[:, None]))
    conv = np.fft.irfft(np.fft.rfft(e, axis=1) * np.fft.rfft(irf_fine)[None, :], n=t.size, axis=1)
    d = bin_fine(np.maximum(conv, 0), cfg)
    return d / d.sum(1, keepdims=True)


class PeriodicDecayModel:
    """decay_fn for src/baselines/mle.py: tau (any shape) -> (n_bins, *shape), by linear interpolation
    in log(tau) on a dense grid (exact model on the grid; avoids an FFT per pixel)."""

    def __init__(self, irf_fine, cfg, tau_bounds=(0.05, 20.0), n=3000):
        self.cfg = cfg
        self.log_grid = np.linspace(np.log(tau_bounds[0]), np.log(tau_bounds[1]), n)
        self.table = periodic_decays(np.exp(self.log_grid), irf_fine, cfg)     # (n, T)

    def __call__(self, tau):
        tau = np.asarray(tau, dtype=np.float64)
        x = np.clip(np.log(tau.ravel()), self.log_grid[0], self.log_grid[-1])
        pos = (x - self.log_grid[0]) / (self.log_grid[1] - self.log_grid[0])
        i = np.clip(pos.astype(int), 0, len(self.log_grid) - 2)
        w = (pos - i)[:, None]
        d = (1 - w) * self.table[i] + w * self.table[i + 1]
        return d.T.reshape(self.cfg.n_bins, *tau.shape)


def rise_time(decay, cfg, frac=0.5):
    """Time (ns) at which the summed decay first crosses frac x (peak - baseline) on its rising edge,
    with linear interpolation between bins; baseline = minimum of the decay. Model-free, used to
    align the IRF of one file to another file."""
    d = np.asarray(decay, dtype=float)
    base = d.min()
    level = base + frac * (d.max() - base)
    k = int(np.argmax(d))
    while k > 0 and d[k - 1] >= level:
        k -= 1
    if k == 0:
        return 0.0
    f = (level - d[k - 1]) / (d[k] - d[k - 1])
    return (k - 1 + f) * cfg.bin_width_ns + cfg.bin_width_ns / 2


def binomial_thin(counts, p, seed):
    """Keep every photon independently with probability p (valid for Poisson counts)."""
    return np.random.default_rng(seed).binomial(counts, p)
