"""Simulate TCSPC FLIM data with exact ground truth for lifetime (tau) and intensity.

Physical model, per pixel (x, y):

    expected[t] = intensity[x, y] * decay(t; tau[x, y]) + background

    decay(t; tau)  = (Gaussian IRF  *  exp(-t / tau)), integrated over each time bin
                     and normalised so that it sums to 1 over the 256 bins.
    intensity      = expected number of SIGNAL photons in the pixel (ground truth).
    background     = constant expected counts per bin (same for every pixel).

    counts[t, x, y] ~ Poisson(expected[t, x, y])          -> int array, shape (T, H, W)

So for every pixel:  E[counts.sum(axis=0)] = intensity + background * T.

Photon level: `photons` = mean signal photons per FOREGROUND (cell) pixel. The foreground
mask marks pixels whose cell signal is clearly above the dim floor between cells.

Simplifications (deliberate; the MLE baseline uses exactly the same model):
  * mono-exponential decays only,
  * no pile-up from the previous laser pulse (no wrap-around of long decays),
  * IRF is a Gaussian at a fixed position (irf_center_ns),
  * the part of the decay beyond the 25 ns window is dropped (decay is renormalised
    inside the window), so `intensity` is exactly the photons you can measure.

Usage:
    python -m src.data.simulate            # build the fixed test set in data/raw/own/sim_test/
    python -m src.data.simulate --force    # rebuild it (overwrites)
"""
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import log_ndtr, ndtr

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ~ 1 / 2.355

# Photon levels (mean signal photons per foreground pixel) used for the fixed test set.
TEST_PHOTON_LEVELS = (10, 20, 50, 100, 500, 1000)


@dataclass
class SimConfig:
    """All simulation settings in one place (saved into every dataset's metadata)."""

    size: int = 64                # image is size x size pixels
    n_bins: int = 256             # TCSPC time bins
    bin_width_ns: float = 0.0977  # 256 * 0.0977 ns = 25 ns window (40 MHz), like FLIMngo
    irf_fwhm_ns: float = 0.2      # Gaussian IRF width
    irf_center_ns: float = 1.0    # where the laser pulse arrives in the window
    tau_min: float = 0.5          # lifetime range of the ground-truth maps (ns)
    tau_max: float = 5.0
    bg_fraction: float = 0.02     # total background photons per pixel = bg_fraction * photons
    n_blobs_min: int = 3          # number of "cells" per image
    n_blobs_max: int = 8
    floor: float = 0.05           # relative brightness between the cells (dim autofluorescence)
    fg_threshold: float = 0.25    # foreground = cell signal > this (5x the floor; cells are 0.5-1.5)

    @property
    def time_edges(self):
        """Bin edges in ns, shape (T + 1,)."""
        return np.arange(self.n_bins + 1) * self.bin_width_ns

    @property
    def time_centers(self):
        """Bin centres in ns, shape (T,)."""
        return (np.arange(self.n_bins) + 0.5) * self.bin_width_ns

    @property
    def period_ns(self):
        """Laser repetition period = full TCSPC window (25 ns at 40 MHz)."""
        return self.n_bins * self.bin_width_ns


# ----------------------------------------------------------------------------------------
# Decay model
# ----------------------------------------------------------------------------------------
def _emg_cdf(t, tau, mu, sigma):
    """CDF of (Gaussian(mu, sigma) + Exponential(tau)) = IRF convolved with the decay.

    F(t) = Phi(z) - exp(-(t - mu)/tau + sigma^2 / (2 tau^2)) * Phi(z - sigma/tau),
    with z = (t - mu)/sigma. The second term is computed in log space so it never overflows.
    """
    z = (t - mu) / sigma
    log_second = -(t - mu) / tau + sigma**2 / (2 * tau**2) + log_ndtr(z - sigma / tau)
    return ndtr(z) - np.exp(log_second)


def decay_curves(tau, cfg):
    """Normalised, IRF-convolved, bin-integrated decay for each lifetime.

    tau: array of any shape (ns). Returns an array of shape (T, *tau.shape) whose
    sum over the time axis is exactly 1.
    """
    tau = np.asarray(tau, dtype=np.float64)
    sigma = cfg.irf_fwhm_ns * FWHM_TO_SIGMA
    edges = cfg.time_edges.reshape(-1, *([1] * tau.ndim))     # (T+1, 1, ..., 1) for broadcasting
    cdf = _emg_cdf(edges, tau[None], cfg.irf_center_ns, sigma)
    prob = np.clip(np.diff(cdf, axis=0), 0.0, None)            # probability per bin (exact integral)
    return prob / prob.sum(axis=0, keepdims=True)              # renormalise inside the window


def irf_curve(cfg):
    """The bin-integrated Gaussian IRF, shape (T,), sums to 1 (what you would measure)."""
    sigma = cfg.irf_fwhm_ns * FWHM_TO_SIGMA
    prob = np.diff(ndtr((cfg.time_edges - cfg.irf_center_ns) / sigma))
    return prob / prob.sum()


def expected_counts(tau, intensity, cfg, bg_per_bin):
    """Noise-free expected counts, shape (T, H, W)."""
    return intensity[None] * decay_curves(tau, cfg) + bg_per_bin


def sample_counts(expected, rng):
    """Poisson-sample the measured histogram. Returns uint16 (enough for <= 65535 counts/bin)."""
    counts = rng.poisson(expected)
    if counts.max() > np.iinfo(np.uint16).max:
        raise ValueError("Counts overflow uint16; lower the photon level.")
    return counts.astype(np.uint16)


def thin(counts, p, rng=None):
    """Photon thinning: keep each detected photon independently with probability p.

    Thinning Poisson(lambda) data gives exactly Poisson(p * lambda) data, so this turns a
    high-photon measurement into a realistic low-photon one (signal AND background scale by p).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    rng = np.random.default_rng() if rng is None else rng
    return rng.binomial(counts, p).astype(counts.dtype)


# ----------------------------------------------------------------------------------------
# Ground-truth maps
# ----------------------------------------------------------------------------------------
def _random_blob(rng, size):
    """One soft-edged ellipse ("cell"), values in [0, 1]."""
    scale = size / 64                                      # radii scale with the image size
    cy, cx = rng.uniform(0, size, 2)
    ry, rx = rng.uniform(4, 12, 2) * scale                 # semi-axes in pixels
    angle = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[:size, :size]
    dy, dx = yy - cy, xx - cx
    u = dx * np.cos(angle) + dy * np.sin(angle)            # rotate into the ellipse frame
    v = -dx * np.sin(angle) + dy * np.cos(angle)
    mask = ((u / rx) ** 2 + (v / ry) ** 2 <= 1.0).astype(np.float64)
    return gaussian_filter(mask, sigma=1.5 * scale)        # blur -> smooth cell boundary


def random_maps(rng, cfg, photons):
    """Random ground truth: tau (H, W), intensity (H, W), foreground mask (H, W) bool.

    * Each cell k gets a brightness w_k and a lifetime tau_k; the space between cells gets
      brightness `floor` and its own lifetime.
    * intensity = (floor + sum_k w_k * blob_k) times a mild smooth texture.
    * tau is the brightness-weighted average of the lifetimes, so it is smooth, stays in
      [tau_min, tau_max], and overlapping cells give gradual lifetime transitions.
    * mask = pixels where the cell signal sum_k w_k * blob_k exceeds fg_threshold.
    * intensity is scaled so that its mean over the FOREGROUND pixels equals `photons`.
    """
    n = cfg.size
    k = rng.integers(cfg.n_blobs_min, cfg.n_blobs_max + 1)
    blobs = np.stack([_random_blob(rng, n) for _ in range(k)])          # (k, H, W)
    weights = rng.uniform(0.5, 1.5, k)
    taus = rng.uniform(cfg.tau_min, cfg.tau_max, k)
    tau_bg = rng.uniform(cfg.tau_min, cfg.tau_max)

    contrib = weights[:, None, None] * blobs                            # brightness per cell
    cell_signal = contrib.sum(axis=0)
    brightness = cfg.floor + cell_signal
    tau = (cfg.floor * tau_bg + (contrib * taus[:, None, None]).sum(axis=0)) / brightness

    mask = cell_signal > cfg.fg_threshold
    if not mask.any():                       # cannot happen with default settings; just in case
        mask = cell_signal >= cell_signal.max()

    texture = 1.0 + 0.8 * gaussian_filter(rng.standard_normal((n, n)), sigma=2.0)
    intensity = brightness * np.clip(texture, 0.5, 1.5)
    intensity *= photons / intensity[mask].mean()
    return tau.astype(np.float32), intensity.astype(np.float32), mask


# ----------------------------------------------------------------------------------------
# Samples and datasets
# ----------------------------------------------------------------------------------------
def simulate_sample(rng, cfg, photons):
    """One image. Returns dict: counts (T,H,W) uint16, tau_gt, intensity_gt (H,W) float32,
    mask (H,W) bool, bg_per_bin (float), photons (float)."""
    tau, intensity, mask = random_maps(rng, cfg, photons)
    bg_per_bin = cfg.bg_fraction * photons / cfg.n_bins
    counts = sample_counts(expected_counts(tau, intensity, cfg, bg_per_bin), rng)
    return {"counts": counts, "tau_gt": tau, "intensity_gt": intensity, "mask": mask,
            "bg_per_bin": bg_per_bin, "photons": float(photons)}


def training_samples(cfg=None, photon_range=(10, 1000), seed=None):
    """Endless generator of random training samples, made on the fly (nothing is saved).

    Each sample's photon level is drawn log-uniformly from photon_range. Yields the same dict
    as simulate_sample. Example:
        gen = training_samples(seed=0)
        batch = [next(gen) for _ in range(8)]
    For a PyTorch IterableDataset, call this in __iter__ with a different seed per worker.
    """
    cfg = SimConfig() if cfg is None else cfg
    rng = np.random.default_rng(seed)
    lo, hi = photon_range
    while True:
        photons = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        yield simulate_sample(rng, cfg, photons)


def generate_dataset(n, photons, cfg=None, seed=0):
    """Generate n samples with a fixed seed.

    photons: a number (same level for all samples) or a (low, high) tuple, in which case each
             sample's level is drawn log-uniformly in that range.
    """
    cfg = SimConfig() if cfg is None else cfg
    rng = np.random.default_rng(seed)
    if np.isscalar(photons):
        levels = np.full(n, float(photons))
    else:
        lo, hi = photons
        levels = np.exp(rng.uniform(np.log(lo), np.log(hi), n))

    samples = [simulate_sample(rng, cfg, p) for p in levels]
    metadata = {**asdict(cfg), "seed": seed, "photons": photons if np.isscalar(photons) else list(photons),
                "model": "mono-exponential * Gaussian IRF + constant background, Poisson noise",
                "photons_definition": "mean signal photons per foreground (mask) pixel",
                "intensity_gt": "expected signal photons per pixel (background excluded)",
                "counts_shape": "(N, T, H, W)"}
    return {
        "counts": np.stack([s["counts"] for s in samples]),               # (N, T, H, W) uint16
        "tau_gt": np.stack([s["tau_gt"] for s in samples]),               # (N, H, W) ns
        "intensity_gt": np.stack([s["intensity_gt"] for s in samples]),   # (N, H, W) photons
        "mask": np.stack([s["mask"] for s in samples]),                   # (N, H, W) bool
        "photons": levels.astype(np.float32),                             # (N,) photon level
        "bg_per_bin": np.array([s["bg_per_bin"] for s in samples], dtype=np.float32),  # (N,)
        "time_ns": cfg.time_centers.astype(np.float32),                   # (T,) bin centres
        "metadata": json.dumps(metadata),
    }


def save_dataset(path, data):
    """Save a dataset dict as compressed .npz (metadata stored as a JSON string)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path


def load_dataset(path):
    """Load a .npz written by save_dataset. metadata comes back as a dict."""
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}
    data["metadata"] = json.loads(str(data["metadata"]))
    return data


def config_from_metadata(meta):
    """Rebuild the SimConfig a dataset was made with."""
    fields = SimConfig.__dataclass_fields__
    return SimConfig(**{k: v for k, v in meta.items() if k in fields})


def test_set_dir():
    from src.paths import RAW_OWN
    return RAW_OWN / "sim_test"


def make_test_set(out_dir=None, levels=TEST_PHOTON_LEVELS, n_per_level=40, seed=12345, force=False):
    """Fixed test set: n_per_level samples at each photon level, one .npz per level.

    Default: 6 levels x 40 = 240 samples in data/raw/own/sim_test/test_p{level}.npz.
    Each level has its own seed (seed + level), so the files never change between runs.
    """
    out_dir = test_set_dir() if out_dir is None else Path(out_dir)
    paths = []
    for level in levels:
        path = out_dir / f"test_p{level}.npz"
        if path.exists() and not force:
            print(f"{path} exists, skipping (use force=True to regenerate)")
        else:
            save_dataset(path, generate_dataset(n_per_level, level, seed=seed + level))
            print(f"Saved {path}  ({path.stat().st_size / 1e6:.1f} MB)")
        paths.append(path)
    return paths


if __name__ == "__main__":
    make_test_set(force="--force" in sys.argv)
