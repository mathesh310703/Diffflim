"""Sanity tests for src/data/simulate.py.

Run:  python tests/test_simulate.py        (or `pytest tests/` if pytest is installed)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.simulate import (  # noqa: E402
    SimConfig, decay_curves, expected_counts, generate_dataset, random_maps, sample_counts, thin,
    training_samples,
)


def test_decay_is_normalised():
    cfg = SimConfig()
    d = decay_curves(np.array([0.5, 1.0, 2.5, 5.0]), cfg)
    assert d.shape == (cfg.n_bins, 4)
    assert np.all(d >= 0)
    assert np.allclose(d.sum(axis=0), 1.0)


def test_maps_in_range():
    cfg = SimConfig()
    tau, inten, mask = random_maps(np.random.default_rng(0), cfg, photons=100)
    assert tau.shape == inten.shape == mask.shape == (cfg.size, cfg.size)
    assert mask.dtype == bool and 0.05 < mask.mean() < 0.95
    assert tau.min() >= cfg.tau_min - 1e-4 and tau.max() <= cfg.tau_max + 1e-4
    assert inten.min() > 0
    assert np.isclose(inten[mask].mean(), 100, rtol=1e-4)          # photons = per foreground pixel
    assert inten[~mask].mean() < 0.3 * inten[mask].mean()          # background clearly dimmer


def test_training_samples():
    gen = training_samples(SimConfig(size=16), photon_range=(10, 1000), seed=0)
    levels = [next(gen)["photons"] for _ in range(50)]
    assert all(10 <= p <= 1000 for p in levels) and len(set(levels)) == 50
    a, b = next(training_samples(seed=5)), next(training_samples(seed=5))
    assert np.array_equal(a["counts"], b["counts"])


def test_mean_counts_match_intensity():
    """Average photon count per pixel over many Poisson draws = intensity + background."""
    cfg = SimConfig(size=32)
    rng = np.random.default_rng(1)
    photons = 50
    tau, inten, _ = random_maps(rng, cfg, photons)
    bg = cfg.bg_fraction * photons / cfg.n_bins
    lam = expected_counts(tau, inten, cfg, bg)
    truth = inten + bg * cfg.n_bins
    assert np.allclose(lam.sum(axis=0), truth, rtol=1e-5)          # noise-free model is exact

    k = 200
    mean_total = np.mean([sample_counts(lam, rng).sum(axis=0) for _ in range(k)], axis=0)
    z = (mean_total - truth) / np.sqrt(truth / k)                    # standard error of the mean
    print(f"  mean counts: max |z| = {np.abs(z).max():.2f}, mean z = {z.mean():+.3f}")
    assert np.abs(z).max() < 5.5                                      # no pixel is off
    assert abs(z.mean()) < 0.2                                        # no global bias


def test_mean_arrival_time_matches_tau():
    """At very high photon counts, mean arrival time - IRF position ~= tau."""
    cfg = SimConfig(size=4, bg_fraction=0.0)
    rng = np.random.default_rng(2)
    t = cfg.time_centers
    window = cfg.n_bins * cfg.bin_width_ns - cfg.irf_center_ns
    for tau0 in [0.5, 1.0, 2.0, 3.0, 5.0]:
        tau = np.full((cfg.size, cfg.size), tau0)
        counts = sample_counts(expected_counts(tau, np.full_like(tau, 1e5), cfg, 0.0), rng)
        total = counts.sum(axis=0)
        mean_t = (counts * t[:, None, None]).sum(axis=0) / total - cfg.irf_center_ns
        est = mean_t.mean()
        # mean of an exponential cut off at the end of the window (matters only for long tau)
        truncated = tau0 - window / np.expm1(window / tau0)
        print(f"  tau={tau0:.1f}: mean arrival {est:.3f} ns (window-corrected expectation {truncated:.3f})")
        assert abs(est - truncated) / tau0 < 0.01
        if tau0 <= 3.0:
            assert abs(est - tau0) / tau0 < 0.03


def test_thinning():
    """Thinning Poisson(lam) by p gives Poisson(p*lam): mean = variance = p*lam."""
    rng = np.random.default_rng(3)
    counts = rng.poisson(20.0, size=200_000).astype(np.uint16)
    thinned = thin(counts, 0.3, rng)
    assert thinned.dtype == counts.dtype
    assert np.all(thinned <= counts)
    assert abs(thinned.mean() - 6.0) < 0.05
    assert abs(thinned.var() - 6.0) < 0.15
    assert np.array_equal(thin(counts, 1.0, rng), counts)
    assert thin(counts, 0.0, rng).sum() == 0


def test_reproducible():
    cfg = SimConfig(size=16)
    a = generate_dataset(2, 50, cfg, seed=7)
    b = generate_dataset(2, 50, cfg, seed=7)
    assert a["counts"].shape == (2, cfg.n_bins, 16, 16) and a["counts"].dtype == np.uint16
    assert a["mask"].dtype == bool
    for key in ["counts", "tau_gt", "intensity_gt", "mask"]:
        assert np.array_equal(a[key], b[key])


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        print(fn.__name__)
        fn()
    print(f"All {len(tests)} tests passed.")
