"""Sanity tests for the MLE and phasor baselines.

Run:  python tests/test_baselines.py        (or `pytest tests/`)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines.mle import fit_mle, fit_pixel_scalar  # noqa: E402
from src.baselines.phasor import fit_phasor  # noqa: E402
from src.data.simulate import SimConfig, expected_counts, sample_counts, simulate_sample  # noqa: E402


def _uniform_image(tau0, photons, cfg, bg_per_bin, rng):
    tau = np.full((cfg.size, cfg.size), tau0)
    return sample_counts(expected_counts(tau, np.full_like(tau, photons), cfg, bg_per_bin), rng)


def test_high_photon_recovery():
    """With 1e5 photons/pixel both methods recover tau to <1% and intensity to <1%."""
    cfg = SimConfig(size=4)
    rng = np.random.default_rng(0)
    bg = 0.02 * 1e5 / cfg.n_bins
    for tau0 in [0.5, 1.0, 2.5, 5.0]:
        counts = _uniform_image(tau0, 1e5, cfg, bg, rng)
        for name, fit in [("MLE", fit_mle), ("Phasor", fit_phasor)]:
            tau, inten = fit(counts, cfg, bg)[:2]
            print(f"  {name:6s} tau0={tau0}: tau={np.median(tau):.4f}, I={np.median(inten):.0f}")
            assert abs(np.median(tau) / tau0 - 1) < 0.01
            assert abs(np.median(inten) / 1e5 - 1) < 0.01


def test_grid_matches_minimize_scalar():
    """The vectorised grid fit equals the one-pixel minimize_scalar fit."""
    cfg = SimConfig(size=8)
    s = simulate_sample(np.random.default_rng(1), cfg, photons=100)
    tau, inten = fit_mle(s["counts"], cfg, s["bg_per_bin"])
    for y, x in [(0, 0), (3, 4), (7, 2), (5, 5)]:
        t_ref, i_ref = fit_pixel_scalar(s["counts"][:, y, x], cfg, s["bg_per_bin"])
        print(f"  pixel ({y},{x}): grid tau {tau[y, x]:.4f} vs scalar {t_ref:.4f}")
        assert abs(tau[y, x] / t_ref - 1) < 2e-3
        assert abs(inten[y, x] / i_ref - 1) < 1e-3


def test_mle_speed():
    cfg = SimConfig()
    s = simulate_sample(np.random.default_rng(2), cfg, photons=100)
    t0 = time.perf_counter()
    fit_mle(s["counts"], cfg, s["bg_per_bin"])
    sec = time.perf_counter() - t0
    print(f"  64x64 MLE fit: {sec:.2f} s")
    assert sec < 30


def test_v3_forward_model_recovery():
    """Mono-exponential v3 data (real PMT IRF): MLE, binned MLE and phasor recover tau."""
    from src.baselines.common import fit_all
    from src.data import simulate_v3 as v3
    cfg3 = v3.SimConfigV3(size=8)
    irf = v3.irf_library(cfg3)[6]["irf"]                        # IRF_PMT
    meta = {"simulator": "v3", "size": 8, "n_bins": 256, "bin_width_ns": cfg3.bin_width_ns}
    rng = np.random.default_rng(5)
    for tau0 in [0.8, 2.5, 6.0]:
        lam = 1e5 * v3.decay_curves_v3(np.full((8, 8), tau0), irf, cfg3) + 2.0
        res = fit_all(sample_counts(lam, rng), 2.0, meta, irf)
        for name, (tau, inten) in res.items():
            print(f"  {name:7s} tau0={tau0}: {np.median(tau):.4f} ns, I={np.median(inten):.0f}")
            # phasor: v3 decays are truncated at the window end (not periodic) -> bias grows with tau
            tol = 0.01 if name != "Phasor" else (0.02 if tau0 < 3 else 0.06)
            assert abs(np.median(tau) / tau0 - 1) < tol, name
            assert abs(np.median(inten) / 1e5 - 1) < 0.01, name


def test_zero_count_pixels():
    cfg = SimConfig(size=4)
    tau, inten = fit_mle(np.zeros((cfg.n_bins, 4, 4), dtype=np.uint16), cfg, 0.0)
    assert np.all(np.isnan(tau)) and np.all(inten == 0)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        print(fn.__name__)
        fn()
    print(f"All {len(tests)} tests passed.")
