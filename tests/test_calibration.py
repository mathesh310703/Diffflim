"""Tests for src/evaluation/calibration.py (synthetic data, no model needed).

Run:  python tests/test_calibration.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.simulate import SimConfig, simulate_sample  # noqa: E402
from src.evaluation import calibration as cal  # noqa: E402


def _record(rng, n_img, levels, std_scale):
    """Truth = mean + N(0, true_sd); the model reports std = std_scale * true_sd."""
    shape = (n_img, 16, 16)
    rec = {"mask": np.ones(shape, bool), "photons": np.repeat(levels, n_img // len(levels)).astype(float)}
    rec["photons_est"] = rec["photons"].copy()
    for q in cal.QUANTITIES:
        true_sd = 1.0 / np.sqrt(rec["photons"])[:, None, None] * np.ones(shape)
        rec[f"{q}_mean"] = np.ones(shape)
        rec[f"{q}_gt"] = 1.0 + rng.standard_normal(shape) * true_sd
        rec[f"{q}_std"] = true_sd * std_scale[:, None, None]
    return rec


def test_conformal_quantile_finite_sample():
    s = np.arange(1, 10)                         # n = 9 scores
    assert cal.conformal_quantile(s, 0.1) == 9   # ceil(10 * 0.9) = 9th smallest
    assert cal.conformal_quantile(s, 0.5) == 5   # ceil(10 * 0.5) = 5
    assert cal.conformal_quantile(s, 0.05) == float("inf")


def test_global_and_per_level():
    rng = np.random.default_rng(0)
    levels = np.array([10.0, 100.0, 1000.0])
    # std is too small at low photons (x0.5) and too large at high photons (x2), like v1
    scale_of = {10.0: 0.5, 100.0: 1.0, 1000.0: 2.0}
    calib = _record(rng, 60, levels, np.array([scale_of[p] for p in np.repeat(levels, 20)]))
    test = _record(rng, 60, levels, np.array([scale_of[p] for p in np.repeat(levels, 20)]))
    fit = cal.fit(calib, (0.10,))
    for lev, s in scale_of.items():   # per-level factor ~ 1.645 / scale
        assert abs(fit["per_level"][("tau", 0.10, lev)] - 1.645 / s) < 0.1 / s
    for v in ["none", "global", "per_level"]:
        covs = [cal.interval_stats(test, "tau", cal.predict_q(fit, "tau", 0.10, v, test["photons_est"][test["photons"] == p]),
                                   test["photons"] == p)[0] for p in levels]
        print(f"  {v:9s} coverage per level: {np.round(covs, 3)}")
        if v == "per_level":
            assert all(abs(c - 0.90) < 0.02 for c in covs)
        if v == "global":
            pooled = cal.interval_stats(test, "tau", cal.predict_q(fit, "tau", 0.10, v, test["photons_est"]))[0]
            assert abs(pooled - 0.90) < 0.02


def test_photon_level_estimate():
    cfg = SimConfig()
    for p in [10, 100, 1000]:
        ratios = []
        for seed in range(5):
            s = simulate_sample(np.random.default_rng(seed), cfg, p)
            ratios.append(cal.estimate_photon_level(s["counts"].sum(0)) / p)
        print(f"  level {p}: estimate / true = {np.round(ratios, 2)}")
        assert all(0.7 < r < 1.4 for r in ratios)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        print(fn.__name__)
        fn()
    print(f"All {len(tests)} tests passed.")
