"""Tests for the FLUTE loader, the periodic FLUTE model and the FLUTE configuration of simulator v3.

Run: python tests/test_flute.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import simulate_v3 as v3  # noqa: E402
from src.data.flute import (FLUTE_DIR, FLUTE_FILES, FluteConfig, PeriodicDecayModel, binomial_thin,  # noqa: E402
                            gaussian_irf_fine, load_flute, periodic_decays, rise_time)


def test_loader():
    f = FLUTE_DIR / FLUTE_FILES["fluorescein_hmsc"]
    if not f.exists():
        print("FLUTE data missing, loader test skipped")
        return
    a = load_flute("fluorescein_hmsc")
    assert a.shape == (56, 512, 512) and a.dtype == np.int64 and a.min() >= 0


def test_periodic_decay_wrap_level():
    """With a narrow IRF the level just before the pulse equals the analytic wrapped tail."""
    cfg = FluteConfig()
    irf = gaussian_irf_fine(cfg, 3.0, 0.02)
    d = periodic_decays([4.0], irf, cfg)[0]
    assert abs(d.sum() - 1) < 1e-9
    k_peak = int(np.argmax(d))
    # level of the bin before the pulse relative to the peak bin ~ exp(-(P - dt) / tau) / (1 - exp(-P / tau)) * ...
    ratio = d[k_peak - 2] / d[k_peak]
    expected = np.exp(-(cfg.period_ns - 2 * cfg.bin_width_ns) / 4.0)
    assert abs(ratio / expected - 1) < 0.15, (ratio, expected)


def test_interpolated_model():
    cfg = FluteConfig()
    irf = gaussian_irf_fine(cfg, 1.0, 0.15)
    m = PeriodicDecayModel(irf, cfg)
    exact = periodic_decays([0.7, 4.0], irf, cfg)
    assert np.abs(m(np.array([0.7, 4.0])).T - exact).max() < 1e-5


def test_rise_time_and_thinning():
    cfg = FluteConfig()
    d1 = periodic_decays([4.0], gaussian_irf_fine(cfg, 1.0, 0.15), cfg)[0]
    d2 = periodic_decays([4.0], gaussian_irf_fine(cfg, 2.0, 0.15), cfg)[0]
    assert abs(rise_time(d2, cfg) - rise_time(d1, cfg) - 1.0) < 0.05
    c = np.random.default_rng(0).poisson(50.0, size=(56, 32, 32))
    t = binomial_thin(c, 0.2, seed=1)
    assert np.all(t <= c) and abs(t.mean() / c.mean() - 0.2) < 0.01


def test_v3_periodic_matches_circular_convolution():
    """simulate_v3.multi_exp_curves with periodic=True equals the circular convolution of the IRF with
    the periodic point-sampled exponential."""
    cfg = v3.flute_config()
    T = cfg.n_bins
    irf = np.zeros(T)
    irf[[50, 51, 52, 0]] = [0.2, 0.5, 0.2, 0.1]                    # wraps around the window
    tau = 3.0
    got = v3.multi_exp_curves(np.ones((1, 1)), np.full((1, 1), tau), irf, cfg)[:, 0]
    r = np.exp(-cfg.bin_width_ns / tau)
    e = r ** np.arange(T) / (1 - r ** T)                           # periodic exponential
    ref = np.real(np.fft.ifft(np.fft.fft(irf) * np.fft.fft(e)))
    ref /= ref.sum()
    assert np.abs(got - ref).max() < 1e-10


def test_flimngo_default_unchanged():
    cfg = v3.SimConfigV3()
    assert cfg.periodic is False and cfg.irf_set == "flimngo"
    lib = v3.irf_library(cfg)
    assert len(lib) == 12 and all(len(x["irf"]) == 256 for x in lib)


def test_flute_config_sample():
    cfg = v3.flute_config()
    lib = v3.irf_library(cfg)
    assert len(lib) == 13 and all(len(x["irf"]) == 56 and abs(x["irf"].sum() - 1) < 1e-9 for x in lib)
    ids = v3.train_ids()
    if not ids:
        print("no HPA data, sample test skipped")
        return
    s = v3.simulate_sample(np.random.default_rng(0), cfg, 100.0, ids[:5])
    assert s["counts"].shape == (56, 64, 64)
    lo, hi = (round(x / cfg.bin_width_ns) for x in cfg.irf_peak_range_ns)
    assert lo <= int(np.argmax(s["irf"])) <= hi
    fg = s["counts"][:, s["mask"]].sum(0)
    assert 60 < fg.mean() < 140


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} passed")
    print("all FLUTE tests passed")
