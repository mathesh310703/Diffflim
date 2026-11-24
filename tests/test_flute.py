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


def test_flute_nadh_option():
    """flute_nadh_config: NAD(P)H scenes have 2 components with tau_free in 0.3-0.5 ns and tau_bound in
    1.5-4 ns; mono scenes have 1 component with 0.2-5 ns; both kinds occur; the plain FLUTE and FLIMngo
    configurations are not affected (lifetime_model default)."""
    assert v3.flute_config().lifetime_model == "flimngo" and v3.SimConfigV3().lifetime_model == "flimngo"
    cfg = v3.flute_nadh_config()
    ids = v3.train_ids()
    if not ids:
        print("no HPA data, NADH test skipped")
        return
    rng = np.random.default_rng(1)
    kinds = set()
    for _ in range(12):
        sc = v3.make_scene(rng, cfg, ids[:5])
        m = sc["mask"]
        nc = int(sc["n_components"][m].max())
        kinds.add(nc)
        if nc == 2:
            tf, tb = sc["taus"][0][m], sc["taus"][1][m]
            assert np.allclose(tf, tf[0]) and 0.3 <= tf[0] <= 0.5
            assert tb.min() >= 1.5 - 1e-9 and tb.max() <= 4.0 + 1e-9
            af = sc["amps"][0][m]
            assert af.min() >= 0.3 - 1e-9 and af.max() <= 0.9 + 1e-9
            assert np.allclose(sc["amps"].sum(0)[m], 1.0)
        else:
            t = sc["taus"][0][m]
            assert sc["amps"].shape[0] == 1 and t.min() >= 0.2 - 1e-9 and t.max() <= 5.0 + 1e-9
        assert np.allclose(sc["tau_gt"][m], (sc["amps"] * sc["taus"]).sum(0)[m])
        assert np.allclose(sc["curves"].sum(0), 1.0)
    assert kinds == {1, 2}, kinds


def test_guidance_fixed_point_is_mle():
    """Repeated kappa = 0 guidance steps (no network) converge to the per-pixel Poisson MLE with the
    same periodic forward model (24 November 2026)."""
    import torch
    from src.baselines.mle import fit_mle
    from src.models.diffflim import denormalize_targets, normalize_targets
    from src.models.guidance import PoissonGuidance
    cfg = FluteConfig()
    model = PeriodicDecayModel(gaussian_irf_fine(cfg, 1.0, 0.15), cfg, tau_bounds=(0.02, 60.0))
    rng = np.random.default_rng(0)
    tau = rng.uniform(0.5, 5.0, (8, 8))
    lam = 200 * model(tau) + 0.05
    counts = rng.poisson(lam)
    c = torch.tensor(counts, dtype=torch.float32)
    step = PoissonGuidance(model, 0.05, 0.0, max_step=0.3).bind(c, (0.1, 12.0))
    x = normalize_targets(torch.full((1, 8, 8), 2.0), c.sum(0)[None], (0.1, 12.0))
    for _ in range(60):
        x = step(x, 0, 1)
    t_g = denormalize_targets(x, (0.1, 12.0))[0][0].numpy()
    t_m, _ = fit_mle(counts, cfg, 0.05, tau_bounds=(0.05, 25.0), decay_fn=model)
    assert np.max(np.abs(t_g - t_m) / t_m) < 0.01
    # a very large prior precision leaves x0 unchanged; outside the window nothing changes either
    x0 = normalize_targets(torch.full((1, 8, 8), 2.0), c.sum(0)[None], (0.1, 12.0))
    assert torch.allclose(PoissonGuidance(model, 0.05, 1e12).bind(c, (0.1, 12.0))(x0, 0, 1), x0, atol=1e-5)
    assert torch.equal(PoissonGuidance(model, 0.05, 0.0, stop_frac=0.5).bind(c, (0.1, 12.0))(x0, 5, 10), x0)


def test_flute_v2_prior_scenes():
    """FLUTE v2 prior: lifetimes inside 0.2 to 6 ns, all four scene types occur, uniform scenes are
    exactly uniform, NAD(P)H-like scenes have two components."""
    cfg = v3.flute_v2prior_config()
    rng = np.random.default_rng(3)
    ids = v3.train_ids()
    kinds = set()
    for _ in range(40):
        s = v3.make_scene(rng, cfg, ids)
        m = s["mask"]
        t = s["tau_gt"][m]
        if s["n_components"][m].max() == 2:
            kinds.add("nadh")
            continue
        assert t.min() >= 0.2 - 1e-9 and t.max() <= 6.0 + 1e-9
        kinds.add("uniform" if np.ptp(t) == 0 else "near" if np.ptp(t) / np.median(t) <= 0.2 else "structured")
        assert np.allclose(s["tau_int"][m], t)
    assert kinds == {"uniform", "near", "structured", "nadh"}, kinds


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} passed")
    print("all FLUTE tests passed")
