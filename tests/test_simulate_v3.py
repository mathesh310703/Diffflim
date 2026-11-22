"""Sanity tests for simulator v3 (needs the HPA images in data/raw/hpa/train/).

Run:  python tests/test_simulate_v3.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import simulate_v3 as v3  # noqa: E402
from src.data.simulate import sample_counts  # noqa: E402

CFG = v3.SimConfigV3()


def test_irfs_normalised():
    lib = v3.irf_library(CFG)
    assert len(lib) == 12 and not v3.EXCLUDED_IRFS
    rng = np.random.default_rng(0)
    for x in lib:
        assert x["irf"].shape == (CFG.n_bins,) and np.all(x["irf"] >= 0)
        assert np.isclose(x["irf"].sum(), 1.0)
        assert 5 < x["irf"].argmax() < 80, x["id"]            # peak well inside the window
        for _ in range(20):
            j, shift = v3.jitter_irf(x["irf"], x["jitter"], rng)
            assert np.isclose(j.sum(), 1.0) and -x["jitter"][0] <= shift < x["jitter"][1]


def test_recursion_equals_convolution():
    rng = np.random.default_rng(1)
    irf = v3.irf_library(CFG)[7]["irf"]
    a = rng.random((4, 30)); a /= a.sum(0)
    t = rng.uniform(0.1, 12, (4, 30))
    fast = v3.multi_exp_curves(a, t, irf, CFG)
    ref = np.stack([np.convolve((a[:, i, None] * np.exp(-CFG.time_starts[None] / t[:, i, None])).sum(0), irf)[:256]
                    for i in range(30)], 1)
    assert np.allclose(fast, ref / ref.sum(0), atol=1e-12)


def test_amplitudes_flimngo():
    present = np.array([[1, 1], [1, 0], [0, 1], [1, 1]], bool)[:, :, None]     # 2 pixels
    p = np.array([[0.3], [0.3]])
    a = v3.channel_amplitudes(present, p)[:, :, 0]
    # pixel 0: channels R, G, Y -> p, p(1-p), (1-p)^2 ; pixel 1: R, B, Y
    assert np.allclose(a[:, 0], [0.3, 0.21, 0, 0.49]) and np.allclose(a[:, 1], [0.3, 0, 0.21, 0.49])
    assert np.allclose(a.sum(0), 1)


def test_counts_match_intensity():
    rng = np.random.default_rng(2)
    s = v3.simulate_sample(rng, CFG, 50, v3.train_ids(write=False), return_extra=True)
    lam, m = s["expected"], s["mask"]
    truth = s["intensity_gt"] + s["bg_per_bin"] * CFG.n_bins
    assert np.allclose(lam.sum(0), truth, rtol=1e-5)
    assert np.isclose(s["intensity_gt"][m].mean(), 50, rtol=1e-4) and np.all(s["intensity_gt"][~m] == 0)
    k = 200
    mean_total = np.mean([sample_counts(lam, rng).sum(0) for _ in range(k)], 0)
    z = (mean_total - truth) / np.sqrt(truth / k)
    print(f"  max |z| = {np.abs(z).max():.2f}, mean z = {z.mean():+.3f}")
    assert np.abs(z).max() < 5.5 and abs(z.mean()) < 0.2


def test_mean_arrival_time_single_component():
    """Mean arrival - IRF mean = mean of the point-sampled exponential, dt / (exp(dt/tau) - 1),
    which is within one bin of tau (and checks the whole forward model incl. the IRF)."""
    rng = np.random.default_rng(3)
    dt = CFG.bin_width_ns
    irf = v3.irf_library(CFG)[5]["irf"]                       # narrow simulated IRF
    t = CFG.time_centers
    irf_mean = (irf * t).sum()
    for tau in [0.5, 1.0, 2.0, 3.0]:
        d = v3.decay_curves_v3(np.full((4, 4), tau), irf, CFG)
        counts = sample_counts(1e5 * d, rng)
        est = ((counts * t[:, None, None]).sum(0) / counts.sum(0)).mean() - irf_mean
        expect = dt / np.expm1(dt / tau)
        print(f"  tau={tau}: mean arrival - IRF mean = {est:.4f} ns (discrete expectation {expect:.4f})")
        assert abs(est - expect) / tau < 0.01 and abs(est - tau) < dt


def test_ground_truth_definition():
    s = v3.simulate_sample(np.random.default_rng(4), CFG, 100, v3.train_ids(write=False), return_extra=True)
    m = s["mask"]
    assert np.allclose(s["tau_gt"][m], (s["amps"] * s["taus"]).sum(0)[m], rtol=1e-5)
    assert np.allclose(s["amps"].sum(0)[m], 1)
    assert np.all(s["tau_int_gt"][m] >= s["tau_gt"][m] - 1e-4)   # intensity-weighted >= amplitude-weighted
    assert np.array_equal(m, s["channels"].astype(int).sum(-1) > 0) and m.mean() >= CFG.min_fg_frac


def test_reproducible():
    ids = v3.test_ids()
    a = v3.generate_dataset(2, 50, ids, seed=7)
    b = v3.generate_dataset(2, 50, ids, seed=7)
    for k in ["counts", "tau_gt", "intensity_gt", "mask", "irf", "crop"]:
        assert np.array_equal(a[k], b[k])
    assert list(a["hpa_id"]) == list(b["hpa_id"]) and set(a["hpa_id"]) <= set(ids)


def test_split_disjoint_and_fixed():
    before = (v3.TEST_IDS_FILE.read_text(), v3.CALIB_IDS_FILE.read_text() if v3.CALIB_IDS_FILE.exists() else None)
    te, ca, tr = v3.test_ids(), v3.calib_ids(), v3.train_ids(write=False)
    assert len(te) == v3.N_TEST_IDS and len(ca) == v3.N_CALIB_IDS
    assert not set(te) & set(tr) and not set(ca) & set(tr) and not set(te) & set(ca)
    assert set(tr) == set(v3.complete_ids()) - set(te) - set(ca)   # every other complete sample trains
    assert v3.TEST_IDS_FILE.read_text() == before[0]               # never rewritten
    if before[1] is not None:
        assert v3.CALIB_IDS_FILE.read_text() == before[1]
    gen = v3.training_samples(seed=0)
    assert all(next(gen)["hpa_id"] in set(tr) for _ in range(20))


def test_paired_levels():
    """Paired data sets: same scenes at every photon level, only photons and noise differ."""
    ids = v3.test_ids()
    a = v3.generate_paired_dataset(3, 10, ids, scene_seed=5, noise_seed=15)
    b = v3.generate_paired_dataset(3, 1000, ids, scene_seed=5, noise_seed=1005)
    for k in ["tau_gt", "mask", "irf", "crop", "n_components"]:
        assert np.array_equal(a[k], b[k]), k
    assert list(a["hpa_id"]) == list(b["hpa_id"])
    assert np.allclose(b["intensity_gt"], 100 * a["intensity_gt"], rtol=1e-5)
    assert not np.array_equal(a["counts"], b["counts"])


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        print(fn.__name__)
        fn()
    print(f"All {len(tests)} tests passed.")
