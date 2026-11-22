"""Real-data evaluation on FLIMngo's raw C. elegans recordings (no ground truth, no training).

Data: external/FLIMngo/sample_data/celegans/raw_data/: 6 abeta_1s_w*.sdt (neurons expressing
mScarlet-tagged amyloid-beta) and 6 mscarlet_1s_w*.sdt (mScarlet only), 1 s acquisitions of
unparalysed worms, 256 x 256 pixels, 256 time bins of 0.09765625 ns (Kapsiani et al. 2025, Fig. 8;
their notebook demo_notebooks/predict_celegans_dynamic.ipynb). Read with sdtfile, time axis moved to
the front as in FLIMngo's utils/eval_utils.py::load_sdt.

The PROTOCOL below is fixed before looking at any result and must not be tuned on these data; later
models are evaluated with the same script (--ckpt / --tag):

  mask       foreground_mask(summed counts, "flimngo_recall") = summed counts > 0.2 x their 99th
             percentile (src/evaluation/calibration.py), computed on the FULL recording and used for
             every method and for both halves. FLIMngo's own manual masks are only used for the
             reproduction of their notebook (supporting rows).
  background constant per bin, estimated per recording as the mean count per bin and pixel OUTSIDE
             the mask in the pre-rise bins 7-18 (bins 0-6 are always empty in these recordings).
  IRF (MLE)  FLIMngo's in-house PMT IRF (IRF_inhouse_PMT.sdt, same lab), aligned once: integer
             shifts of -5..+5 bins are tried on the decay pooled over the masks of all 12 recordings
             (mono-exponential Poisson MLE); the shift with the lowest negative log-likelihood is used
             for every recording (no group information is used).
  MLE        per-pixel mono-exponential Poisson MLE with the v3 point-sampled forward model and the
             aligned IRF (src/baselines/mle.py), tau bounds 0.05-15 ns; binned MLE 5 x 5 likewise on
             the 5 x 5 summed histograms (background x 25).
  FLIMngo    pretrained network with its own preprocessing (mask, per-pixel max normalisation), on
             the full 256 x 256 image, bin width from the .sdt metadata.
  DiffFLIM   --steps DDIM steps x --samples samples (default 10 x 10) on overlapping 64 x 64 tiles
             (stride 32, 7 x 7 = 49 tiles; seed 1000 + tile index + SEED_OFFSET[part], so that the full
             recording and the two halves never share the starting noise of the sampler). Tiles are blended with a
             separable linear ramp weight (1/32 at the tile edge, 1 from 32 px inwards): mean =
             weighted mean of the tile means, std = sqrt(weighted mean of the tile variances). The 90 %
             interval is mean +/- q * std, with q = the per-level lifetime factor of the main
             calibration (--factors, quadratic curve in log photons) at the photon level ESTIMATED
             from the recording (estimate_photon_level, "otsu_adaptive"); estimates outside the
             calibrated range are clamped to it and flagged.
  Test A     per recording and method: median lifetime inside the mask (full data); Abeta versus
             mScarlet with a two-sided exact Mann-Whitney U test per method (6 versus 6 recordings:
             the smallest possible p value is 2/924 = 0.0022).
  Test B     split-half consistency: each recording's photons are split by binomial thinning with
             p = 0.5 (seed 20261122 + recording index) into half A and half B = counts - A, two
             independent Poisson recordings with half the photons. Every method runs on A, B and the
             full data. Reported inside the mask: median |tau_A - tau_B| (ns and relative to their
             mean), median |tau_half - tau_full|; for DiffFLIM the fraction of pixels whose calibrated
             90 % intervals of A and B overlap, whose B mean lies in A's interval (and vice versa), and
             whose full-data mean lies in a half's interval. These fractions are NOT coverages: the
             two halves share the same truth, so errors that do not depend on the photon noise (for
             example the pull of the learned prior) are common to both halves and do not show up in
             A - B. Their expected values for a calibrated model are therefore taken from the same
             split-half analysis on simulated data with known truth (below).
  Sim. ref.  the same split-half analysis on the v3 test set (first 10 images at 20, 50 and 100
             photons, i.e. halves of 10, 25 and 50 photons, which bracket the real halves; 64 x 64,
             no tiling), with the calibrated coverage of each half against the ground truth.
  Examples   abeta_1s_w1 and mscarlet_1s_w1 (first recording of each group), lifetime colour range
             1-5 ns, 90 % half-width 0-1.5 ns, cropped to the bounding box of the mask plus 8 pixels.

Writes (tag = --tag, default v2):
  results/tables/real_celegans_groups_{tag}.csv      Test A (per recording, plus test rows)
  results/tables/real_celegans_splithalf_{tag}.csv   Test B (per recording, plus median rows)
  results/tables/real_celegans_splithalf_simref_{tag}.csv   split-half reference on simulated data
  results/figures/real_celegans_groups_{tag}.png
  results/figures/real_celegans_examples_{tag}.png
  data/processed/real_celegans_baselines.npz, real_celegans_diffflim_{tag}.npz (cached predictions)

Usage:
  python scripts/eval_real_celegans.py
  python scripts/eval_real_celegans.py --ckpt checkpoints/<model>.pt --tag <tag> --factors <factors.csv>
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sdtfile  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines import flimngo_baseline as pre  # noqa: E402
from src.baselines.binned import bin_spatial  # noqa: E402
from src.baselines.mle import fit_mle  # noqa: E402
from src.data import simulate_v3 as v3  # noqa: E402
from src.evaluation.calibration import estimate_photon_level, foreground_mask  # noqa: E402
from src.paths import FIGURES, PROCESSED, TABLES  # noqa: E402

DATA_DIR = ROOT / "external" / "FLIMngo" / "sample_data" / "celegans" / "raw_data"
FLIMNGO_MASK_DIR = ROOT / "external" / "FLIMngo" / "sample_data" / "celegans" / "masks"
GROUPS = ("abeta", "mscarlet")
GROUP_LABEL = {"abeta": "Abeta-mScarlet", "mscarlet": "mScarlet"}
MASK_METHOD = "flimngo_recall"
IRF_ID = "IRF_PMT"
SHIFTS = range(-5, 6)
BG_BINS = slice(7, 19)
TAU_BOUNDS = (0.05, 15.0)
BIN_K = 5
SPLIT_P, SPLIT_SEED = 0.5, 20261122
TILE, STRIDE = 64, 32
ALPHA = 0.10
PHOTON_ESTIMATOR = "otsu_adaptive"
TAU_DISPLAY = (1.0, 5.0)
HALFWIDTH_DISPLAY = (0.0, 1.5)   # ns, colour range of the uncertainty panels (display only)
EXAMPLES = ("abeta_1s_w1", "mscarlet_1s_w1")
PARTS = ("full", "A", "B")
SEED_OFFSET = {"full": 0, "A": 100000, "B": 200000}   # independent sampler noise per part
METHODS = ("MLE", "Binned MLE 5x5", "FLIMngo pretrained", "DiffFLIM")


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
def load_recordings():
    recs = []
    for i, f in enumerate(sorted(DATA_DIR.glob("*.sdt"))):
        sdt = sdtfile.SdtFile(f)
        counts = np.moveaxis(sdt.data[0], -1, 0).astype(np.int64)            # (T, Y, X)
        bin_width = float(np.diff(sdt.times[0][:2])[0] * 1e9)                  # ns
        s = counts.sum(0)
        mask = foreground_mask(s, MASK_METHOD)
        bg = float(counts[BG_BINS][:, ~mask].mean())
        rng = np.random.default_rng(SPLIT_SEED + i)
        half_a = rng.binomial(counts, SPLIT_P)
        their_mask = np.asarray(Image.open(FLIMNGO_MASK_DIR / f"{f.stem}.tif")) > 0
        recs.append({"name": f.stem, "group": f.stem.split("_")[0], "bin_width": bin_width, "mask": mask,
                     "their_mask": their_mask, "bg_full": bg,
                     "parts": {"full": counts, "A": half_a, "B": counts - half_a}})
    return recs


def shift_irf(irf, s):
    out = np.concatenate([np.zeros(s), irf[:len(irf) - s]]) if s >= 0 else np.concatenate([irf[-s:], np.zeros(-s)])
    return out / out.sum()


def align_irf(recs, cfg):
    """Integer IRF shift with the lowest mono-exponential NLL on the decay pooled over all masks."""
    irf0 = {x["id"]: x["irf"] for x in v3.irf_library(cfg)}[IRF_ID]
    pooled = sum(r["parts"]["full"][:, r["mask"]].sum(1) for r in recs).astype(np.float64)
    b = sum(r["bg_full"] * r["mask"].sum() for r in recs)
    nll = {}
    for s in SHIFTS:
        irf = shift_irf(irf0, s)
        fn = lambda tau, irf=irf: v3.decay_curves_v3(tau, irf, cfg)  # noqa: E731
        tau, inten = fit_mle(pooled.reshape(-1, 1, 1), cfg, b, tau_bounds=TAU_BOUNDS, decay_fn=fn)
        lam = inten[0, 0] * fn(np.array([tau[0, 0]]))[:, 0] + b
        nll[s] = float(np.sum(lam - pooled * np.log(lam)))
        print(f"  IRF shift {s:+d} bins: tau {tau[0, 0]:.3f} ns, NLL {nll[s]:.1f}")
    best = min(nll, key=nll.get)
    return best, shift_irf(irf0, best), nll


# ----------------------------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------------------------
def run_mle(counts, mask, bg, irf, cfg, k=1):
    """Per-pixel (k = 1) or k x k binned MLE, only on mask pixels. Returns tau (H, W), NaN outside."""
    fn = lambda tau: v3.decay_curves_v3(tau, irf, cfg)  # noqa: E731
    c = bin_spatial(counts, k) if k > 1 else counts
    sel = c[:, mask].reshape(c.shape[0], -1, 1)
    tau, _ = fit_mle(sel, cfg, bg * k * k, tau_bounds=TAU_BOUNDS, decay_fn=fn)
    out = np.full(mask.shape, np.nan)
    out[mask] = tau[:, 0]
    return out


def tile_weight():
    i = np.arange(TILE)
    ramp = np.minimum(np.minimum(i + 1, TILE - i) / (TILE / 2), 1.0)
    return np.outer(ramp, ramp)


def run_diffflim_tiled(model, counts, steps, n_samples, seed0=1000):
    """Lifetime mean and std (H, W) from overlapping 64 x 64 tiles."""
    _, H, W = counts.shape
    ys = sorted(set(list(range(0, H - TILE + 1, STRIDE)) + [H - TILE]))
    xs = sorted(set(list(range(0, W - TILE + 1, STRIDE)) + [W - TILE]))
    w = tile_weight()
    acc_m, acc_v, acc_w = np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
    k = 0
    for y in ys:
        for x in xs:
            s = model.sample(counts[:, y:y + TILE, x:x + TILE], n_samples=n_samples, steps=steps, seed=seed0 + k)
            acc_m[y:y + TILE, x:x + TILE] += w * s[:, 0].mean(0)
            acc_v[y:y + TILE, x:x + TILE] += w * s[:, 0].var(0, ddof=1)
            acc_w[y:y + TILE, x:x + TILE] += w
            k += 1
    return acc_m / acc_w, np.sqrt(acc_v / acc_w)


def load_factor_curve(path, alpha=ALPHA):
    """Per-level lifetime factors of a calibration_factors_*.csv -> (coef, lo, hi) of the quadratic
    curve in log(estimated photons), exactly as calibration.predict_q(..., smooth=True)."""
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["quantity"] == "tau" and r["variant"] == "per_level"
                and abs(float(r["nominal_coverage"]) - (1 - alpha)) < 1e-6]
    x = np.log([float(r["photons_est_median"]) for r in rows])
    q = np.array([float(r["factor"]) for r in rows])
    coef = np.polyfit(x, q, 2)
    if rows[0].get("factor_curve"):
        assert np.allclose(np.polyval(coef, x), [float(r["factor_curve"]) for r in rows], atol=2e-3)
    return coef, float(np.exp(x.min())), float(np.exp(x.max()))


def factor_at(curve, photons_est):
    coef, lo, hi = curve
    return float(np.polyval(coef, np.log(np.clip(photons_est, lo, hi)))), not (lo <= photons_est <= hi)


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/diffflim_v2.pt")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--factors", default="results/tables/calibration_factors_v2_extra_s10.csv")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    cfg = v3.SimConfigV3(size=256)
    t0 = time.perf_counter()
    recs = load_recordings()
    print(f"{len(recs)} recordings, bin width {recs[0]['bin_width']:.8f} ns, image {recs[0]['mask'].shape}")

    print("IRF alignment (pooled over all masks):")
    shift, irf, _ = align_irf(recs, cfg)
    print(f"  -> shift {shift:+d} bins")

    # ---- baselines (model independent, cached)
    base_cache = PROCESSED / "real_celegans_baselines.npz"
    if base_cache.exists() and not args.recompute:
        with np.load(base_cache) as z:
            base = {k: z[k] for k in z.files}
    else:
        base = {}
        for r in recs:
            for part in PARTS:
                c = r["parts"][part]
                bg = r["bg_full"] * (SPLIT_P if part != "full" else 1.0)
                base[f"{r['name']}|{part}|MLE"] = run_mle(c, r["mask"], bg, irf, cfg)
                base[f"{r['name']}|{part}|Binned MLE 5x5"] = run_mle(c, r["mask"], bg, irf, cfg, k=BIN_K)
                base[f"{r['name']}|{part}|FLIMngo pretrained"] = pre.predict(
                    c[None].astype(np.float32), r["mask"][None], bin_width=r["bin_width"])[0][0]
            base[f"{r['name']}|full|FLIMngo their mask"] = pre.predict(
                r["parts"]["full"][None].astype(np.float32), r["their_mask"][None], bin_width=r["bin_width"])[0][0]
            print(f"  baselines {r['name']} done ({time.perf_counter() - t0:.0f} s)", flush=True)
        np.savez_compressed(base_cache, **base)

    # ---- DiffFLIM (cached per tag)
    dif_cache = PROCESSED / f"real_celegans_diffflim_{args.tag}.npz"
    if dif_cache.exists() and not args.recompute:
        with np.load(dif_cache) as z:
            dif = {k: z[k] for k in z.files}
    else:
        from src.models.diffflim import load_model
        model, _ = load_model(ROOT / args.ckpt, "cuda")
        dif = {}
        for r in recs:
            for part in PARTS:
                t1 = time.perf_counter()
                m, s = run_diffflim_tiled(model, r["parts"][part].astype(np.float32), args.steps, args.samples,
                                          seed0=1000 + SEED_OFFSET[part])
                dif[f"{r['name']}|{part}|mean"], dif[f"{r['name']}|{part}|std"] = m, s
            print(f"  DiffFLIM {r['name']}: {time.perf_counter() - t1:.1f} s per part", flush=True)
        np.savez_compressed(dif_cache, **dif)

    curve = load_factor_curve(ROOT / args.factors)
    print(f"Calibration: {args.factors}, calibrated photon range {curve[1]:.1f}-{curve[2]:.1f}")
    dname = f"DiffFLIM {args.tag}"

    def tau_map(r, part, method):
        if method == "DiffFLIM":
            return dif[f"{r['name']}|{part}|mean"]
        return base[f"{r['name']}|{part}|{method}"]

    # ---- Test A
    rows_a = []
    for r in recs:
        s = r["parts"]["full"].sum(0)
        m = r["mask"]
        est = estimate_photon_level(s, method=PHOTON_ESTIMATOR)
        q, flag = factor_at(curve, est)
        common = {"file": r["name"], "group": r["group"], "mask": MASK_METHOD, "n_mask_px": int(m.sum()),
                  "photons_median": float(np.median(s[m])), "photons_p10": float(np.percentile(s[m], 10)),
                  "photons_p90": float(np.percentile(s[m], 90)), "frac_mask_px_below_10": float(np.mean(s[m] < 10)),
                  "photons_est": est, "photons_est_outside_calib": int(flag), "bg_per_bin": r["bg_full"],
                  "irf_shift_bins": shift}
        for method in METHODS:
            t = tau_map(r, "full", method)[m]
            rows_a.append({"row_type": "image", "method": dname if method == "DiffFLIM" else method, **common,
                           "median_tau_ns": float(np.nanmedian(t)), "mean_tau_ns": float(np.nanmean(t)),
                           "tau_q90_factor": q if method == "DiffFLIM" else ""})
        t = base[f"{r['name']}|full|FLIMngo their mask"]
        rows_a.append({"row_type": "image", "method": "FLIMngo pretrained (their masks, notebook statistic)",
                       **{**common, "mask": "FLIMngo manual", "n_mask_px": int(r["their_mask"].sum())},
                       "median_tau_ns": float(np.median(t[r["their_mask"]])),
                       "mean_tau_ns": float(np.mean(t[t != 0])), "tau_q90_factor": ""})
    tests = []
    for method in [dname, "MLE", "Binned MLE 5x5", "FLIMngo pretrained",
                   "FLIMngo pretrained (their masks, notebook statistic)"]:
        stat = "mean_tau_ns" if "notebook" in method else "median_tau_ns"
        a = [x[stat] for x in rows_a if x["method"] == method and x["group"] == "abeta"]
        b = [x[stat] for x in rows_a if x["method"] == method and x["group"] == "mscarlet"]
        u, p = mannwhitneyu(a, b, alternative="two-sided", method="exact")
        tests.append({"row_type": "test", "method": method, "statistic": stat, "n_abeta": len(a), "n_mscarlet": len(b),
                      "median_abeta_ns": float(np.median(a)), "median_mscarlet_ns": float(np.median(b)),
                      "difference_ns": float(np.median(a) - np.median(b)), "U": float(u), "p_value": float(p)})
        print(f"Test A {method:52s} Abeta {np.median(a):.3f} ns, mScarlet {np.median(b):.3f} ns, U {u:.0f}, p {p:.4f}")
    write_csv(TABLES / f"real_celegans_groups_{args.tag}.csv", rows_a + tests)

    # ---- Test B
    rows_b = []
    for r in recs:
        m = r["mask"]
        est = {p: estimate_photon_level(r["parts"][p].sum(0), method=PHOTON_ESTIMATOR) for p in PARTS}
        for method in METHODS:
            ta, tb, tf = (tau_map(r, p, method)[m] for p in PARTS)
            ok = np.isfinite(ta) & np.isfinite(tb) & np.isfinite(tf)
            ta, tb, tf = ta[ok], tb[ok], tf[ok]
            row = {"row_type": "image", "file": r["name"], "group": r["group"],
                   "method": dname if method == "DiffFLIM" else method, "n_px": int(ok.sum()),
                   "n_px_invalid": int((~ok).sum()), "photons_est_A": est["A"], "photons_est_B": est["B"],
                   "median_tau_A_ns": float(np.median(ta)), "median_tau_B_ns": float(np.median(tb)),
                   "median_tau_full_ns": float(np.median(tf)),
                   "mad_AB_ns": float(np.median(np.abs(ta - tb))),
                   "mad_AB_rel_pct": 100 * float(np.median(np.abs(ta - tb) / np.maximum((ta + tb) / 2, 1e-6))),
                   "mad_A_full_ns": float(np.median(np.abs(ta - tf))), "mad_B_full_ns": float(np.median(np.abs(tb - tf)))}
            if method == "DiffFLIM":
                qa, fa = factor_at(curve, est["A"])
                qb, fb = factor_at(curve, est["B"])
                sa = dif[f"{r['name']}|A|std"][m][ok]
                sb = dif[f"{r['name']}|B|std"][m][ok]
                ha, hb = qa * sa, qb * sb
                d = np.abs(ta - tb)
                row.update({"q90_A": qa, "q90_B": qb, "photons_est_outside_calib": int(fa or fb),
                            "median_halfwidth90_A_ns": float(np.median(ha)),
                            "overlap90_frac": float(np.mean(d <= ha + hb)),
                            "B_in_A90_frac": float(np.mean(d <= ha)), "A_in_B90_frac": float(np.mean(d <= hb)),
                            "full_in_A90_frac": float(np.mean(np.abs(tf - ta) <= ha)),
                            "full_in_B90_frac": float(np.mean(np.abs(tf - tb) <= hb))})
            rows_b.append(row)
    keys = [k for k in rows_b[0] if k.startswith(("mad_", "median_tau"))] + \
        ["overlap90_frac", "B_in_A90_frac", "A_in_B90_frac", "full_in_A90_frac", "full_in_B90_frac",
         "median_halfwidth90_A_ns"]
    for method in [dname, "MLE", "Binned MLE 5x5", "FLIMngo pretrained"]:
        for grp in ("all",) + GROUPS:
            sel = [x for x in rows_b if x["method"] == method and (grp == "all" or x["group"] == grp)]
            summ = {"row_type": "median over recordings", "file": "", "group": grp, "method": method,
                    "n_px": sum(x["n_px"] for x in sel)}
            for k in keys:
                if k in sel[0]:
                    summ[k] = float(np.median([x[k] for x in sel]))
            rows_b.append(summ)
            if grp == "all":
                print(f"Test B {method:20s} median |A-B| {summ['mad_AB_ns']:.3f} ns ({summ['mad_AB_rel_pct']:.1f} %), "
                      f"|A-full| {summ['mad_A_full_ns']:.3f} ns"
                      + (f", overlap {summ['overlap90_frac']:.3f}, B in A {summ['B_in_A90_frac']:.3f}, "
                         f"full in A {summ['full_in_A90_frac']:.3f}" if "overlap90_frac" in summ else ""))
    write_csv(TABLES / f"real_celegans_splithalf_{args.tag}.csv", rows_b)

    from src.models.diffflim import load_model
    model, _ = load_model(ROOT / args.ckpt, "cuda")
    write_csv(TABLES / f"real_celegans_splithalf_simref_{args.tag}.csv",
              sim_reference(model, curve, args.steps, args.samples))
    for r in recs:     # real data: relative half-width, for comparison with the simulated reference
        m = r["mask"]
        q, _ = factor_at(curve, estimate_photon_level(r["parts"]["A"].sum(0), method=PHOTON_ESTIMATOR))
        r["hw_rel"] = float(np.median(q * dif[f"{r['name']}|A|std"][m] / dif[f"{r['name']}|A|mean"][m]))
    print("Real data: median relative 90 % half-width of half A: "
          + ", ".join(f"{r['name']} {100 * r['hw_rel']:.1f} %" for r in recs))
    plot_groups(rows_a, tests, dname, FIGURES / f"real_celegans_groups_{args.tag}.png")
    plot_examples(recs, base, dif, curve, dname, FIGURES / f"real_celegans_examples_{args.tag}.png")
    print(f"Done in {time.perf_counter() - t0:.0f} s")


def sim_reference(model, curve, steps, n_samples, levels=(20, 50, 100), n_img=10):
    """Split-half statistics of DiffFLIM on simulated v3 test images with known truth."""
    from src.data.simulate import load_dataset
    rows = []
    for lev in levels:
        d = load_dataset(ROOT / "data" / "raw" / "own" / "sim_v3_test" / f"test_p{lev}.npz")
        acc = {k: [] for k in ("d", "rel", "overlap", "b_in_a", "cov_a", "hw_rel", "err_a")}
        for i in range(n_img):
            c, m, gt = d["counts"][i].astype(np.int64), d["mask"][i], d["tau_gt"][i]
            rng = np.random.default_rng(SPLIT_SEED + 100000 + 100 * lev + i)
            half_a = rng.binomial(c, SPLIT_P)
            out = {}
            for name, cc in (("A", half_a), ("B", c - half_a)):
                smp = model.sample(cc.astype(np.float32), n_samples=n_samples, steps=steps,
                                   seed=1000 + SEED_OFFSET[name] + i)
                q, _ = factor_at(curve, estimate_photon_level(cc.sum(0), method=PHOTON_ESTIMATOR))
                out[name] = (smp[:, 0].mean(0)[m], q * smp[:, 0].std(0, ddof=1)[m])
            (ma, ha), (mb, hb), t = out["A"], out["B"], gt[m]
            dd = np.abs(ma - mb)
            acc["d"].append(dd); acc["rel"].append(dd / ((ma + mb) / 2)); acc["overlap"].append(dd <= ha + hb)
            acc["b_in_a"].append(dd <= ha); acc["cov_a"].append(np.abs(t - ma) <= ha)
            acc["hw_rel"].append(ha / ma); acc["err_a"].append(np.abs(t - ma) / t)
        cat = {k: np.concatenate(v) for k, v in acc.items()}
        rows.append({"photons_full": lev, "photons_half": lev * SPLIT_P, "n_images": n_img,
                     "mad_AB_rel_pct": 100 * float(np.median(cat["rel"])),
                     "overlap90_frac": float(cat["overlap"].mean()), "B_in_A90_frac": float(cat["b_in_a"].mean()),
                     "coverage90_A_vs_truth": float(cat["cov_a"].mean()),
                     "median_halfwidth90_rel_pct": 100 * float(np.median(cat["hw_rel"])),
                     "mare_A_vs_truth_pct": 100 * float(np.median(cat["err_a"]))})
        r = rows[-1]
        print(f"Sim. reference {lev} photons (halves {lev * SPLIT_P:g}): |A-B| {r['mad_AB_rel_pct']:.1f} %, "
              f"overlap {r['overlap90_frac']:.3f}, B in A {r['B_in_A90_frac']:.3f}, coverage of A vs truth "
              f"{r['coverage90_A_vs_truth']:.3f}, half-width {r['median_halfwidth90_rel_pct']:.1f} %, "
              f"error of A {r['mare_A_vs_truth_pct']:.1f} %")
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved {path}")


def plot_groups(rows, tests, dname, out):
    methods = ["MLE", "Binned MLE 5x5", "FLIMngo pretrained", dname,
               "FLIMngo pretrained (their masks, notebook statistic)"]
    titles = {methods[-1]: "FLIMngo, their masks\n(mean, as their notebook)"}
    fig, ax = plt.subplots(1, len(methods), figsize=(3.1 * len(methods), 3.8), sharey=False)
    rng = np.random.default_rng(0)
    for a, m in zip(ax, methods):
        stat = "mean_tau_ns" if "notebook" in m else "median_tau_ns"
        vals = [[x[stat] for x in rows if x["method"] == m and x["group"] == g] for g in GROUPS]
        a.boxplot(vals, widths=0.5, showfliers=False)
        for i, v in enumerate(vals):
            a.scatter(i + 1 + rng.uniform(-0.08, 0.08, len(v)), v, s=22, color=["C3", "C0"][i], zorder=3,
                      edgecolor="k", linewidth=0.5)
        t = next(x for x in tests if x["method"] == m)
        a.set_xticks([1, 2])
        a.set_xticklabels(["Abeta-mScarlet", "mScarlet"], fontsize=8)
        a.set_title(f"{titles.get(m, m)}\nMann-Whitney p = {t['p_value']:.3f}", fontsize=9)
        a.grid(alpha=0.3, axis="y")
    ax[0].set_ylabel("Lifetime per recording (ns)\n(median inside the mask)")
    fig.suptitle("FLIMngo C. elegans recordings (1 s, unparalysed): Abeta-mScarlet versus mScarlet, 6 versus 6",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def plot_examples(recs, base, dif, curve, dname, out):
    sel = [r for r in recs if r["name"] in EXAMPLES]
    cols = ["Summed counts", "MLE", "Binned MLE 5x5", "FLIMngo pretrained", f"{dname} mean",
            f"{dname} 90 % half-width"]
    fig, ax = plt.subplots(len(sel), len(cols), figsize=(3.0 * len(cols), 3.0 * len(sel)), squeeze=False)
    for row, r in zip(ax, sel):
        m = r["mask"]
        ys, xs = np.where(m)
        y0, y1 = max(ys.min() - 8, 0), min(ys.max() + 9, m.shape[0])
        x0, x1 = max(xs.min() - 8, 0), min(xs.max() + 9, m.shape[1])
        s = r["parts"]["full"].sum(0)
        q, _ = factor_at(curve, estimate_photon_level(s, method=PHOTON_ESTIMATOR))
        maps = [np.log10(1 + s), base[f"{r['name']}|full|MLE"], base[f"{r['name']}|full|Binned MLE 5x5"],
                base[f"{r['name']}|full|FLIMngo pretrained"], dif[f"{r['name']}|full|mean"],
                q * dif[f"{r['name']}|full|std"]]
        for k, (a, img, title) in enumerate(zip(row, maps, cols)):
            img = img if k == 0 else np.where(m, img, np.nan)
            kw = {"cmap": "magma"} if k == 0 else ({"cmap": "viridis", "vmin": TAU_DISPLAY[0], "vmax": TAU_DISPLAY[1]}
                                                    if k < 5 else {"cmap": "inferno", "vmin": HALFWIDTH_DISPLAY[0], "vmax": HALFWIDTH_DISPLAY[1]})
            im = a.imshow(img[y0:y1, x0:x1], **kw)
            fig.colorbar(im, ax=a, fraction=0.046)
            a.set_title(("log10(1 + counts)" if k == 0 else title + (" (ns)" if k < 5 else ", ns")), fontsize=8)
            a.axis("off")
        row[0].set_title(f"{r['name']}\nlog10(1 + summed counts)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
