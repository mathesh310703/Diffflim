"""Split-conformal calibration of DiffFLIM uncertainty, evaluated on a fixed test set.

Steps
  1. Calibration set: an existing folder with calib_p*.npz (simulator v3: data/raw/own/sim_v3_calib,
     made by `python -m src.data.simulate_v3`), or -- for the old v2 simulator -- generated here
     in data/raw/own/sim_calib/ if missing.
  2. Predictions (N samples of --steps DDIM steps -> mean, std) on calibration and test images,
     cached in data/processed/preds_{tag}_{calib,test}.npz for 50 steps / 10 samples and in
     preds_{tag}_s{steps}_n{samples}_{calib,test}.npz otherwise (delete them or use --recompute).
     Default since 2026-11-22: 10 DDIM steps, 10 samples (50 steps before; use --steps 50).
  3. Factors (src/evaluation/calibration.py): global, and per photon level as a curve in
     log(photons) (--smooth: quadratic fit; otherwise linear interpolation).
  4. Coverage + relative interval width before/after calibration on the test set, per level and
     pooled, at 90 % and 68 %, with image-level bootstrap 95 % confidence intervals (images are
     resampled, because pixels of one image are strongly correlated).

Outputs
  results/tables/calibration_factors_{tag}.csv
  results/tables/coverage_calibrated_{tag}.csv
  results/figures/calibration_{tag}.png            promised vs observed coverage (pooled)
  results/figures/calibration_{tag}_vs_photons.png coverage vs photon level (with bootstrap CIs)

Several calibration folders can be pooled (--calib-dir a,b). Predictions are cached per folder:
data/processed/preds_{preds_tag}_calib.npz for sim_calib / sim_v3_calib (as before) and
preds_{preds_tag}_calib_{folder name}.npz for any other folder. --photon-estimator chooses the
photon-level estimate used by the per-level factors (src/evaluation/calibration.py); it is
recomputed from the counts, so cached model predictions are reused. Example (v2, pooled cells):
  python scripts/calibrate_diffflim.py --ckpt checkpoints/diffflim_v2.pt --tag v2_extra --preds-tag v2
      --smooth --photon-estimator otsu_adaptive --test-set data/raw/own/sim_v3_test
      --calib-dir data/raw/own/sim_v3_calib,data/raw/own/sim_v3_calib_extra

Guidance (24 November 2026): --guide-kappa K --guide-stop S runs every prediction with the Poisson
data-consistency guidance of src/models/guidance.py (forward model of the data set with each image's IRF
and background, as for the baselines). Give it its own --preds-tag.

Usage:
  python scripts/calibrate_diffflim.py                                   # v1 model, v2 simulator
  python scripts/calibrate_diffflim.py --ckpt checkpoints/diffflim_v2.pt --tag v2 --smooth \
      --test-set data/raw/own/sim_v3_test --calib-dir data/raw/own/sim_v3_calib
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
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.simulate import generate_dataset, load_dataset, save_dataset, test_set_dir  # noqa: E402
from src.evaluation import calibration as cal  # noqa: E402
from src.paths import CHECKPOINTS, FIGURES, PROCESSED, RAW_OWN, TABLES  # noqa: E402

LEVELS = (10, 20, 50, 100, 500, 1000)
CALIB_SEED = 54321          # v2-simulator calibration set (test set uses 12345 + level)
ALPHAS = (0.10, 0.32)
VARIANTS = {"none": "uncalibrated", "global": "(a) global", "per_level": "(b) per photon level"}
QNAMES = {"tau": "Lifetime", "int": "Intensity"}
N_BOOT = 1000


def make_v2_calib_set(out_dir, n_per_level):
    for lev in LEVELS:
        path = out_dir / f"calib_p{lev}.npz"
        if not path.exists():
            save_dataset(path, generate_dataset(n_per_level, lev, seed=CALIB_SEED + lev))
            print(f"Saved {path}")


def calib_cache_key(folder):
    """Cache name of a calibration folder (the two original folders keep their old name)."""
    return "calib" if folder.name in ("sim_calib", "sim_v3_calib") else f"calib_{folder.name}"


def recompute_photons_est(rec, files, n_images, method):
    """Replace rec["photons_est"] by the estimate of `method`, from the counts in `files` (same
    image order as predict_split)."""
    est = []
    for path in files:
        d = load_dataset(path)
        n = min(n_images or len(d["counts"]), len(d["counts"]))
        est += [cal.estimate_photon_level(d["counts"][i].sum(0), method=method) for i in range(n)]
    assert len(est) == len(rec["photons"]), "cached predictions do not match the data files"
    rec["photons_est"] = np.array(est)
    return rec


def concat_records(recs):
    return {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}


def preds_cache(ptag, key, steps, n_samples):
    """Prediction cache. 50 steps / 10 samples keep the old names (preds_{ptag}_{key}.npz)."""
    if (steps, n_samples) == (50, 10):
        return PROCESSED / f"preds_{ptag}_{key}.npz"
    return PROCESSED / f"preds_{ptag}_s{steps}_n{n_samples}_{key}.npz"


def image_guidance(d, i, guide):
    """Poisson data-consistency guidance for image i of data set d (guide = (kappa, stop) or None)."""
    if guide is None:
        return None
    from src.baselines.common import forward_setup
    from src.models.guidance import PoissonGuidance
    fn = forward_setup(d["metadata"], d["irf"][i] if "irf" in d else None)[1]["decay_fn"]
    return PoissonGuidance(fn, float(d["bg_per_bin"][i]), guide[0], stop_frac=guide[1])


def predict_split(model, files, n_images, n_samples, seed0, steps=10, guide=None):
    """Run the model on every image -> prediction record (see calibration.py)."""
    keys = ["tau_mean", "tau_std", "int_mean", "int_std", "tau_gt", "int_gt", "mask", "photons", "photons_est"]
    rec = {k: [] for k in keys}
    for path in files:
        d = load_dataset(path)
        n = min(n_images or len(d["counts"]), len(d["counts"]))
        t0 = time.perf_counter()
        for i in range(n):
            s = model.sample(d["counts"][i], n_samples=n_samples, steps=steps, seed=seed0 + i,
                             guidance=image_guidance(d, i, guide))
            rec["tau_mean"].append(s[:, 0].mean(0)); rec["tau_std"].append(s[:, 0].std(0, ddof=1))
            rec["int_mean"].append(s[:, 1].mean(0)); rec["int_std"].append(s[:, 1].std(0, ddof=1))
            rec["tau_gt"].append(d["tau_gt"][i]); rec["int_gt"].append(d["intensity_gt"][i])
            rec["mask"].append(d["mask"][i]); rec["photons"].append(float(d["photons"][i]))
            rec["photons_est"].append(cal.estimate_photon_level(d["counts"][i].sum(0)))
        print(f"  {path.name}: {n} images, {time.perf_counter() - t0:.0f} s", flush=True)
    return {k: np.stack(v) if np.ndim(v[0]) else np.array(v) for k, v in rec.items()}


def load_or_predict(path, model_fn, files, n_images, n_samples, seed0, recompute, steps=10, guide=None):
    if path.exists() and not recompute:
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    rec = predict_split(model_fn(), files, n_images, n_samples, seed0, steps, guide)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **rec)
    return rec


def per_image_hits(rec, q, factor):
    """(pixels inside the interval, foreground pixels, sum of relative widths) per image."""
    m = rec["mask"]
    f = np.asarray(factor, dtype=np.float64).reshape(-1, 1, 1) * np.ones_like(rec[f"{q}_std"])
    half = f * np.maximum(rec[f"{q}_std"], cal.STD_FLOOR)
    inside = (np.abs(rec[f"{q}_gt"] - rec[f"{q}_mean"]) <= half) & m
    width = np.where(m, 2 * half / np.maximum(np.abs(rec[f"{q}_gt"]), cal.STD_FLOOR), 0.0)
    return inside.sum((1, 2)).astype(float), m.sum((1, 2)).astype(float), width.sum((1, 2))


def coverage_with_ci(hit, tot, wsum, idx, rng):
    """Pixel coverage and mean relative width over images idx, with image-bootstrap 95 % CI."""
    h, t, w = hit[idx], tot[idx], wsum[idx]
    cov, width = h.sum() / t.sum(), w.sum() / t.sum()
    b = rng.integers(0, len(h), size=(N_BOOT, len(h)))
    boot = h[b].sum(1) / t[b].sum(1)
    return cov, np.percentile(boot, 2.5), np.percentile(boot, 97.5), width


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CHECKPOINTS / "diffflim_v1.pt"))
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--test-set", default=None, help="folder with test_p*.npz (default: v2 test set)")
    ap.add_argument("--calib-dir", default=str(RAW_OWN / "sim_calib"), help="one folder or several, comma-separated")
    ap.add_argument("--preds-tag", default=None, help="tag of the prediction caches (default: --tag)")
    ap.add_argument("--photon-estimator", default="otsu_log", choices=cal.PHOTON_ESTIMATORS)
    ap.add_argument("--calib-per-level", type=int, default=20, help="only used to create a v2 calibration set")
    ap.add_argument("--n-test", type=int, default=0, help="test images per level (0 = all)")
    ap.add_argument("--n-calib", type=int, default=0, help="calibration images per level (0 = all)")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--steps", type=int, default=10, help="DDIM steps (default 10 since 2026-11-22; 50 before)")
    ap.add_argument("--smooth", action="store_true", help="per-level factors as a quadratic in log(photons)")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--guide-kappa", type=float, default=None,
                    help="Poisson data-consistency guidance (src/models/guidance.py) with this prior precision; "
                         "use a separate --preds-tag, the caches do not record the setting")
    ap.add_argument("--guide-stop", type=float, default=1.0, help="guide the first fraction of the DDIM steps")
    args = ap.parse_args()

    calib_dirs = [Path(c) for c in args.calib_dir.split(",")]
    if calib_dirs == [RAW_OWN / "sim_calib"]:
        make_v2_calib_set(calib_dirs[0], args.calib_per_level)
    order = lambda p: int(p.stem.split("_p")[1])  # noqa: E731
    calib_files = {c: sorted(c.glob("calib_p*.npz"), key=order) for c in calib_dirs}
    test_dir = Path(args.test_set) if args.test_set else test_set_dir()
    test_files = sorted(test_dir.glob("test_p*.npz"), key=order)
    if not all(calib_files.values()) or not test_files:
        sys.exit(f"Missing calibration ({calib_dirs}) or test ({test_dir}) files.")
    ptag = args.preds_tag or args.tag
    guide = None if args.guide_kappa is None else (args.guide_kappa, args.guide_stop)

    def model_fn():
        from src.models.diffflim import load_model
        model, ckpt = load_model(args.ckpt, "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loaded {args.ckpt} (step {ckpt['step']})")
        return model

    recs = []
    for k, (c, files) in enumerate(calib_files.items()):
        print(f"Predicting calibration set {c}")
        key = calib_cache_key(c)
        rec = load_or_predict(preds_cache(ptag, key, args.steps, args.n_samples), model_fn, files,
                              args.n_calib, args.n_samples, 5000 if key == "calib" else 7000, args.recompute,
                              args.steps, guide)
        if args.photon_estimator != "otsu_log":
            rec = recompute_photons_est(rec, files, args.n_calib, args.photon_estimator)
        print(f"  {len(rec['photons'])} images")
        recs.append(rec)
    calib = concat_records(recs)
    print("Predicting test set")
    test = load_or_predict(preds_cache(ptag, "test", args.steps, args.n_samples), model_fn, test_files,
                           args.n_test, args.n_samples, 1000, args.recompute, args.steps, guide)
    if args.photon_estimator != "otsu_log":
        test = recompute_photons_est(test, test_files, args.n_test, args.photon_estimator)

    for name, rec in [("calibration", calib), ("test", test)]:
        ratio = rec["photons_est"] / rec["photons"]
        print(f"Photon-level estimate ({args.photon_estimator}) / true level on {name}: median {np.median(ratio):.3f}, "
              f"range {ratio.min():.2f}-{ratio.max():.2f}, below 0.8: {np.mean(ratio < 0.8):.1%}, "
              f"above 1.25: {np.mean(ratio > 1.25):.1%}")

    fitted = cal.fit(calib, ALPHAS, smooth=args.smooth)

    # ---- factors table
    rows = []
    for q in cal.QUANTITIES:
        for a in ALPHAS:
            z = float(cal.predict_q(fitted, q, a, "none"))
            rows.append({"quantity": q, "nominal_coverage": 1 - a, "variant": "global", "photons": "all",
                         "photons_est_median": "", "factor": fitted["global"][(q, a)], "factor_curve": "",
                         "gaussian_z": z, "n_scores": fitted["n"][(q, "all")]})
            curve = cal.predict_q(fitted, q, a, "per_level", np.array(fitted["level_x"]))
            for lev, x, c in zip(fitted["levels"], fitted["level_x"], curve):
                rows.append({"quantity": q, "nominal_coverage": 1 - a, "variant": "per_level",
                             "photons": int(lev), "photons_est_median": x,
                             "factor": fitted["per_level"][(q, a, float(lev))], "factor_curve": float(c),
                             "gaussian_z": z, "n_scores": fitted["n"][(q, float(lev))]})
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    f_factors = TABLES / f"calibration_factors_{args.tag}.csv"
    write_csv(f_factors, rows)

    # ---- coverage on the test set, per level and pooled, with image-bootstrap CIs
    rng = np.random.default_rng(0)
    groups = [(int(l), np.where(test["photons"] == l)[0]) for l in LEVELS if np.any(test["photons"] == l)]
    groups.append(("all", np.arange(len(test["photons"]))))
    cov_rows = []
    for q in cal.QUANTITIES:
        for a in ALPHAS:
            per_img = {v: per_image_hits(test, q, cal.predict_q(fitted, q, a, v, test["photons_est"])) for v in VARIANTS}
            for lev, idx in groups:
                row = {"photons": lev, "quantity": q, "nominal_coverage": 1 - a, "n_images": len(idx)}
                for v in VARIANTS:
                    c, lo, hi, w = coverage_with_ci(*per_img[v], idx, rng)
                    row.update({f"cov_{v}": c, f"cov_{v}_ci_lo": lo, f"cov_{v}_ci_hi": hi, f"relwidth_{v}": w})
                cov_rows.append(row)
    f_cov = TABLES / f"coverage_calibrated_{args.tag}.csv"
    write_csv(f_cov, cov_rows)

    # ---- figure 1: promised vs observed (all levels pooled)
    promised = np.linspace(0.10, 0.95, 18)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    for a_, q in zip(ax, cal.QUANTITIES):
        for v, style in zip(VARIANTS, ["o-", "s-", "D-"]):
            obs = [cal.interval_stats(test, q, cal.predict_q(fitted, q, 1 - p, v, test["photons_est"]))[0]
                   for p in promised]
            a_.plot(100 * promised, 100 * np.array(obs), style, ms=4, label=VARIANTS[v])
        a_.plot([0, 100], [0, 100], "k--", lw=1, label="perfect")
        a_.set_xlim(5, 100); a_.set_ylim(0, 100); a_.set_aspect("equal")
        a_.set_xlabel("Promised coverage (%)"); a_.set_ylabel("Observed coverage (%)")
        a_.set_title(f"{QNAMES[q]}: test set, all photon levels"); a_.grid(alpha=0.3); a_.legend(fontsize=8)
    fig.tight_layout()
    f_fig1 = FIGURES / f"calibration_{args.tag}.png"
    fig.savefig(f_fig1, dpi=120)

    # ---- figure 2: coverage vs photon level with bootstrap CIs
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for j, a in enumerate(ALPHAS):
        for i, q in enumerate(cal.QUANTITIES):
            r = [x for x in cov_rows if x["quantity"] == q and np.isclose(x["nominal_coverage"], 1 - a)
                 and x["photons"] != "all"]
            p = np.array([x["photons"] for x in r], dtype=float)
            for k, (v, style) in enumerate(zip(VARIANTS, ["o-", "s-", "D-"])):
                y = np.array([x[f"cov_{v}"] for x in r]) * 100
                err = np.array([[x[f"cov_{v}"] - x[f"cov_{v}_ci_lo"], x[f"cov_{v}_ci_hi"] - x[f"cov_{v}"]] for x in r]).T * 100
                ax[j, i].errorbar(p * (1 + 0.04 * (k - 1)), y, yerr=err, fmt=style, capsize=3, label=VARIANTS[v])
            ax[j, i].axhline(100 * (1 - a), color="k", ls="--", lw=1, label="target")
            ax[j, i].set_xscale("log"); ax[j, i].grid(alpha=0.3, which="both")
            ax[j, i].set_title(f"{QNAMES[q]}, nominal {100 * (1 - a):.0f}% (95% image-bootstrap CI)", fontsize=10)
            ax[j, i].set_ylabel("Observed coverage (%)")
            if j == 1:
                ax[j, i].set_xlabel("Photons per foreground pixel")
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    f_fig2 = FIGURES / f"calibration_{args.tag}_vs_photons.png"
    fig.savefig(f_fig2, dpi=120)

    # ---- console summary
    print("\nFactors (gaussian z: 1.645 @90%, 0.994 @68%)")
    for r in rows:
        extra = f"  curve {r['factor_curve']:.3f}" if r["factor_curve"] != "" else ""
        print(f"  {r['quantity']:3s} {r['nominal_coverage']:.2f} {r['variant']:9s} {str(r['photons']):>5s}  "
              f"q={r['factor']:.3f}{extra}")
    print("\nCoverage on test [95% CI]: uncalibrated | global | per level      rel. width")
    for r in cov_rows:
        cells = " | ".join(f"{r[f'cov_{v}']:.3f} [{r[f'cov_{v}_ci_lo']:.2f},{r[f'cov_{v}_ci_hi']:.2f}]" for v in VARIANTS)
        widths = " / ".join(f"{r[f'relwidth_{v}']:.3f}" for v in VARIANTS)
        print(f"  p={str(r['photons']):>5s} {r['quantity']:3s} {r['nominal_coverage']:.2f}: {cells}   {widths}")
    print("\n" + "\n".join(f"Saved {p}" for p in [f_factors, f_cov, f_fig1, f_fig2]))


if __name__ == "__main__":
    main()
