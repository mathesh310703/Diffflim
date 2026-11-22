"""Cell-level confidence intervals for the calibrated coverage of DiffFLIM v2.

The image-level bootstrap in calibrate_diffflim.py resamples test IMAGES. Images of the same HPA
cell (the same crop source, and the same scene at every photon level, since the test set is
paired) are not independent, so it understates the uncertainty. Here the unit is the HPA cell
(hpa_id):

  (1) test-cell bootstrap: the factors are fixed (fitted once on all 74 calibration cells), and the
      30 test cells are resampled with replacement; all images of a drawn cell enter together.
  (2) joint bootstrap: the 74 calibration cells AND the 30 test cells are resampled; the factors are
      refitted on every replicate, so the interval also contains the uncertainty of the factors.

Calibration as in coverage_calibrated_v2_extra: cached predictions of DiffFLIM v2
(data/processed/preds_v2_*.npz), photon estimate "otsu_adaptive", global factor and per-level
factors as a quadratic in log(estimated photons).

Writes results/tables/coverage_cellboot_v2.csv (50 DDIM steps, 10 samples) or
       results/tables/coverage_cellboot_v2_s{steps}_n{samples}.csv (other settings; default 10 steps).

Usage:
    python scripts/coverage_cell_bootstrap.py               # 10 DDIM steps (default since 2026-11-22)
    python scripts/coverage_cell_bootstrap.py --steps 50    # previous setting
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import calibrate_diffflim as cd  # noqa: E402
from src.data.simulate import load_dataset  # noqa: E402
from src.evaluation import calibration as cal  # noqa: E402
from src.paths import RAW_OWN, TABLES  # noqa: E402

ALPHAS = (0.10, 0.32)
LEVELS = (10, 20, 50, 100, 500, 1000)
N_BOOT_TEST = 5000
N_BOOT_JOINT = 500
EST = "otsu_adaptive"


def files_of(folder, prefix):
    return sorted(folder.glob(f"{prefix}_p*.npz"), key=lambda p: int(p.stem.split("_p")[1]))


def hpa_ids(files):
    return np.concatenate([np.asarray(load_dataset(f)["hpa_id"]).astype(str) for f in files])


def load_rec(cache, files):
    with np.load(cache) as f:
        rec = {k: f[k] for k in f.files}
    rec = cd.recompute_photons_est(rec, files, 0, EST)
    rec["hpa_id"] = hpa_ids(files)
    assert len(rec["hpa_id"]) == len(rec["photons"])
    return rec


def image_scores(rec, q):
    """Sorted normalised errors |gt - mean| / std on the foreground, one array per image."""
    out = []
    for i in range(len(rec["photons"])):
        m = rec["mask"][i]
        s = np.abs(rec[f"{q}_gt"][i] - rec[f"{q}_mean"][i])[m] / np.maximum(rec[f"{q}_std"][i][m], cal.STD_FLOOR)
        out.append(np.sort(s))
    return out


def conformal_quantile(s, alpha):
    """Same as calibration.conformal_quantile, with np.partition instead of a full sort."""
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return float(np.partition(s, k - 1)[k - 1]) if k <= len(s) else float("inf")


def fit_factors(score_lists, photons, photons_est, alpha):
    """Global and per-level conformal factors from a list of per-image score arrays."""
    allsc = np.concatenate(score_lists)
    g = conformal_quantile(allsc, alpha)
    lev_q, lev_x = [], []
    for lev in LEVELS:
        idx = np.where(photons == lev)[0]
        lev_q.append(conformal_quantile(np.concatenate([score_lists[i] for i in idx]), alpha))
        lev_x.append(np.median(photons_est[idx]))
    x, qs = np.log(lev_x), np.array(lev_q)
    coef = np.polyfit(x, qs, 2)
    return g, (coef, x.min(), x.max())


def per_level_factor(curve, photons_est):
    coef, lo, hi = curve
    return np.polyval(coef, np.clip(np.log(photons_est), lo, hi))


def hits_for(test_scores, factors):
    """(pixels inside, foreground pixels) per test image for per-image factors."""
    h = np.array([np.searchsorted(s, f, side="right") for s, f in zip(test_scores, factors)], float)
    t = np.array([len(s) for s in test_scores], float)
    return h, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10, help="DDIM steps of the cached predictions (10 default, 50 before)")
    ap.add_argument("--n-samples", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(2026)
    cal_dirs = [RAW_OWN / "sim_v3_calib", RAW_OWN / "sim_v3_calib_extra"]
    calib = cd.concat_records([load_rec(cd.preds_cache("v2", cd.calib_cache_key(d), args.steps, args.n_samples),
                                        files_of(d, "calib"))
                               for d in cal_dirs])
    test_files = files_of(RAW_OWN / "sim_v3_test", "test")
    test = load_rec(cd.preds_cache("v2", "test", args.steps, args.n_samples), test_files)
    cal_cells, cal_cell_idx = np.unique(calib["hpa_id"], return_inverse=True)
    test_cells, test_cell_idx = np.unique(test["hpa_id"], return_inverse=True)
    print(f"Calibration: {len(calib['photons'])} images from {len(cal_cells)} cells; "
          f"test: {len(test['photons'])} images from {len(test_cells)} cells")
    test_levels = test["photons"].astype(int)

    rows = []
    for q in cal.QUANTITIES:
        cs, ts = image_scores(calib, q), image_scores(test, q)
        for a in ALPHAS:
            g, curve = fit_factors(cs, calib["photons"].astype(int), calib["photons_est"], a)
            factors = {"none": np.full(len(ts), cal.predict_q(None, q, a, "none")),
                       "global": np.full(len(ts), g),
                       "per_level": per_level_factor(curve, test["photons_est"])}
            hits = {v: hits_for(ts, f) for v, f in factors.items()}
            # per cell x level sums -> fast bootstrap with multinomial cell weights
            H = {v: np.zeros((len(test_cells), len(LEVELS))) for v in factors}
            T = np.zeros((len(test_cells), len(LEVELS)))
            for i in range(len(ts)):
                j = LEVELS.index(test_levels[i])
                T[test_cell_idx[i], j] += hits["none"][1][i]
                for v in factors:
                    H[v][test_cell_idx[i], j] += hits[v][0][i]
            W = rng.multinomial(len(test_cells), np.full(len(test_cells), 1 / len(test_cells)), size=N_BOOT_TEST)

            # joint bootstrap: resample calibration cells too and refit the factors
            cal_by_cell = [np.where(cal_cell_idx == k)[0] for k in range(len(cal_cells))]
            joint = {v: [] for v in ("global", "per_level")}
            joint_lev = {v: [] for v in ("global", "per_level")}
            for b in range(N_BOOT_JOINT):
                cdraw = rng.integers(0, len(cal_cells), len(cal_cells))
                idx = np.concatenate([cal_by_cell[k] for k in cdraw])
                gb, cb = fit_factors([cs[i] for i in idx], calib["photons"][idx].astype(int),
                                     calib["photons_est"][idx], a)
                fb = {"global": np.full(len(ts), gb), "per_level": per_level_factor(cb, test["photons_est"])}
                w = W[b]
                for v in fb:
                    h, t = hits_for(ts, fb[v])
                    hc = np.zeros((len(test_cells), len(LEVELS))); tc = np.zeros_like(hc)
                    for i in range(len(ts)):
                        hc[test_cell_idx[i], LEVELS.index(test_levels[i])] += h[i]
                        tc[test_cell_idx[i], LEVELS.index(test_levels[i])] += t[i]
                    joint[v].append((w @ hc).sum() / (w @ tc).sum())
                    joint_lev[v].append((w @ hc) / (w @ tc))
            for lev_j, lev in [(None, "all")] + list(enumerate(LEVELS)):
                sl = slice(None) if lev_j is None else slice(lev_j, lev_j + 1)
                for v in factors:
                    cov = H[v][:, sl].sum() / T[:, sl].sum()
                    boot = (W @ H[v][:, sl]).sum(1) / (W @ T[:, sl]).sum(1)
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    row = {"quantity": q, "nominal_coverage": 1 - a, "photons": lev, "variant": v,
                           "coverage": cov, "testcell_ci_lo": lo, "testcell_ci_hi": hi,
                           "joint_ci_lo": "", "joint_ci_hi": "", "n_test_cells": len(test_cells),
                           "n_calib_cells": len(cal_cells)}
                    if v in joint:
                        jb = np.array(joint[v]) if lev_j is None else np.array(joint_lev[v])[:, lev_j]
                        row["joint_ci_lo"], row["joint_ci_hi"] = np.percentile(jb, [2.5, 97.5])
                    rows.append(row)
                    if lev == "all":
                        j = f"  joint [{row['joint_ci_lo']:.3f}, {row['joint_ci_hi']:.3f}]" if v in joint else ""
                        print(f"{q:3s} {1 - a:.2f} {v:9s} all: {cov:.3f}  test-cell [{lo:.3f}, {hi:.3f}]{j}", flush=True)

    TABLES.mkdir(parents=True, exist_ok=True)
    suffix = "" if (args.steps, args.n_samples) == (50, 10) else f"_s{args.steps}_n{args.n_samples}"
    out = TABLES / f"coverage_cellboot_v2{suffix}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
