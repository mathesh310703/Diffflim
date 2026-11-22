"""Paired comparison of DiffFLIM model versions on the v3 test set, with a cell-level bootstrap.

For each pair of models (A, B) and each quantity (lifetime, intensity), the per-image median
absolute relative error on the true foreground is computed from the cached test predictions
(data/processed/preds_<tag>_s10_n10_test.npz, 10 DDIM steps x 10 samples). The difference A - B
is averaged over the images of a photon level (and over all levels). Both models are evaluated on
the same images, so the difference is paired. The 95 % confidence interval resamples the 30 HPA
test cells with replacement (all images of a drawn cell enter together), because images of the
same cell are not independent (the test set is paired across photon levels).

Writes results/tables/model_versions_paired.csv

Usage:
    python scripts/compare_versions.py [--pairs v3cells:v2,v3cells_3h:v2_3h,v2:v2_3h,v3cells:v3cells_3h]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.simulate import load_dataset  # noqa: E402
from src.paths import PROCESSED, RAW_OWN, TABLES  # noqa: E402

LEVELS = (10, 20, 50, 100, 500, 1000)
N_BOOT = 5000
SEED = 0


def test_hpa_ids():
    files = sorted((RAW_OWN / "sim_v3_test").glob("test_p*.npz"), key=lambda p: int(p.stem.split("_p")[1]))
    return np.concatenate([np.asarray(load_dataset(f)["hpa_id"]).astype(str) for f in files])


def per_image_error(tag, q):
    with np.load(PROCESSED / f"preds_{tag}_s10_n10_test.npz") as z:
        est, gt, mask, photons = z[f"{q}_mean"], z[f"{q}_gt"], z["mask"], z["photons"]
    err = np.array([np.median(np.abs(est[i][mask[i]] - gt[i][mask[i]]) / gt[i][mask[i]]) for i in range(len(est))])
    return 100 * err, np.round(photons).astype(int), gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="v3cells:v2,v3cells_3h:v2_3h,v2:v2_3h,v3cells:v3cells_3h")
    args = ap.parse_args()
    ids = test_hpa_ids()
    cells = np.unique(ids)
    rng = np.random.default_rng(SEED)
    draws = [rng.choice(cells, len(cells)) for _ in range(N_BOOT)]
    rows = []
    for pair in args.pairs.split(","):
        a, b = pair.split(":")
        for q in ("tau", "int"):
            ea, lev, gta = per_image_error(a, q)
            eb, levb, gtb = per_image_error(b, q)
            assert len(ea) == len(ids) and np.array_equal(lev, levb) and np.allclose(gta, gtb), "images differ"
            for level in LEVELS + ("all",):
                sel = np.ones(len(ea), bool) if level == "all" else lev == level
                d = (ea - eb)[sel]
                cid = ids[sel]
                by_cell = {c: d[cid == c] for c in cells}
                boot = [np.mean(np.concatenate([by_cell[c] for c in draw])) for draw in draws]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                rows.append({"model_a": a, "model_b": b, "quantity": q, "photons": level, "n_images": int(sel.sum()),
                             "mean_err_a_pct": ea[sel].mean(), "mean_err_b_pct": eb[sel].mean(),
                             "diff_a_minus_b_pct": d.mean(), "ci95_lo": lo, "ci95_hi": hi})
                print(f"{a:>11s} - {b:<11s} {q:3s} {str(level):>5s}: {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] percentage points")
    out = TABLES / "model_versions_paired.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
