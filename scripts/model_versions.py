"""Versions table: one row per DiffFLIM model, built only from existing result files and caches.

A model with tag T is included when these files exist (all written by the post-training chain,
scripts/post_training_chain.sh):
  data/processed/preds_T_s10_n10_test.npz            test-set predictions, 10 DDIM steps x 10 samples
                                                     (calibrate_diffflim.py --preds-tag T)
  results/tables/coverage_calibrated_T_extra_s10.csv calibration on all 74 calibration cells
  results/tables/real_celegans_groups_T.csv          real data, Test A   (eval_real_celegans.py --tag T)
  results/tables/real_celegans_splithalf_T.csv       real data, Test B
The checkpoint is checkpoints/diffflim_T.pt.

Columns: training (minutes, steps, number of training IDs), lifetime and intensity error on the v3
test set (median absolute relative error, true foreground, all images present in the cache, at 10,
100 and 1000 photons and pooled), lifetime bias, pooled 90 % coverage with per-level factors
(lifetime, intensity), and the real-data results (Abeta and mScarlet medians, Mann-Whitney p,
split-half difference).

Writes results/tables/model_versions.csv

Usage:
    python scripts/model_versions.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import simulate_v3 as v3  # noqa: E402
from src.paths import CHECKPOINTS, PROCESSED, TABLES  # noqa: E402

LEVELS = (10, 100, 1000)


def mare(est, gt, mask):
    return 100 * float(np.median(np.abs(est[mask] - gt[mask]) / gt[mask]))


def bias(est, gt, mask):
    return 100 * float(np.median((est[mask] - gt[mask]) / gt[mask]))


def row_for(tag):
    files = {"preds": PROCESSED / f"preds_{tag}_s10_n10_test.npz",
             "cov": TABLES / f"coverage_calibrated_{tag}_extra_s10.csv",
             "groups": TABLES / f"real_celegans_groups_{tag}.csv",
             "split": TABLES / f"real_celegans_splithalf_{tag}.csv",
             "ckpt": CHECKPOINTS / f"diffflim_{tag}.pt"}
    if not all(f.exists() for f in files.values()):
        return None
    ck = torch.load(files["ckpt"], map_location="cpu", weights_only=False)
    ids = ck.get("args", {}).get("train_ids")
    n_ids = len(ids) if ids else (len(v3.diffflim_v2_train_ids()) if tag == "v2" else "")
    row = {"model": tag, "checkpoint": files["ckpt"].relative_to(ROOT).as_posix(),
           "train_minutes": round(ck["elapsed_s"] / 60, 1), "train_steps": ck["step"], "n_train_ids": n_ids}
    with np.load(files["preds"]) as z:
        p = {k: z[k] for k in z.files}
    row["n_test_images"] = len(p["photons"])
    for lev in LEVELS:
        i = np.round(p["photons"]) == lev
        row[f"tau_mare_pct_{lev}"] = mare(p["tau_mean"][i], p["tau_gt"][i], p["mask"][i])
    row["tau_mare_pct_all"] = mare(p["tau_mean"], p["tau_gt"], p["mask"])
    row["tau_bias_pct_all"] = bias(p["tau_mean"], p["tau_gt"], p["mask"])
    for lev in LEVELS:
        i = np.round(p["photons"]) == lev
        row[f"int_mare_pct_{lev}"] = mare(p["int_mean"][i], p["int_gt"][i], p["mask"][i])
    row["int_mare_pct_all"] = mare(p["int_mean"], p["int_gt"], p["mask"])
    with open(files["cov"]) as f:
        cov = [r for r in csv.DictReader(f) if r["photons"] == "all" and abs(float(r["nominal_coverage"]) - 0.9) < 1e-6]
    for r in cov:
        row[f"cov90_{r['quantity']}_per_level"] = float(r["cov_per_level"])
    with open(files["groups"]) as f:
        test = next(r for r in csv.DictReader(f) if r["row_type"] == "test" and r["method"] == f"DiffFLIM {tag}")
    row.update({"real_tau_abeta_ns": float(test["median_abeta_ns"]), "real_tau_mscarlet_ns": float(test["median_mscarlet_ns"]),
                "real_mannwhitney_p": float(test["p_value"])})
    with open(files["split"]) as f:
        sh = next(r for r in csv.DictReader(f) if r["row_type"] == "median over recordings"
                  and r["method"] == f"DiffFLIM {tag}" and r["group"] == "all")
    row["real_splithalf_diff_pct"] = float(sh["mad_AB_rel_pct"])
    return row


def main():
    tags = sorted({p.stem[len("real_celegans_groups_"):] for p in TABLES.glob("real_celegans_groups_*.csv")})
    rows = [r for r in (row_for(t) for t in tags) if r]
    out = TABLES / "model_versions.csv"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        print(f"{r['model']:20s} {r['train_minutes']} min, {r['n_train_ids']} IDs | tau {r['tau_mare_pct_10']:.2f}/"
              f"{r['tau_mare_pct_100']:.2f}/{r['tau_mare_pct_1000']:.2f} % | cov90 tau {r.get('cov90_tau_per_level', 0):.3f} "
              f"int {r.get('cov90_int_per_level', 0):.3f} | real Abeta {r['real_tau_abeta_ns']:.2f} mScarlet "
              f"{r['real_tau_mscarlet_ns']:.2f} ns p {r['real_mannwhitney_p']:.4f} split {r['real_splithalf_diff_pct']:.1f} %")
    print(f"Saved {out} ({len(rows)} models)")


if __name__ == "__main__":
    main()
