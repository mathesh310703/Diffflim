"""Figure comparing DiffFLIM model versions: results/figures/versions.png.

Panels (all from files written by scripts/post_training_chain.sh; models whose files are missing
are skipped):
  (a) lifetime error and (b) intensity error per photon level on the v3 test set (median absolute
      relative error, true foreground, 10 DDIM steps x 10 samples; data/processed/preds_<tag>_s10_n10_test.npz),
  (c) lifetime bias per photon level,
  (d) coverage of the calibrated 90 % interval per photon level, per-level factors, lifetime (solid)
      and intensity (dashed) (results/tables/coverage_calibrated_<tag>_extra_s10.csv),
  (e) real data, Test A: median lifetime per C. elegans recording, Abeta-mScarlet (red) and
      mScarlet (blue), with the Mann-Whitney p value (results/tables/real_celegans_groups_<tag>.csv),
  (f) real data, Test B: split-half difference per recording (results/tables/real_celegans_splithalf_<tag>.csv).

Usage:
    python scripts/plot_versions.py [--tags v2_3h,v2,v2long_9h,v2long,v3cells_3h,v3cells]
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import FIGURES, PROCESSED, TABLES  # noqa: E402

LEVELS = (10, 20, 50, 100, 500, 1000)
LABELS = {"v2_3h": "v2, 3 h (246 cells)", "v2": "v2, 6 h (246 cells)",
          "v2long_9h": "v2 long, step 279k (246 cells)", "v2long": "v2 long, final (246 cells)",
          "v3cells_3h": "v3cells, step 92k", "v3cells": "v3cells, step 186k"}
STYLES = {"v2_3h": ("C0", "--", "o"), "v2": ("C0", "-", "o"), "v2long_9h": ("C1", "--", "^"), "v2long": ("C1", "-", "^"),
          "v3cells_3h": ("C3", "--", "s"), "v3cells": ("C3", "-", "s")}
SHORT = {"v2_3h": "v2\nstep 92k", "v2": "v2\nstep 186k", "v2long_9h": "v2 long\nstep 279k",
         "v2long": "v2 long\nstep 350k", "v3cells_3h": "v3cells\nstep 92k", "v3cells": "v3cells\nstep 186k"}


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def per_level(p, key):
    out = []
    for lev in LEVELS:
        i = np.round(p["photons"]) == lev
        est, gt, m = p[f"{key}_mean"][i], p[f"{key}_gt"][i], p["mask"][i]
        rel = (est[m] - gt[m]) / gt[m]
        out.append((100 * float(np.median(np.abs(rel))), 100 * float(np.median(rel))))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v2_3h,v2,v2long_9h,v2long,v3cells_3h,v3cells")
    ap.add_argument("--out", default=str(FIGURES / "versions.png"))
    args = ap.parse_args()
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))
    ax = ax.ravel()
    shown = []
    for tag in args.tags.split(","):
        files = [PROCESSED / f"preds_{tag}_s10_n10_test.npz", TABLES / f"coverage_calibrated_{tag}_extra_s10.csv",
                 TABLES / f"real_celegans_groups_{tag}.csv", TABLES / f"real_celegans_splithalf_{tag}.csv"]
        if not all(f.exists() for f in files):
            print(f"skipping {tag}: missing " + ", ".join(f.name for f in files if not f.exists()))
            continue
        shown.append(tag)
        c, ls, mk = STYLES.get(tag, ("C2", "-", "^"))
        lab = LABELS.get(tag, tag)
        with np.load(files[0]) as z:
            p = {k: z[k] for k in z.files}
        tau, inten = per_level(p, "tau"), per_level(p, "int")
        ax[0].plot(LEVELS, tau[:, 0], ls, color=c, marker=mk, label=lab)
        ax[1].plot(LEVELS, inten[:, 0], ls, color=c, marker=mk, label=lab)
        ax[2].plot(LEVELS, tau[:, 1], ls, color=c, marker=mk, label=lab)
        cov = [r for r in read(files[1]) if r["photons"] != "all" and abs(float(r["nominal_coverage"]) - 0.9) < 1e-6]
        for q, qls in (("tau", "-"), ("int", ":")):
            pts = sorted((int(r["photons"]), 100 * float(r["cov_per_level"])) for r in cov if r["quantity"] == q)
            ax[3].plot([x for x, _ in pts], [y for _, y in pts], qls, color=c, marker=mk, ms=4,
                       alpha=1.0 if ls == "-" else 0.55, label=f"{lab}, {'lifetime' if q == 'tau' else 'intensity'}")
        k = len(shown) - 1
        g = read(files[2])
        dname = f"DiffFLIM {tag}"
        test = next(r for r in g if r["row_type"] == "test" and r["method"] == dname)
        for j, (grp, col) in enumerate((("abeta", "C3"), ("mscarlet", "C0"))):
            v = [float(r["median_tau_ns"]) for r in g if r["row_type"] == "image" and r["method"] == dname and r["group"] == grp]
            ax[4].scatter(np.full(len(v), k + (j - 0.5) * 0.25), v, color=col, s=18, edgecolor="k", linewidth=0.4,
                          label=("Abeta-mScarlet" if grp == "abeta" else "mScarlet") if k == 0 else None)
        ax[4].text(k, ax[4].get_ylim()[1] if False else 3.45, f"p={float(test['p_value']):.3f}", ha="center", fontsize=8)
        sh = [r for r in read(files[3]) if r["row_type"] == "image" and r["method"] == dname]
        for j, (grp, col) in enumerate((("abeta", "C3"), ("mscarlet", "C0"))):
            v = [float(r["mad_AB_rel_pct"]) for r in sh if r["group"] == grp]
            ax[5].scatter(np.full(len(v), k + (j - 0.5) * 0.25), v, color=col, s=18, edgecolor="k", linewidth=0.4)
    for a, yl, t in zip(ax[:3], ["Median |relative error| (%)", "Median |relative error| (%)", "Median relative error (%)"],
                        ["(a) Lifetime error, v3 test set", "(b) Intensity error, v3 test set", "(c) Lifetime bias, v3 test set"]):
        a.set_xscale("log"); a.set_xlabel("Photons per foreground pixel"); a.set_ylabel(yl); a.set_title(t, fontsize=10)
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=7)
    ax[0].set_yscale("log"); ax[1].set_yscale("log")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[3].axhline(90, color="k", ls="--", lw=1)
    ax[3].set_xscale("log"); ax[3].set_xlabel("Photons per foreground pixel"); ax[3].set_ylabel("Observed coverage (%)")
    ax[3].set_title("(d) Calibrated 90 % interval, per-level factors\n(solid lifetime, dotted intensity)", fontsize=10)
    ax[3].grid(alpha=0.3, which="both"); ax[3].legend(fontsize=6)
    for a, yl, t in ((ax[4], "Median lifetime in the mask (ns)", "(e) Real data, Test A (C. elegans, 6 vs 6)"),
                     (ax[5], "Split-half difference (%)", "(f) Real data, Test B (split-half consistency)")):
        a.set_xticks(range(len(shown)))
        a.set_xlim(-0.6, len(shown) - 0.4)
        a.set_xticklabels([SHORT.get(t_, LABELS.get(t_, t_).replace(", ", "\n", 1)) for t_ in shown], fontsize=7)
        a.set_ylabel(yl); a.set_title(t, fontsize=10); a.grid(alpha=0.3, axis="y")
    ax[4].set_ylim(2.4, 3.55)
    ax[4].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"Saved {args.out} ({', '.join(shown)})")


if __name__ == "__main__":
    main()
