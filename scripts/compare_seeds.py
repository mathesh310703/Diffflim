"""Run-to-run variation of DiffFLIM v2: seed 0 (checkpoints/diffflim_v2.pt) against seed 1
(checkpoints/diffflim_v2_seed1.pt, identical settings and training IDs, only --seed 1), and the earlier
version differences put into that context.

Inputs (all written by scripts/post_training_chain.sh; the second test set is not used):
  data/processed/preds_<tag>_s10_n10_{test,calib,calib_sim_v3_calib_extra}.npz
  results/tables/real_celegans_groups_<tag>.csv, real_celegans_splithalf_<tag>.csv
  results/tables/coverage_cellboot_<tag>_s10_n10.csv (computed here with coverage_cell_bootstrap.py
  --tag <tag> when missing; cell-level 95 % intervals of the calibrated coverage)

Comparisons (model_a - model_b):
  seed pairs:    v2_seed1 - v2 (final, 186,116 steps), v2_seed1_3h - v2_3h (step 92,168)
  context pairs: v3cells - v2, v3cells_3h - v2_3h (more training cells), v2long - v2, v2long_9h - v2
                 (longer training; a continuation of the seed-0 run, so it shares its seed)
Each context difference is compared with the seed difference of the same metric and training length
(final or 3 h; v2long uses the final seed pair).

Metrics per comparison:
  tau_mare / int_mare at each photon level and pooled: mean over images of the per-image median
    absolute relative error (true foreground), paired, with a 95 % cell-bootstrap interval of the
    difference (30 test cells, 5000 draws; as scripts/compare_versions.py)
  tau_mare_pooled, int_mare_pooled, tau_bias_pooled: median over all foreground pixels of the level
  cov90_tau, cov90_int, cov68_tau, cov68_int: pooled calibrated coverage (per-level factors), with the
    test-cell intervals of each model
  real_diff_ns, real_p: Abeta - mScarlet difference of the per-recording medians and Mann-Whitney p
  real_splithalf_pct (all, abeta, mscarlet): median split-half difference over recordings

Writes results/tables/seed_variation_v2.csv and results/figures/seed_variation_v2.png

Usage:
    python scripts/compare_seeds.py [--no-cellboot]
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_versions import per_image_error, test_hpa_ids  # noqa: E402
from src.paths import FIGURES, PROCESSED, TABLES  # noqa: E402

LEVELS = (10, 20, 50, 100, 500, 1000)
N_BOOT = 5000
SEED_PAIRS = {"final": ("v2_seed1", "v2"), "3h": ("v2_seed1_3h", "v2_3h")}
CONTEXT = [("v3cells", "v2", "final"), ("v3cells_3h", "v2_3h", "3h"),
           ("v2long", "v2", "final"), ("v2long_9h", "v2", "final")]
TAGS = ["v2", "v2_seed1", "v2_3h", "v2_seed1_3h", "v3cells", "v3cells_3h", "v2long", "v2long_9h"]


def have(tag):
    return (PROCESSED / f"preds_{tag}_s10_n10_test.npz").exists()


def pooled(tag, q, kind):
    with np.load(PROCESSED / f"preds_{tag}_s10_n10_test.npz") as z:
        est, gt, mask, ph = z[f"{q}_mean"], z[f"{q}_gt"], z["mask"], np.round(z["photons"]).astype(int)
    out = {}
    for lev in LEVELS + ("all",):
        sel = np.ones(len(ph), bool) if lev == "all" else ph == lev
        r = (est[sel][mask[sel]] - gt[sel][mask[sel]]) / gt[sel][mask[sel]]
        out[lev] = 100 * float(np.median(np.abs(r) if kind == "mare" else r))
    return out


def paired(a, b, q, ids, cells, draws):
    ea, lev, gta = per_image_error(a, q)
    eb, levb, gtb = per_image_error(b, q)
    assert len(ea) == len(ids) and np.array_equal(lev, levb) and np.allclose(gta, gtb), "images differ"
    out = {}
    for level in LEVELS + ("all",):
        sel = np.ones(len(ea), bool) if level == "all" else lev == level
        d, cid = (ea - eb)[sel], ids[sel]
        by_cell = {c: d[cid == c] for c in cells}
        boot = [np.mean(np.concatenate([by_cell[c] for c in draw])) for draw in draws]
        out[level] = (ea[sel].mean(), eb[sel].mean(), d.mean(), *np.percentile(boot, [2.5, 97.5]))
    return out


def cellboot(tag, run):
    f = TABLES / f"coverage_cellboot_{tag}_s10_n10.csv"
    if not f.exists() and run:
        print(f"cell bootstrap for {tag} ...", flush=True)
        subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts/coverage_cell_bootstrap.py"), "--tag", tag],
                       check=True, stdout=subprocess.DEVNULL)
    if not f.exists():
        return {}
    out = {}
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if r["photons"] == "all" and r["variant"] == "per_level":
                key = f"cov{round(100 * float(r['nominal_coverage']))}_{r['quantity']}"
                out[key] = (100 * float(r["coverage"]), 100 * float(r["testcell_ci_lo"]), 100 * float(r["testcell_ci_hi"]))
    return out


def real(tag):
    out = {}
    g = TABLES / f"real_celegans_groups_{tag}.csv"
    s = TABLES / f"real_celegans_splithalf_{tag}.csv"
    if g.exists():
        with open(g) as fh:
            for r in csv.DictReader(fh):
                if r["row_type"] == "test" and r["method"].startswith("DiffFLIM"):
                    out["real_abeta_ns"] = float(r["median_abeta_ns"])
                    out["real_mscarlet_ns"] = float(r["median_mscarlet_ns"])
                    out["real_diff_ns"] = float(r["difference_ns"])
                    out["real_p"] = float(r["p_value"])
    if s.exists():
        with open(s) as fh:
            for r in csv.DictReader(fh):
                if r["row_type"] == "median over recordings" and r["method"].startswith("DiffFLIM"):
                    out[f"real_splithalf_{r['group']}_pct"] = float(r["mad_AB_rel_pct"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cellboot", action="store_true", help="do not compute missing cell-bootstrap tables")
    args = ap.parse_args()
    tags = [t for t in TAGS if have(t)]
    missing = [t for t in TAGS if t not in tags]
    if missing:
        print("missing test predictions (skipped):", ", ".join(missing))
    ids = test_hpa_ids()
    cells = np.unique(ids)
    rng = np.random.default_rng(0)
    draws = [rng.choice(cells, len(cells)) for _ in range(N_BOOT)]

    # per-model values
    val = {}
    for t in tags:
        v = {}
        for q in ("tau", "int"):
            for lev, x in pooled(t, q, "mare").items():
                v[(f"{q}_mare_pooled", lev)] = x
        for lev, x in pooled(t, "tau", "bias").items():
            v[("tau_bias_pooled", lev)] = x
        for k, x in cellboot(t, not args.no_cellboot).items():
            v[(k, "all")] = x[0]
            v[(k + "_ci_lo", "all")] = x[1]
            v[(k + "_ci_hi", "all")] = x[2]
        for k, x in real(t).items():
            v[(k, "")] = x
        val[t] = v

    rows = []
    seed_diff = {}
    pairs = [(a, b, ln, "seed") for ln, (a, b) in SEED_PAIRS.items()] + [(a, b, ln, "context") for a, b, ln in CONTEXT]
    for a, b, length, kind in pairs:
        if a not in val or b not in val:
            continue
        for q in ("tau", "int"):
            for lev, (ma, mb, d, lo, hi) in paired(a, b, q, ids, cells, draws).items():
                rows.append({"comparison": kind, "length": length, "model_a": a, "model_b": b,
                             "metric": f"{q}_mare_mean_image", "photons": lev, "value_a": ma, "value_b": mb,
                             "diff_a_minus_b": d, "ci95_lo": lo, "ci95_hi": hi})
        for key in sorted(set(val[a]) & set(val[b]), key=str):
            if key[0].endswith(("_ci_lo", "_ci_hi")):
                continue
            ra = {"value_a": val[a][key], "value_b": val[b][key], "diff_a_minus_b": val[a][key] - val[b][key],
                  "ci95_lo": "", "ci95_hi": ""}
            if (key[0] + "_ci_lo", "all") in val[a]:
                ra["a_ci95"] = f"{val[a][(key[0] + '_ci_lo', 'all')]:.1f}-{val[a][(key[0] + '_ci_hi', 'all')]:.1f}"
                ra["b_ci95"] = f"{val[b][(key[0] + '_ci_lo', 'all')]:.1f}-{val[b][(key[0] + '_ci_hi', 'all')]:.1f}"
            rows.append({"comparison": kind, "length": length, "model_a": a, "model_b": b,
                         "metric": key[0], "photons": key[1], **ra})
    for r in rows:
        if r["comparison"] == "seed":
            seed_diff[(r["length"], r["metric"], r["photons"])] = r["diff_a_minus_b"]
    for r in rows:
        sd = seed_diff.get((r["length"], r["metric"], r["photons"]))
        r["seed_diff_same_metric"] = sd if sd is not None else ""
        if r["comparison"] == "context" and sd is not None:
            r["abs_ratio_to_seed_diff"] = abs(r["diff_a_minus_b"]) / max(abs(sd), 1e-9)
            r["larger_than_seed_diff"] = int(abs(r["diff_a_minus_b"]) > abs(sd))
    keys = ["comparison", "length", "model_a", "model_b", "metric", "photons", "value_a", "value_b",
            "diff_a_minus_b", "ci95_lo", "ci95_hi", "a_ci95", "b_ci95", "seed_diff_same_metric",
            "abs_ratio_to_seed_diff", "larger_than_seed_diff"]
    out = TABLES / "seed_variation_v2.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4g}" if isinstance(r.get(k), float) else r.get(k, "")) for k in keys})
    print(f"Saved {out}")
    for r in rows:
        if r["metric"] in ("tau_mare_mean_image", "int_mare_mean_image") and r["photons"] in (10, 100, 1000, "all") \
                or r["metric"].startswith(("cov90", "real_diff", "real_p", "real_splithalf_all", "tau_bias")) and r["photons"] in ("all", ""):
            ci = f" [{r['ci95_lo']:+.2f}, {r['ci95_hi']:+.2f}]" if r["ci95_lo"] != "" else ""
            sd = f"  seed diff {r['seed_diff_same_metric']:+.3f}" if r["comparison"] == "context" and r["seed_diff_same_metric"] != "" else ""
            print(f"{r['model_a']:>12s} - {r['model_b']:<6s} {r['metric']:22s} {str(r['photons']):>5s}: "
                  f"{r['value_a']:.4g} vs {r['value_b']:.4g}, diff {r['diff_a_minus_b']:+.3f}{ci}{sd}")
    plot(rows)


def plot(rows):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {"v2_seed1": "C3", "v2_seed1_3h": "C3", "v3cells": "C0", "v3cells_3h": "C0", "v2long": "C2", "v2long_9h": "C2"}
    lab = {"v2_seed1": "seed 1 - seed 0 (final)", "v2_seed1_3h": "seed 1 - seed 0 (3 h)",
           "v3cells": "v3cells - v2", "v3cells_3h": "v3cells 3 h - v2 3 h", "v2long": "v2long - v2",
           "v2long_9h": "v2long 9 h - v2"}
    for k, q in enumerate(("tau", "int")):
        a_ = ax[k]
        sel = [r for r in rows if r["metric"] == f"{q}_mare_mean_image" and r["photons"] != "all"]
        names = list(dict.fromkeys(r["model_a"] for r in sel))
        for j, n in enumerate(names):
            rr = [r for r in sel if r["model_a"] == n]
            x = np.array([r["photons"] for r in rr], float) * (1 + 0.05 * (j - len(names) / 2))
            y = np.array([r["diff_a_minus_b"] for r in rr])
            e = np.array([[r["diff_a_minus_b"] - r["ci95_lo"], r["ci95_hi"] - r["diff_a_minus_b"]] for r in rr]).T
            a_.errorbar(x, y, yerr=e, fmt="o-" if not n.endswith(("_3h", "_9h")) else "s--", ms=4, capsize=2,
                        color=colors.get(n, "k"), label=lab.get(n, n), alpha=0.9 if "seed" in n else 0.7)
        a_.axhline(0, color="k", lw=0.8)
        a_.set_xscale("log"); a_.grid(alpha=0.3, which="both")
        a_.set_xlabel("Photons per foreground pixel"); a_.set_ylabel("Paired difference (percentage points)")
        a_.set_title(f"{'Lifetime' if q == 'tau' else 'Intensity'} error, A - B (95 % cell bootstrap)", fontsize=10)
        if k == 0:
            a_.legend(fontsize=7)
    a_ = ax[2]
    real_rows = [r for r in rows if r["metric"] in ("real_diff_ns", "real_splithalf_all_pct")]
    names = list(dict.fromkeys(r["model_a"] for r in real_rows))
    for j, n in enumerate(names):
        for m, (metric, scale) in enumerate((("real_diff_ns", 10.0), ("real_splithalf_all_pct", 1.0))):
            r = [x for x in real_rows if x["model_a"] == n and x["metric"] == metric]
            if r:
                a_.bar(m * (len(names) + 1) + j, r[0]["diff_a_minus_b"] * scale, color=colors.get(n, "k"),
                       hatch="//" if n.endswith(("_3h", "_9h")) else "", label=lab.get(n, n) if m == 0 else None)
    a_.set_xticks([(len(names) - 1) / 2, len(names) + 1 + (len(names) - 1) / 2])
    a_.set_xticklabels(["group difference\n(A - B, 0.1 ns)", "split-half difference\n(A - B, points)"])
    a_.axhline(0, color="k", lw=0.8); a_.grid(alpha=0.3, axis="y")
    a_.set_title("Real C. elegans protocol", fontsize=10)
    a_.legend(fontsize=7)
    fig.tight_layout()
    f = FIGURES / "seed_variation_v2.png"
    fig.savefig(f, dpi=120)
    print(f"Saved {f}")


if __name__ == "__main__":
    main()
