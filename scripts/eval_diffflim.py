"""Evaluate a trained DiffFLIM checkpoint on the fixed simulated test set.

For each photon level: the first --n-images images, --n-samples DDIM samples per image ->
mean (prediction) and std (uncertainty) maps. MLE and phasor are run on the same images.

Writes
    results/tables/diffflim_{tag}.csv        errors per level / method / pixel subset
    results/tables/coverage_{tag}.csv        calibration: is the truth inside mean +/- z*std?
    results/figures/error_vs_photons_{tag}.png
    results/figures/diffflim_{tag}_examples.png

Pixel subsets (always foreground only):
    all_fg     every foreground pixel (DiffFLIM also predicts pixels with 0 photons)
    mle_valid  foreground pixels where MLE gives a finite tau (>= 1 photon) -> fair comparison
    bright     MLE-valid foreground pixels with true intensity >= 20 % of the crop's intensity
               range above its dimmest foreground pixel (metrics.bright_mask)
Every row also gives the median local jitter of the true lifetime on its pixel set.

Usage:
    python scripts/eval_diffflim.py                       # checkpoints/diffflim_v1.pt
    python scripts/eval_diffflim.py --n-images 1 --tag test --ckpt checkpoints/diffflim_smoke.pt
    python scripts/eval_diffflim.py --test-set data/raw/own/sim_v3_test --tag v1_on_v3

Baselines (MLE, phasor, binned MLE 3x3 and 5x5) use the forward model of the test set
(src/baselines/common.py); on simulator-v3 data that is a mono-exponential with the image's IRF.
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
from src.baselines.common import METHODS, fit_all  # noqa: E402
from src.data.simulate import load_dataset, test_set_dir  # noqa: E402
from src.evaluation.metrics import BRIGHT_FRAC, bright_mask, jitter_summary, summarize  # noqa: E402
from src.models.diffflim import load_model  # noqa: E402
from src.paths import CHECKPOINTS, FIGURES, TABLES  # noqa: E402

Z90, Z68 = 1.645, 1.0
EXAMPLE_LEVELS = (10, 100)


def run_level(model, path, n_images, n_samples, steps):
    d = load_dataset(path)
    n = min(n_images, len(d["counts"]))
    keys = ["tau_mean", "tau_std", "int_mean", "int_std"] + [f"{q}_{m}" for m in METHODS for q in ("tau", "int")]
    out = {k: [] for k in keys}
    for i in range(n):
        counts, bg = d["counts"][i], float(d["bg_per_bin"][i])
        s = model.sample(counts, n_samples=n_samples, steps=steps, seed=1000 + i)
        out["tau_mean"].append(s[:, 0].mean(0)); out["tau_std"].append(s[:, 0].std(0, ddof=1))
        out["int_mean"].append(s[:, 1].mean(0)); out["int_std"].append(s[:, 1].std(0, ddof=1))
        base = fit_all(counts, bg, d["metadata"], d["irf"][i] if "irf" in d else None)
        for m, (t, I) in base.items():
            out[f"tau_{m}"].append(t); out[f"int_{m}"].append(I)
    out = {k: np.stack(v) for k, v in out.items()}
    out.update(tau_gt=d["tau_gt"][:n], int_gt=d["intensity_gt"][:n], mask=d["mask"][:n],
               counts_sum=d["counts"][:n].sum(1), photons=int(round(float(d["photons"][0]))), n=n)
    return out


def error_rows(r):
    fg = r["mask"]
    mle_valid = fg & np.isfinite(r["tau_MLE"])
    bright = mle_valid & bright_mask(r["int_gt"], fg)
    sets = {"all_fg": fg, "mle_valid": mle_valid, "bright": bright}
    jit = {k: jitter_summary(r["tau_gt"], m) for k, m in sets.items()}
    rows = []
    for method, tau, inten, subsets in [
        ("DiffFLIM", r["tau_mean"], r["int_mean"], ["all_fg", "mle_valid", "bright"]),
    ] + [(m, r[f"tau_{m}"], r[f"int_{m}"], ["mle_valid", "bright"]) for m in METHODS]:
        for subset in subsets:
            m = sets[subset]
            st, si = summarize(tau, r["tau_gt"], m), summarize(inten, r["int_gt"], m)
            rows.append({"method": method, "photons": r["photons"], "subset": subset, "n_images": r["n"],
                         "tau_mare_pct": st["mare_pct"], "tau_bias_pct": st["bias_pct"],
                         "int_mare_pct": si["mare_pct"], "int_bias_pct": si["bias_pct"],
                         "tau_invalid_frac": st["invalid_frac"], "n_pixels": st["n_pixels"],
                         "gt_jitter_median_ns": jit[subset]["jitter_median_ns"],
                         "gt_jitter_median_pct": jit[subset]["jitter_median_pct"]})
    return rows


def coverage_rows(r):
    sets = {"all_fg": r["mask"], "bright": r["mask"] & bright_mask(r["int_gt"], r["mask"])}
    return [coverage_row(r, m, name) for name, m in sets.items()]


def coverage_row(r, m, subset):
    row = {"photons": r["photons"], "subset": subset, "n_images": r["n"], "n_pixels": int(m.sum())}
    for name, gt, mean, std in [("tau", r["tau_gt"], r["tau_mean"], r["tau_std"]),
                                ("int", r["int_gt"], r["int_mean"], r["int_std"])]:
        err = np.abs(gt - mean)[m]
        row[f"cov90_{name}"] = float(np.mean(err <= Z90 * std[m]))
        row[f"cov68_{name}"] = float(np.mean(err <= Z68 * std[m]))
        row[f"median_rel_std_{name}"] = float(np.median(std[m] / np.maximum(mean[m], 1e-6)))
    return row


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


def plot_errors(rows, out):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    p_all = np.array(sorted({x["photons"] for x in rows}), dtype=float)
    panels = [("all foreground pixels", [("DiffFLIM", "all_fg", "D-", "DiffFLIM (all fg pixels)"),
                                         ("DiffFLIM", "mle_valid", "D:", "DiffFLIM (MLE-valid pixels)")], "mle_valid"),
              (f"bright fg pixels (>= {BRIGHT_FRAC:.0%} of intensity range)",
               [("DiffFLIM", "bright", "D-", "DiffFLIM")], "bright")]
    for i, (title, dlines, base_subset) in enumerate(panels):
        lines = [("MLE", base_subset, "o-", "MLE"), ("Phasor", base_subset, "s--", "Phasor"),
                 ("BinMLE3", base_subset, "^-", "Binned MLE 3x3"),
                 ("BinMLE5", base_subset, "v-", "Binned MLE 5x5")] + dlines
        for method, subset, style, label in lines:
            r = sorted((x for x in rows if x["method"] == method and x["subset"] == subset), key=lambda x: x["photons"])
            p = [x["photons"] for x in r]
            ax[i, 0].plot(p, [x["tau_mare_pct"] for x in r], style, label=label)
            ax[i, 1].plot(p, [x["int_mare_pct"] for x in r], style, label=label)
        jit = np.median([x["gt_jitter_median_pct"] for x in rows if x["subset"] == base_subset])
        if jit > 0:
            ax[i, 0].axhline(jit, color="gray", ls=":", label=f"GT local jitter ({jit:.1f}%)")
        ax[i, 1].plot(p_all, 100 * 0.674 / np.sqrt(p_all), ":", color="gray", label="Poisson limit (per pixel)")
        for a, q in zip(ax[i], ["Lifetime", "Intensity"]):
            a.set_xscale("log"); a.set_yscale("log")
            a.set_xlabel("Photons per foreground pixel"); a.set_ylabel("Median |relative error| (%)")
            a.set_title(f"{q} error, {title}", fontsize=10); a.grid(True, which="both", alpha=0.3)
            a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def plot_examples(results, out):
    ex = [r for r in results if r["photons"] in EXAMPLE_LEVELS]
    fig, ax = plt.subplots(len(ex), 5, figsize=(19, 3.8 * len(ex)), squeeze=False)
    for row, r in zip(ax, ex):
        m = r["mask"][0]
        err = lambda t: summarize(t, r["tau_gt"][0], m)["mare_pct"]  # noqa: E731
        lim = tuple(np.percentile(r["tau_gt"][0][m], [1, 99]))           # colour range from the truth
        panels = [
            (r["counts_sum"][0], "Input: summed counts", "magma", None),
            (np.where(m, r["tau_gt"][0], np.nan), "True tau (ns)", "viridis", lim),
            (r["tau_mean"][0], f"DiffFLIM mean tau (err {err(r['tau_mean'][0]):.1f}%)", "viridis", lim),
            (r["tau_std"][0], "DiffFLIM std tau (ns)", "inferno", None),
            (r["tau_MLE"][0], f"MLE tau (err {err(r['tau_MLE'][0]):.1f}%)", "viridis", lim),
        ]
        for a, (img, title, cmap, lim) in zip(row, panels):
            im = a.imshow(img, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
            fig.colorbar(im, ax=a, fraction=0.046)
            a.contour(m, levels=[0.5], colors="white", linewidths=0.5)
            a.set_title(title, fontsize=10); a.axis("off")
        row[0].set_title(f"{r['photons']} photons/fg pixel\nInput: summed counts", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CHECKPOINTS / "diffflim_v1.pt"))
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--n-images", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--test-set", default=None, help="folder with test_p*.npz (default: v2 test set)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.ckpt, device)
    print(f"Loaded {args.ckpt}: step {ckpt['step']}, trained {ckpt['elapsed_s'] / 60:.1f} min (EMA weights)")

    folder = Path(args.test_set) if args.test_set else test_set_dir()
    files = sorted(folder.glob("test_p*.npz"), key=lambda p: int(p.stem.split("_p")[1]))
    print(f"Test set: {folder}")
    results, err_rows, cov_rows = [], [], []
    for path in files:
        t0 = time.perf_counter()
        r = run_level(model, path, args.n_images, args.n_samples, args.steps)
        results.append(r)
        err_rows += error_rows(r)
        cov_rows += coverage_rows(r)
        print(f"{path.name}: {time.perf_counter() - t0:.0f} s")

    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = [TABLES / f"diffflim_{args.tag}.csv", TABLES / f"coverage_{args.tag}.csv",
               FIGURES / f"error_vs_photons_{args.tag}.png", FIGURES / f"diffflim_{args.tag}_examples.png"]
    write_csv(outputs[0], err_rows)
    write_csv(outputs[1], cov_rows)
    plot_errors(err_rows, outputs[2])
    plot_examples(results, outputs[3])

    print("\nmethod    subset     photons  tau_err  tau_bias  int_err  int_bias")
    for x in err_rows:
        print(f"{x['method']:9s} {x['subset']:10s} {x['photons']:6d}  {x['tau_mare_pct']:6.2f}%  "
              f"{x['tau_bias_pct']:+6.2f}%  {x['int_mare_pct']:6.2f}%  {x['int_bias_pct']:+6.2f}%")
    print("\nphotons subset  cov90_tau  cov90_int  cov68_tau  cov68_int  rel_std_tau  rel_std_int")
    for c in cov_rows:
        print(f"{c['photons']:6d} {c['subset']:7s} {c['cov90_tau']:8.3f}  {c['cov90_int']:8.3f}  {c['cov68_tau']:8.3f}  "
              f"{c['cov68_int']:8.3f}  {c['median_rel_std_tau']:10.3f}  {c['median_rel_std_int']:10.3f}")
    print("\n" + "\n".join(f"Saved {p}" for p in outputs))


if __name__ == "__main__":
    main()
