"""Evaluate the classical baselines (MLE, phasor, binned MLE 3x3 / 5x5) on a fixed test set.

The forward model is chosen from the test set's metadata (src/baselines/common.py): simulate.py
data use the Gaussian-IRF model, simulator-v3 data use each image's own IRF (mono-exponential).

Pixel sets: "fg" = all foreground pixels; "bright" = foreground pixels with true intensity at
least 20 % of the crop's intensity range above its dimmest foreground pixel
(src/evaluation/metrics.py: bright_mask). Each row also reports the median local jitter of the
ground-truth lifetime on that pixel set (|tau - 3x3 median|), to help interpret the errors.

Writes  results/tables/baselines{_tag}.csv
        results/figures/error_vs_photons{_baselines_tag}.png

Usage:
    python scripts/eval_baselines.py                                         # v2 test set (old outputs)
    python scripts/eval_baselines.py --test-set data/raw/own/sim_v3_test --tag v3
    python scripts/eval_baselines.py --n 5                                   # quick run
"""
import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.common import METHODS, fit_all  # noqa: E402
from src.data.simulate import load_dataset, test_set_dir  # noqa: E402
from src.evaluation.metrics import BRIGHT_FRAC, bright_mask, jitter_summary, summarize  # noqa: E402
from src.paths import FIGURES, TABLES  # noqa: E402

STYLES = {"MLE": "o-", "Phasor": "s--", "BinMLE3": "^-", "BinMLE5": "v-"}


def _fit_image(args):
    counts, bg, meta, irf, methods = args
    t0 = time.perf_counter()
    out = fit_all(counts, bg, meta, irf, methods)
    return out, time.perf_counter() - t0


def evaluate_level(path, n_max, methods, pool):
    data = load_dataset(path)
    meta = data["metadata"]
    n = min(n_max, len(data["counts"])) if n_max else len(data["counts"])
    irfs = data["irf"] if "irf" in data else [None] * n
    jobs = [(data["counts"][i], float(data["bg_per_bin"][i]), meta, irfs[i], methods) for i in range(n)]
    results = list(pool.map(_fit_image, jobs))
    fg = data["mask"][:n]
    subsets = {"fg": fg, "bright": bright_mask(data["intensity_gt"][:n], fg)}
    jit = {k: jitter_summary(data["tau_gt"][:n], m) for k, m in subsets.items()}
    rows = []
    for name in methods:
        tau = np.stack([r[0][name][0] for r in results])
        inten = np.stack([r[0][name][1] for r in results])
        for subset, m in subsets.items():
            mt = summarize(tau, data["tau_gt"][:n], m)
            mi = summarize(inten, data["intensity_gt"][:n], m)
            rows.append({
                "method": name, "photons": int(round(float(data["photons"][0]))), "subset": subset,
                "n_images": n, "tau_mare_pct": mt["mare_pct"], "tau_bias_pct": mt["bias_pct"],
                "int_mare_pct": mi["mare_pct"], "int_bias_pct": mi["bias_pct"],
                "tau_invalid_frac": mt["invalid_frac"], "n_pixels": mt["n_pixels"],
                "gt_jitter_median_ns": jit[subset]["jitter_median_ns"],
                "gt_jitter_median_pct": jit[subset]["jitter_median_pct"],
            })
            if "tau_int_gt" in data:   # v3: also against the intensity-weighted lifetime
                rows[-1]["tau_mare_vs_tau_int_pct"] = summarize(tau, data["tau_int_gt"][:n], m)["mare_pct"]
            print(f"  {name:7s} {subset:6s} p={rows[-1]['photons']:5d}  tau err {mt['mare_pct']:6.2f}%  "
                  f"bias {mt['bias_pct']:+6.2f}%   I err {mi['mare_pct']:5.2f}%", flush=True)
    print(f"  GT jitter: fg {jit['fg']['jitter_median_pct']:.2f}%, bright {jit['bright']['jitter_median_pct']:.2f}%; "
          f"bright = {subsets['bright'].sum() / fg.sum():.0%} of fg")
    print(f"  ({np.mean([r[1] for r in results]):.1f} s per image for all methods, one worker)")
    return rows


def plot(rows, methods, out):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    titles = {"fg": "all foreground pixels",
              "bright": f"bright fg pixels (>= {BRIGHT_FRAC:.0%} of intensity range)"}
    p_all = np.array(sorted({x["photons"] for x in rows}), dtype=float)
    for i, subset in enumerate(["fg", "bright"]):
        for name in methods:
            r = sorted((x for x in rows if x["method"] == name and x["subset"] == subset), key=lambda x: x["photons"])
            p = [x["photons"] for x in r]
            ax[i, 0].plot(p, [x["tau_mare_pct"] for x in r], STYLES.get(name, "o-"), label=name)
            ax[i, 1].plot(p, [x["int_mare_pct"] for x in r], STYLES.get(name, "o-"), label=name)
        jit = np.median([x["gt_jitter_median_pct"] for x in rows if x["subset"] == subset])
        if jit > 0:
            ax[i, 0].axhline(jit, color="gray", ls=":", label=f"GT local jitter ({jit:.1f}%)")
        ax[i, 1].plot(p_all, 100 * 0.674 / np.sqrt(p_all), ":", color="gray", label="Poisson limit (per pixel)")
        for a, q in zip(ax[i], ["Lifetime", "Intensity"]):
            a.set_xscale("log"); a.set_yscale("log")
            a.set_xlabel("Photons per foreground pixel"); a.set_ylabel("Median |relative error| (%)")
            a.set_title(f"{q} error, {titles[subset]}", fontsize=10); a.grid(True, which="both", alpha=0.3)
            a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default=None, help="folder with test_p*.npz (default: v2 test set)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. v3")
    ap.add_argument("--n", type=int, default=0, help="images per level (0 = all)")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    folder = Path(args.test_set) if args.test_set else test_set_dir()
    files = sorted(folder.glob("test_p*.npz"), key=lambda p: int(p.stem.split("_p")[1]))
    if not files:
        sys.exit(f"No test set found in {folder}")
    methods = args.methods.split(",")

    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for path in files:
            print(path.name, flush=True)
            rows += evaluate_level(path, args.n, methods, pool)

    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    table = TABLES / (f"baselines_{args.tag}.csv" if args.tag else "baselines.csv")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    fig_path = FIGURES / (f"error_vs_photons_baselines_{args.tag}.png" if args.tag else "error_vs_photons.png")
    plot(rows, methods, fig_path)
    print(f"Saved {table}\nSaved {fig_path}")


if __name__ == "__main__":
    main()
