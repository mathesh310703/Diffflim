"""Evaluate the MLE and phasor baselines on the fixed simulated test set.

Reads   data/raw/own/sim_test/test_p*.npz   (make it with: python -m src.data.simulate)
Writes  results/tables/baselines.csv
        results/figures/error_vs_photons.png

Metrics are computed on foreground pixels only, pooled over all images of a photon level.

Usage:
    python scripts/eval_baselines.py              # all 40 images per level
    python scripts/eval_baselines.py --n 5        # quick run on 5 images per level
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines.mle import fit_mle  # noqa: E402
from src.baselines.phasor import fit_phasor  # noqa: E402
from src.data.simulate import config_from_metadata, load_dataset, test_set_dir  # noqa: E402
from src.evaluation.metrics import summarize  # noqa: E402
from src.paths import FIGURES, TABLES  # noqa: E402

METHODS = {
    "MLE": lambda c, cfg, b: fit_mle(c, cfg, b)[:2],
    "Phasor": lambda c, cfg, b: fit_phasor(c, cfg, b)[:2],
}


def evaluate_level(path, n_max):
    data = load_dataset(path)
    cfg = config_from_metadata(data["metadata"])
    n = min(n_max, len(data["counts"])) if n_max else len(data["counts"])
    rows = []
    for name, fit in METHODS.items():
        taus, ints, t0 = [], [], time.perf_counter()
        for i in range(n):
            tau, inten = fit(data["counts"][i], cfg, float(data["bg_per_bin"][i]))
            taus.append(tau)
            ints.append(inten)
        sec = (time.perf_counter() - t0) / n
        m = data["mask"][:n]
        mt = summarize(np.stack(taus), data["tau_gt"][:n], m)
        mi = summarize(np.stack(ints), data["intensity_gt"][:n], m)
        rows.append({
            "method": name, "photons": int(round(float(data["photons"][0]))), "n_images": n,
            "tau_mare_pct": mt["mare_pct"], "tau_bias_pct": mt["bias_pct"],
            "int_mare_pct": mi["mare_pct"], "int_bias_pct": mi["bias_pct"],
            "tau_invalid_frac": mt["invalid_frac"], "n_fg_pixels": mt["n_pixels"],
            "sec_per_image": sec,
        })
        print(f"  {name:7s} p={rows[-1]['photons']:5d}  tau err {mt['mare_pct']:6.2f}%  "
              f"bias {mt['bias_pct']:+6.2f}%   I err {mi['mare_pct']:5.2f}%  ({sec:.2f} s/img)")
    return rows


def plot(rows, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, style in zip(METHODS, ["o-", "s--"]):
        r = sorted((x for x in rows if x["method"] == name), key=lambda x: x["photons"])
        p = [x["photons"] for x in r]
        ax[0].plot(p, [x["tau_mare_pct"] for x in r], style, label=name)
        ax[1].plot(p, [x["int_mare_pct"] for x in r], style, label=name)
    p = np.array(sorted({x["photons"] for x in rows}), dtype=float)
    # Poisson limit for intensity: median |N - I| / I ~= 0.674 / sqrt(I)
    ax[1].plot(p, 100 * 0.674 / np.sqrt(p), ":", color="gray", label="Poisson limit (0.674/sqrt(N))")
    for a, title in zip(ax, ["Lifetime", "Intensity"]):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("Photons per foreground pixel")
        a.set_ylabel("Median |relative error| (%)")
        a.set_title(f"{title} error (foreground pixels)")
        a.grid(True, which="both", alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="images per level (0 = all)")
    args = ap.parse_args()

    files = sorted(test_set_dir().glob("test_p*.npz"), key=lambda p: int(p.stem.split("_p")[1]))
    if not files:
        sys.exit("No test set found. Run: python -m src.data.simulate")

    rows = []
    for path in files:
        print(path.name)
        rows += evaluate_level(path, args.n)

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    table = TABLES / "baselines.csv"
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    fig_path = FIGURES / "error_vs_photons.png"
    plot(rows, fig_path)
    print(f"Saved {table}\nSaved {fig_path}")

    # Sanity check: at the highest level the MLE should be within a few percent.
    top = max(r["photons"] for r in rows)
    mle = next(r for r in rows if r["method"] == "MLE" and r["photons"] == top)
    ok = mle["tau_mare_pct"] < 3.0 and abs(mle["tau_bias_pct"]) < 1.0
    print(f"Sanity check (MLE at {top} photons): tau error {mle['tau_mare_pct']:.2f}%, "
          f"bias {mle['tau_bias_pct']:+.2f}% -> {'OK' if ok else 'NOT OK, investigate'}")


if __name__ == "__main__":
    main()
