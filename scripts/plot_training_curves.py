"""Training-curve figures, regenerated from the saved training logs (no model or GPU needed).

  diffflim   results/figures/loss_v2.png from logs/diffflim_v2.csv (scripts/train_diffflim.py):
             left: logged loss (every 50 steps) and its running median over 100 log entries (centred), log
             axis, with the time of the 3-hour snapshot; right: data-wait fraction (centred running mean
             over 100 log entries) with the 50 % limit.
  flimngo    results/figures/training_flimngo_v3.png from logs/flimngo_v3.csv and
             logs/flimngo_v3_val.csv (scripts/train_flimngo.py): left: training MSE (logged values and
             moving mean over 20 log entries) and validation MSE, log axis; right: median relative
             lifetime error on the validation set.

Both figures were first made with inline commands; this script reproduces them from the same logs.

Usage:
    python scripts/plot_training_curves.py              # both figures
    python scripts/plot_training_curves.py diffflim     # only loss_v2.png
    python scripts/plot_training_curves.py flimngo --log logs/flimngo_v3.csv \
        --val logs/flimngo_v3_val.csv --out results/figures/training_flimngo_v3.png
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
from src.paths import FIGURES, LOGS  # noqa: E402


def read_columns(path):
    """CSV log -> {column: float array}."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def running(x, k, fn):
    """Centred running statistic over a window of k values, shortened at the ends (same length as x)."""
    h = k // 2
    return np.array([fn(x[max(0, i - h):i + h + 1]) for i in range(len(x))])


def plot_diffflim(log, out, snapshot_min=180.0, title="DiffFLIM v2", window=100):
    d = read_columns(log)
    t = d["elapsed_min"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(t, d["loss"], color="C0", lw=0.5, alpha=0.35, label="logged loss (every 50 steps)")
    ax[0].plot(t, running(d["loss"], window, np.median), color="C0", lw=1.5,
               label=f"running median ({window} points)")
    if snapshot_min:
        ax[0].axvline(snapshot_min, color="gray", ls=":", lw=1.5, label="3-hour snapshot")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Training time (min)")
    ax[0].set_ylabel("Training loss (v-prediction MSE)")
    ax[0].set_title(f"{title} training loss")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, 100 * running(d["data_wait_frac"], window, np.mean), color="C1")
    ax[1].axhline(50, color="red", ls="--", label="50 % limit")
    ax[1].set_ylim(0, 60)
    ax[1].set_xlabel("Training time (min)")
    ax[1].set_ylabel("Data wait (% of step time)")
    ax[1].set_title("Fraction of time the GPU waited for data")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_flimngo(log, val, out, title="FLIMngo architecture trained on v3 (flimngo_v3)", window=20):
    tr, va = read_columns(log), read_columns(val)
    t, loss = tr["elapsed_min"], tr["loss"]
    smooth = np.convolve(loss, np.ones(window) / window, "valid")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(t, loss, alpha=0.25, color="C0")
    ax[0].plot(t[window - 1:], smooth, color="C0", label="training (moving mean)")
    ax[0].plot(va["elapsed_min"], va["val_mse"], "o-", color="C1", label="validation")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Training time (min)")
    ax[0].set_ylabel("MSE (ns$^2$)")
    ax[0].legend()
    ax[0].grid(alpha=0.3, which="both")
    ax[1].plot(va["elapsed_min"], va["val_mare_pct"], "o-", color="C1")
    ax[1].set_xlabel("Training time (min)")
    ax[1].set_ylabel("Validation median |rel. error| (%)")
    ax[1].grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", choices=["diffflim", "flimngo", "both"], default="both")
    ap.add_argument("--log", default=None, help="training log (default: logs/diffflim_v2.csv or logs/flimngo_v3.csv)")
    ap.add_argument("--val", default=None, help="validation log (flimngo only; default logs/flimngo_v3_val.csv)")
    ap.add_argument("--out", default=None, help="output figure")
    args = ap.parse_args()
    if args.which in ("diffflim", "both"):
        plot_diffflim(Path(args.log) if args.log and args.which == "diffflim" else LOGS / "diffflim_v2.csv",
                      Path(args.out) if args.out and args.which == "diffflim" else FIGURES / "loss_v2.png")
    if args.which in ("flimngo", "both"):
        plot_flimngo(Path(args.log) if args.log and args.which == "flimngo" else LOGS / "flimngo_v3.csv",
                     Path(args.val) if args.val else LOGS / "flimngo_v3_val.csv",
                     Path(args.out) if args.out and args.which == "flimngo" else FIGURES / "training_flimngo_v3.png")


if __name__ == "__main__":
    main()
