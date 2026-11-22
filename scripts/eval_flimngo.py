"""Evaluate the pretrained FLIMngo network (src/baselines/flimngo_baseline.py) on the v3 test set.

FLIMngo predicts the lifetime only. Main setting (as in FLIMngo's own simulated data): background
outside the ground-truth foreground set to zero, per-pixel max normalisation of the decays,
64 x 64 input. Sensitivity variants: zero-padded to 256 x 256, Otsu mask from the counts, no
mask, no normalisation.

Metrics as for the other baselines (src/evaluation/metrics.py) on all foreground pixels ("fg")
and bright foreground pixels ("bright"); also the error against the intensity-weighted lifetime
and the fraction of foreground pixels predicted as exactly 0 ns (the network ends in a ReLU).

Writes  results/tables/flimngo_v3.csv
        results/figures/error_vs_photons_v2_flimngo.png   (with MLE, binned MLE, phasor, DiffFLIM v2)

Usage:
    python scripts/eval_flimngo.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines import flimngo_baseline as fg_net  # noqa: E402
from src.data.simulate import load_dataset  # noqa: E402
from src.evaluation.metrics import BRIGHT_FRAC, bright_mask, summarize  # noqa: E402
from src.paths import FIGURES, TABLES  # noqa: E402

# name -> (mask rule, normalise, pad_to)
VARIANTS = {
    "main":       ("gt", True, None),
    "pad256":     ("gt", True, 256),
    "otsu_mask":  ("otsu", True, None),
    "no_mask":    ("none", True, None),
    "no_norm":    ("gt", False, None),
}


def run_variant(data, n, bin_width, rule, norm, pad):
    counts = data["counts"][:n]
    if rule == "gt":
        mask = data["mask"][:n]
    elif rule == "otsu":
        mask = np.stack([fg_net.otsu_mask(c) for c in counts])
    else:
        mask = None
    return fg_net.predict(counts, mask, bin_width=bin_width, normalise_decays=norm, pad_to=pad)


def timing(data, bin_width, pad=None, reps=20):
    """Seconds per image at batch size 1 (network only; preprocessing timed separately)."""
    c, m = data["counts"][:1], data["mask"][:1]
    fg_net.predict(c, m, bin_width=bin_width, pad_to=pad)            # warm-up (cuDNN autotune)
    net_t = np.median([fg_net.predict(c, m, bin_width=bin_width, pad_to=pad, batch=1)[1] for _ in range(reps)])
    t0 = time.perf_counter()
    for _ in range(reps):
        fg_net.preprocess(c[0], m[0], True, pad)
    return float(net_t), (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default="data/raw/own/sim_v3_test")
    ap.add_argument("--n", type=int, default=0, help="images per level (0 = all)")
    args = ap.parse_args()
    folder = ROOT / args.test_set
    files = sorted(folder.glob("test_p*.npz"), key=lambda p: int(p.stem.split("_p")[1]))

    rows = []
    for path in files:
        data = load_dataset(path)
        bw = float(data["metadata"].get("bin_width_ns", 0.0977)) if isinstance(data["metadata"], dict) else 0.0977
        n = args.n or len(data["counts"])
        p = int(round(float(data["photons"][0])))
        subsets = {"fg": data["mask"][:n], "bright": bright_mask(data["intensity_gt"][:n], data["mask"][:n])}
        for name, (rule, norm, pad) in VARIANTS.items():
            tau, sec = run_variant(data, n, bw, rule, norm, pad)
            for subset, m in subsets.items():
                st = summarize(tau, data["tau_gt"][:n], m)
                rows.append({
                    "method": "FLIMngo", "variant": name, "photons": p, "subset": subset, "n_images": n,
                    "tau_mare_pct": st["mare_pct"], "tau_bias_pct": st["bias_pct"],
                    "tau_mare_vs_tau_int_pct": summarize(tau, data["tau_int_gt"][:n], m)["mare_pct"],
                    "tau_zero_frac": float(np.mean(tau[m] <= 0)), "tau_invalid_frac": st["invalid_frac"],
                    "n_pixels": st["n_pixels"], "sec_per_image_batched": sec,
                })
                print(f"p={p:5d} {name:9s} {subset:6s} tau err {st['mare_pct']:6.2f}%  bias {st['bias_pct']:+6.2f}%  "
                      f"zero {rows[-1]['tau_zero_frac']:.3f}", flush=True)

    t64, pre64 = timing(data, bw)
    t256, _ = timing(data, bw, pad=256)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Speed ({gpu}, batch 1, network only): 64x64 {1e3 * t64:.1f} ms, padded 256x256 {1e3 * t256:.1f} ms; "
          f"preprocessing {1e3 * pre64:.1f} ms per image")
    for r in rows:
        r["sec_per_image_batch1_64"] = t64
        r["sec_per_image_batch1_256"] = t256

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "flimngo_v3.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved {out}")
    plot(rows, FIGURES / "error_vs_photons_v2_flimngo.png")


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot(rows, out):
    base = _read(TABLES / "baselines_v3.csv")
    dif = _read(TABLES / "diffflim_v2.csv")
    curves = {}
    for subset in ["fg", "bright"]:
        for m in ["MLE", "Phasor", "BinMLE3", "BinMLE5"]:
            curves[(m, subset)] = sorted((int(r["photons"]), float(r["tau_mare_pct"]))
                                         for r in base if r["method"] == m and r["subset"] == subset)
        key = "all_fg" if subset == "fg" else "bright"
        curves[("DiffFLIM v2", subset)] = sorted((int(r["photons"]), float(r["tau_mare_pct"]))
                                                 for r in dif if r["method"] == "DiffFLIM" and r["subset"] == key)
        curves[("FLIMngo", subset)] = sorted((r["photons"], r["tau_mare_pct"]) for r in rows
                                             if r["variant"] == "main" and r["subset"] == subset)
    styles = {"MLE": ("o-", "C0"), "Phasor": ("s--", "C1"), "BinMLE3": ("^-", "C2"), "BinMLE5": ("v-", "C3"),
              "DiffFLIM v2": ("D-", "C4"), "FLIMngo": ("P-", "C5")}
    labels = {"BinMLE3": "Binned MLE 3x3", "BinMLE5": "Binned MLE 5x5", "FLIMngo": "FLIMngo (pretrained)"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    titles = {"fg": "all foreground pixels", "bright": f"bright fg pixels (>= {BRIGHT_FRAC:.0%} of intensity range)"}
    for a, subset in zip(ax, ["fg", "bright"]):
        for m, (ls, c) in styles.items():
            pts = curves[(m, subset)]
            a.plot([x for x, _ in pts], [y for _, y in pts], ls, color=c, label=labels.get(m, m))
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("Photons per foreground pixel"); a.set_ylabel("Median |relative error| (%)")
        a.set_title(f"Lifetime error, {titles[subset]}", fontsize=10); a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
