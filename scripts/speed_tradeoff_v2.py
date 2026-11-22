"""Speed / accuracy / calibration trade-off of DiffFLIM v2: DDIM steps (50, 20, 10) x samples (10, 5).

For every setting:
  * calibration: predictions on both v3 calibration sets (74 cells, 1200 images) with the same
    setting, then the same procedure as coverage_calibrated_v2_extra (photon estimate
    "otsu_adaptive", global factor and per-level quadratic curve in log photons);
  * test: the first 10 images per photon level of the v3 test set (60 images): lifetime and
    intensity error (median absolute relative error, true foreground), 90 % coverage after
    calibration, and the time per image (sampling incl. binning and copy to the CPU, GPU idle
    otherwise, median over the 60 images).
The 50-step / 10-sample calibration predictions are taken from the existing caches.

Writes results/tables/speed_tradeoff_v2.csv
       results/figures/speed_tradeoff_v2.png

Usage:
    python scripts/speed_tradeoff_v2.py
    python scripts/speed_tradeoff_v2.py --plot-only      # figure from the saved table
"""
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
sys.path.insert(0, str(ROOT / "scripts"))
import calibrate_diffflim as cd  # noqa: E402
from src.data.simulate import load_dataset  # noqa: E402
from src.evaluation import calibration as cal  # noqa: E402
from src.evaluation.metrics import summarize  # noqa: E402
from src.models.diffflim import load_model  # noqa: E402
from src.paths import CHECKPOINTS, FIGURES, PROCESSED, RAW_OWN, TABLES  # noqa: E402

STEPS = (50, 20, 10)
SAMPLES = (10, 5)
N_TEST = 10
EST = "otsu_adaptive"
LEVELS = (10, 20, 50, 100, 500, 1000)


def files_of(folder, prefix):
    return sorted(folder.glob(f"{prefix}_p*.npz"), key=lambda p: int(p.stem.split("_p")[1]))


def predict(model, files, n_images, n_samples, steps, seed0, timed=False):
    keys = ["tau_mean", "tau_std", "int_mean", "int_std", "tau_gt", "int_gt", "mask", "photons", "photons_est"]
    rec = {k: [] for k in keys}
    times = []
    for path in files:
        d = load_dataset(path)
        n = min(n_images or len(d["counts"]), len(d["counts"]))
        for i in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            s = model.sample(d["counts"][i], n_samples=n_samples, steps=steps, seed=seed0 + i)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            rec["tau_mean"].append(s[:, 0].mean(0)); rec["tau_std"].append(s[:, 0].std(0, ddof=1))
            rec["int_mean"].append(s[:, 1].mean(0)); rec["int_std"].append(s[:, 1].std(0, ddof=1))
            rec["tau_gt"].append(d["tau_gt"][i]); rec["int_gt"].append(d["intensity_gt"][i])
            rec["mask"].append(d["mask"][i]); rec["photons"].append(float(d["photons"][i]))
            rec["photons_est"].append(cal.estimate_photon_level(d["counts"][i].sum(0), method=EST))
    rec = {k: np.stack(v) if np.ndim(v[0]) else np.array(v) for k, v in rec.items()}
    return (rec, times) if timed else rec


def calib_record(model, steps, n_samples):
    recs = []
    for k, d in enumerate([RAW_OWN / "sim_v3_calib", RAW_OWN / "sim_v3_calib_extra"]):
        key = cd.calib_cache_key(d)
        files = files_of(d, "calib")
        if steps == 50 and n_samples == 10:          # existing caches (same seeds as below)
            with np.load(PROCESSED / f"preds_v2_{key}.npz") as f:
                rec = {x: f[x] for x in f.files}
            rec = cd.recompute_photons_est(rec, files, 0, EST)
        else:
            cache = PROCESSED / f"preds_v2_s{steps}_n{n_samples}_{key}.npz"
            if cache.exists():
                with np.load(cache) as f:
                    rec = {x: f[x] for x in f.files}
            else:
                t0 = time.perf_counter()
                rec = predict(model, files, 0, n_samples, steps, 5000 if key == "calib" else 7000)
                np.savez_compressed(cache, **rec)
                print(f"    {d.name}: {len(rec['photons'])} images in {time.perf_counter() - t0:.0f} s", flush=True)
        recs.append(rec)
    return cd.concat_records(recs)


def main():
    model, ckpt = load_model(CHECKPOINTS / "diffflim_v2.pt", "cuda")
    test_files = files_of(RAW_OWN / "sim_v3_test", "test")
    # warm-up (cuDNN autotuning) so the first timed image is not slower
    d0 = load_dataset(test_files[0])
    for st in STEPS:
        model.sample(d0["counts"][0], n_samples=10, steps=st, seed=0)

    rows = []
    for steps in STEPS:
        for ns in SAMPLES:
            print(f"steps {steps}, samples {ns}", flush=True)
            test, times = predict(model, test_files, N_TEST, ns, steps, 1000, timed=True)
            calib = calib_record(model, steps, ns)
            fitted = cal.fit(calib, (0.10,), smooth=True)
            groups = [("all", np.arange(len(test["photons"])))] + \
                     [(lev, np.where(test["photons"] == lev)[0]) for lev in LEVELS]
            for lev, idx in groups:
                sub = {k: v[idx] for k, v in test.items()}
                row = {"steps": steps, "n_samples": ns, "photons": lev, "n_images": len(idx),
                       "sec_per_image": float(np.median(np.array(times)[idx])),
                       "tau_mare_pct": summarize(sub["tau_mean"], sub["tau_gt"], sub["mask"])["mare_pct"],
                       "int_mare_pct": summarize(sub["int_mean"], sub["int_gt"], sub["mask"])["mare_pct"]}
                for q in cal.QUANTITIES:
                    for v in ("none", "global", "per_level"):
                        f = cal.predict_q(fitted, q, 0.10, v, sub["photons_est"])
                        row[f"cov90_{q}_{v}"] = cal.interval_stats(sub, q, f)[0]
                    row[f"factor90_{q}_global"] = fitted["global"][(q, 0.10)]
                rows.append(row)
                if lev == "all":
                    print(f"  {1e3 * row['sec_per_image']:.0f} ms/image  tau {row['tau_mare_pct']:.2f}%  "
                          f"I {row['int_mare_pct']:.2f}%  cov90 tau {row['cov90_tau_none']:.3f}->"
                          f"{row['cov90_tau_global']:.3f}/{row['cov90_tau_per_level']:.3f}  int "
                          f"{row['cov90_int_none']:.3f}->{row['cov90_int_global']:.3f}/{row['cov90_int_per_level']:.3f}",
                          flush=True)

    out = TABLES / "speed_tradeoff_v2.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved {out}")
    plot(out)


def plot(table):
    """Figure from the saved table (all photon levels pooled)."""
    with open(table) as f:
        allr = [{k: (v if k == "photons" else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f) if r["photons"] == "all"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    for ns, (mk, mk_int) in zip(SAMPLES, [("o-", "o:"), ("s--", "s-.")]):
        r = sorted((x for x in allr if x["n_samples"] == ns), key=lambda x: x["sec_per_image"])
        t = [1e3 * x["sec_per_image"] for x in r]
        ax[0].plot(t, [x["tau_mare_pct"] for x in r], mk, label=f"{ns} samples")
        ax[1].plot(t, [x["int_mare_pct"] for x in r], mk, label=f"{ns} samples")
        ax[2].plot(t, [100 * x["cov90_tau_global"] for x in r], mk, label=f"τ, {ns} samples")
        ax[2].plot(t, [100 * x["cov90_int_global"] for x in r], mk_int, label=f"intensity, {ns} samples")
        for x, tt in zip(r, t):
            ax[0].annotate(f"{x['steps']:.0f}", (tt, x["tau_mare_pct"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[2].axhline(90, color="k", ls="--", lw=1)
    for a, yl, ti in zip(ax, ["Lifetime error (%)", "Intensity error (%)", "Coverage of 90 % interval (%)"],
                         ["Lifetime (labels: DDIM steps)", "Intensity", "Calibrated coverage (global factor)"]):
        a.set_xscale("log"); a.set_xlabel("Time per image (ms)"); a.set_ylabel(yl); a.set_title(ti, fontsize=10)
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "speed_tradeoff_v2.png", dpi=120)
    print(f"Saved {FIGURES / 'speed_tradeoff_v2.png'}")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(TABLES / "speed_tradeoff_v2.csv")
    else:
        main()
