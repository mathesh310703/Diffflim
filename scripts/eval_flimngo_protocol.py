"""Evaluation under FLIMngo's protocol: mean squared error of the lifetime (ns^2) per image over the
foreground, as box plots, with a Kruskal-Wallis test per condition and paired tests of DiffFLIM v2
against every other method. No training; images are built from the 30 TEST IDs only (never the
second test set).

Data (paired: the same 40 scenes in every condition; scene seed SCENE_SEED, 40 scenes round robin
over the 30 test IDs, the v3 simulator otherwise unchanged: HPA crops, up to four components,
amplitude-weighted ground truth, background 2 % of the mean foreground intensity):
  * IRF widths: the three simulated skew-normal IRFs of the FLIMngo library without secondary peak
    (IRF_simulated_narrow, IRF_simulated, IRF_simulated_broad; FWHM printed at the start, about
    220, 400 and 800 ps), with FLIMngo's laser jitter (fixed per scene and IRF).
  * Photon conditions:
      "high": FLIMngo's high setting, intensity = min + normalised channel sum x delta with
              min ~ U(100, 2000), delta ~ U(100, 500) photons per pixel (100 to 2500);
      "low":  the same with min ~ U(25, 60), delta ~ U(10, 40) (25 to 100 photons per pixel; FLIMngo's
              own low setting starts at 10);
      sweep 100, 80, 50, 20, 10: the v3 intensity structure scaled to this mean number of signal
              photons per foreground pixel (the photon level of the v3 test set).
    The per-scene draws of min and delta are fixed (seed SCENE_SEED + 7).
Methods: MLE, phasor, binned MLE 5 x 5 (v3 mono-exponential model with each image's IRF), FLIMngo
pretrained and FLIMngo retrained on v3 (recall-oriented mask computed from the counts), DiffFLIM v2
and v3cells (10 DDIM steps x 10 samples, mean). DiffFLIM was trained on 10 to 1000 photons per
foreground pixel, so the "high" condition lies partly outside its training range.
Score: per image, mean of (estimate - truth)^2 over the true foreground pixels where the estimate
is finite (the MLE has no lifetime for pixels without photons; their share is reported).
Tests per condition: Kruskal-Wallis over all methods; DiffFLIM v2 against each other method with the
two-sided Wilcoxon signed-rank test (paired images), Holm correction within the condition.

Writes results/tables/flimngo_protocol.csv (summary), results/tables/flimngo_protocol_images.csv
       (per image), results/figures/flimngo_protocol_<irf>.png (box plots, one per IRF width) and
       results/figures/flimngo_protocol_summary.png
Caches data/processed/flimngo_protocol_cache.npz (per-image errors, so that a restart skips done sets)

Usage:
    python scripts/eval_flimngo_protocol.py [--n-scenes 40] [--workers 12]
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
from scipy.stats import kruskal, wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import simulate_v3 as v3  # noqa: E402
from src.data.simulate import sample_counts  # noqa: E402
from src.evaluation import methods as M  # noqa: E402
from src.paths import FIGURES, PROCESSED, TABLES  # noqa: E402

SCENE_SEED = 91000
IRFS = (("narrow", "IRF_simulated_narrow"), ("medium", "IRF_simulated"), ("broad", "IRF_simulated_broad"))
CONDITIONS = ("high", "low", "100", "80", "50", "20", "10")
RANGES = {"high": ((100, 2000), (100, 500)), "low": ((25, 60), (10, 40))}
METHODS = ("MLE", "Phasor", "Binned MLE 5x5", "FLIMngo pretrained", "FLIMngo retrained v3", "DiffFLIM v2", "DiffFLIM v3cells")
REF = "DiffFLIM v2"
COLORS = {"MLE": "C0", "Phasor": "C7", "Binned MLE 5x5": "C3", "FLIMngo pretrained": "C5", "FLIMngo retrained v3": "C1",
          "DiffFLIM v2": "C4", "DiffFLIM v3cells": "C2"}


def fwhm_ps(irf, dt_ns):
    t = np.arange(len(irf)) * dt_ns
    fine = np.linspace(0, t[-1], 20000)
    y = np.interp(fine, t, irf)
    above = fine[y >= y.max() / 2]
    return 1000 * (above[-1] - above[0])


def make_scenes(n, cfg):
    ids = v3.test_ids()
    rng = np.random.default_rng(SCENE_SEED)
    order = [ids[i] for i in rng.permutation(len(ids))]
    scenes = [v3.make_scene(rng, cfg, ids, fixed_id=order[i % len(order)]) for i in range(n)]
    draw = np.random.default_rng(SCENE_SEED + 7)
    for s in scenes:
        chan = np.moveaxis(s["channels"], -1, 0).astype(np.float64).sum(0)
        s["norm_sum"] = (chan - chan.min()) / max(chan.max() - chan.min(), 1e-12)
        s["draws"] = {c: (draw.uniform(*RANGES[c][0]), draw.uniform(*RANGES[c][1])) for c in RANGES}
    return scenes


def irf_for(scenes, name, cfg, k):
    lib = {x["id"]: x for x in v3.irf_library(cfg)}[name]
    rng = np.random.default_rng(SCENE_SEED + 100 + k)
    out = []
    for s in scenes:
        irf, _ = v3.jitter_irf(lib["irf"], lib["jitter"], rng)
        m = s["mask"]
        out.append((irf, v3.multi_exp_curves(s["amps"][:, m], s["taus"][:, m], irf, cfg)))
    return out


def render(scenes, irfcurves, cond, cfg, seed):
    rng = np.random.default_rng(seed)
    counts, inten_all, bgs = [], [], []
    for s, (irf, curves) in zip(scenes, irfcurves):
        m = s["mask"]
        if cond in RANGES:
            lo, delta = s["draws"][cond]
            inten = np.where(m, lo + s["norm_sum"] * delta, 0.0)
        else:
            inten = s["rel_intensity"] * float(cond)
        bg = cfg.bg_fraction * inten[m].mean() / cfg.n_bins
        expected = np.full((cfg.n_bins,) + m.shape, bg)
        expected[:, m] += inten[m][None] * curves
        counts.append(sample_counts(expected, rng))
        inten_all.append(inten); bgs.append(bg)
    return np.stack(counts), np.stack(inten_all), np.array(bgs)


def mse_per_image(est, gt, mask):
    out, inv = [], []
    for e, g, m in zip(est, gt, mask):
        ok = m & np.isfinite(e)
        out.append(float(np.mean((e[ok] - g[ok]) ** 2)))
        inv.append(1 - ok.sum() / m.sum())
    return np.array(out), np.array(inv)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=40)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    t0 = time.perf_counter()
    cfg = v3.SimConfigV3()
    meta = v3._meta(cfg)
    for name, irf_id in IRFS:
        irf = {x["id"]: x for x in v3.irf_library(cfg)}[irf_id]["irf"]
        print(f"IRF {name}: {irf_id}, FWHM {fwhm_ps(irf, cfg.bin_width_ns):.0f} ps")
    scenes = make_scenes(args.n_scenes, cfg)
    gt = np.stack([s["tau_gt"] for s in scenes])
    mask = np.stack([s["mask"] for s in scenes])
    hpa = [s["hpa_id"] for s in scenes]
    cache_path = PROCESSED / "flimngo_protocol_cache.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    models = {"DiffFLIM v2": M.load_diffflim("diffflim_v2.pt"), "DiffFLIM v3cells": M.load_diffflim("diffflim_v3cells.pt")}
    net_re = M.load_flimngo_retrained()
    photons_mean = {}
    for k, (iname, irf_id) in enumerate(IRFS):
        irfcurves = irf_for(scenes, irf_id, cfg, k)
        irfs = np.stack([x[0] for x in irfcurves]).astype(np.float32)
        for c, cond in enumerate(CONDITIONS):
            key = f"{iname}|{cond}"
            counts, inten, bgs = render(scenes, irfcurves, cond, cfg, SCENE_SEED + 1000 + 100 * k + c)
            photons_mean[key] = float(np.mean([inten[i][mask[i]].mean() for i in range(len(inten))]))
            if f"{key}|{REF}|mse" in cache:
                continue
            t1 = time.perf_counter()
            preds = {}
            b = M.run_baselines(counts, bgs, irfs, meta, ("MLE", "Phasor", "BinMLE5"), args.workers)
            preds["MLE"], preds["Phasor"], preds["Binned MLE 5x5"] = b["MLE"][0], b["Phasor"][0], b["BinMLE5"][0]
            fm = M.flimngo_masks(counts)
            preds["FLIMngo pretrained"] = M.run_flimngo_pretrained(counts, cfg.bin_width_ns, fm)
            preds["FLIMngo retrained v3"] = M.run_flimngo_retrained(counts, net_re, fm)
            for name, model in models.items():
                preds[name] = M.run_diffflim(model, counts, seed0=1000)["tau_mean"]
            for name, est in preds.items():
                mse, inv = mse_per_image(est, gt, mask)
                cache[f"{key}|{name}|mse"], cache[f"{key}|{name}|invalid"] = mse, inv
            np.savez(cache_path, **cache)
            print(f"IRF {iname:6s} {cond:>4s} (mean {photons_mean[key]:7.1f} photons per fg pixel): " +
                  "  ".join(f"{n.replace('DiffFLIM ', 'D-').replace('FLIMngo ', 'F-')} {np.median(cache[f'{key}|{n}|mse']):.3f}"
                            for n in METHODS) + f"   ({time.perf_counter() - t1:.0f} s)", flush=True)

    rows, img_rows = [], []
    for iname, _ in IRFS:
        for cond in CONDITIONS:
            key = f"{iname}|{cond}"
            data = {n: cache[f"{key}|{n}|mse"] for n in METHODS}
            H, p_kw = kruskal(*data.values())
            others = [n for n in METHODS if n != REF]
            p_raw = [wilcoxon(data[REF], data[n]).pvalue for n in others]
            p_adj = dict(zip(others, holm(p_raw)))
            p_raw = dict(zip(others, p_raw))
            for n in METHODS:
                x = data[n]
                rows.append({"irf": iname, "condition": cond, "mean_photons_fg": photons_mean[key], "method": n,
                             "n_images": len(x), "median_mse_ns2": float(np.median(x)),
                             "q25_mse_ns2": float(np.percentile(x, 25)), "q75_mse_ns2": float(np.percentile(x, 75)),
                             "mean_mse_ns2": float(np.mean(x)), "invalid_frac": float(np.mean(cache[f"{key}|{n}|invalid"])),
                             "kruskal_H": float(H), "kruskal_p": float(p_kw),
                             "median_diff_vs_diffflim_v2_ns2": "" if n == REF else float(np.median(data[REF] - x)),
                             "frac_images_diffflim_v2_better": "" if n == REF else float(np.mean(data[REF] < x)),
                             "wilcoxon_p_vs_diffflim_v2": "" if n == REF else float(p_raw[n]),
                             "holm_p_vs_diffflim_v2": "" if n == REF else float(p_adj[n])})
                for i, v in enumerate(x):
                    img_rows.append({"irf": iname, "condition": cond, "image": i, "hpa_id": hpa[i], "method": n,
                                     "mse_ns2": float(v), "invalid_frac": float(cache[f"{key}|{n}|invalid"][i])})
    for path, rr in ((TABLES / "flimngo_protocol.csv", rows), (TABLES / "flimngo_protocol_images.csv", img_rows)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0]))
            w.writeheader()
            for r in rr:
                w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
        print(f"Saved {path}")
    for iname, _ in IRFS:
        fig, ax = plt.subplots(1, len(CONDITIONS), figsize=(3.0 * len(CONDITIONS), 5.2), sharey=True)
        for j, cond in enumerate(CONDITIONS):
            key = f"{iname}|{cond}"
            data = [cache[f"{key}|{n}|mse"] for n in METHODS]
            bp = ax[j].boxplot(data, patch_artist=True, widths=0.6, showfliers=True, flierprops={"markersize": 2})
            for patch, n in zip(bp["boxes"], METHODS):
                patch.set_facecolor(COLORS[n]); patch.set_alpha(0.6)
            ax[j].set_yscale("log")
            ax[j].set_xticks(range(1, len(METHODS) + 1))
            ax[j].set_xticklabels([n.replace("FLIMngo ", "FLIMngo\n").replace("DiffFLIM ", "DiffFLIM\n").replace("Binned MLE 5x5", "Binned\nMLE 5x5")
                                   for n in METHODS], rotation=90, fontsize=7)
            kw = next(r["kruskal_p"] for r in rows if r["irf"] == iname and r["condition"] == cond)
            label = f"{cond} photons" if cond not in RANGES else f"'{cond}' ({'100-2500' if cond == 'high' else '25-100'})"
            ax[j].set_title(f"{label}\nKruskal-Wallis p = {kw:.1e}", fontsize=8)
            ax[j].grid(alpha=0.3, axis="y")
        ax[0].set_ylabel("MSE per image over the foreground (ns$^2$)")
        fig.suptitle(f"FLIMngo protocol, IRF {iname} ({dict(IRFS)[iname]}), {args.n_scenes} images per condition from test cells", fontsize=10)
        fig.tight_layout()
        out = FIGURES / f"flimngo_protocol_{iname}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"Saved {out}")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for j, (iname, _) in enumerate(IRFS):
        for n in METHODS:
            y = [next(r["median_mse_ns2"] for r in rows if r["irf"] == iname and r["condition"] == c and r["method"] == n)
                 for c in CONDITIONS]
            ax[j].plot(range(len(CONDITIONS)), y, "o-", color=COLORS[n], label=n)
        ax[j].set_xticks(range(len(CONDITIONS)))
        ax[j].set_xticklabels(["high\n100-2500", "low\n25-100"] + list(CONDITIONS[2:]))
        ax[j].set_yscale("log"); ax[j].grid(alpha=0.3, which="both")
        ax[j].set_xlabel("Photon condition (sweep: mean photons per foreground pixel)")
        ax[j].set_title(f"IRF {iname}: median MSE per image", fontsize=10)
    ax[0].set_ylabel("Median MSE (ns$^2$)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "flimngo_protocol_summary.png", dpi=110)
    plt.close(fig)
    print(f"Saved {FIGURES / 'flimngo_protocol_summary.png'}")
    print(f"Total {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
