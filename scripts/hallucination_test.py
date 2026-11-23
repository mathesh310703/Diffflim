"""Hallucination test for DiffFLIM v2 (and v3cells for comparison). No training; only the fixed v3
test set and new simulated images built from the 30 TEST IDs (never the second test set).

Part 1 - uniform lifetime. 30 scenes (one per test ID, scene seed UNI_SEED) with the real HPA cell
  shapes and the v3 intensity structure, but every foreground pixel has the same single-component
  lifetime (1, 2.5 or 4 ns), at 10, 50 and 500 photons per foreground pixel (paired: same scenes,
  same IRF; noise seed UNI_SEED + 1000 * tau index + level). Any spatial structure in a lifetime
  map is then an error. Per image, inside the true mask: spatial standard deviation of the
  estimate (% of tau), median absolute relative error, bias, the Pearson correlation between the
  estimate and the true intensity, and the "intensity trend" (|slope| x std(intensity) / tau, %),
  i.e. how much lifetime structure follows the intensity. For DiffFLIM also the predicted standard
  deviation, the calibrated 90 % half-width (factor of calibration_factors_<tag>_extra_s10.csv at
  the estimated photon level) and its coverage of the constant truth, and the same predicted
  standard deviation on the v3 test set at the same level for comparison.
Part 2 - error against uncertainty on the v3 test set (cached 10-step predictions
  data/processed/preds_<tag>_s10_n10_test.npz): pixels binned by DiffFLIM's relative standard
  deviation (deciles per photon level), actual error per bin; error of the rest after removing the
  10 % / 20 % most uncertain pixels, compared with removing the 10 % / 20 % largest errors (oracle).
Part 3 - confident errors: per level, fraction of pixels with relative error > 20 % whose
  calibrated 90 % interval does not contain the truth; three example crops.
Part 4 - small structures: band-pass of the log lifetime (masked Gaussian sigma 1 minus sigma 4) of
  truth and estimate; true features |bp| > 0.1 (about 10 % local contrast). Kept = estimate has
  the same sign and at least half the threshold; lost = the rest; invented = estimate beyond the
  threshold where the truth is below half of it or has the opposite sign. Also the amplitude ratio
  (regression slope of estimated on true band-pass; < 1 = smoothed, > 1 = exaggerated) and the
  correlation. Methods: DiffFLIM mean and single sample, MLE, binned MLE 5 x 5, FLIMngo pretrained.

Writes results/tables/hallucination_uniform.csv, hallucination_error_vs_uncertainty.csv,
       hallucination_confident_errors.csv, hallucination_structures.csv
       results/figures/hallucination_uniform.png, hallucination_uniform_examples.png,
       hallucination_error_vs_uncertainty.png, hallucination_confident_errors.png,
       hallucination_structures.png
Caches data/processed/hallucination_uniform_preds.npz, hallucination_test_preds.npz

Usage:
    python scripts/hallucination_test.py [--n-scenes 30] [--workers 12]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_real_celegans import factor_at, load_factor_curve  # noqa: E402
from src.data import simulate_v3 as v3  # noqa: E402
from src.data.simulate import load_dataset  # noqa: E402
from src.evaluation import methods as M  # noqa: E402
from src.evaluation.calibration import estimate_photon_level  # noqa: E402
from src.paths import FIGURES, PROCESSED, RAW_OWN, TABLES  # noqa: E402

UNI_TAUS = (1.0, 2.5, 4.0)
UNI_LEVELS = (10, 50, 500)
UNI_SEED = 81000
MODELS = (("DiffFLIM v2", "diffflim_v2.pt", "v2"), ("DiffFLIM v3cells", "diffflim_v3cells.pt", "v3cells"))
LEVELS = (10, 20, 50, 100, 500, 1000)
BIG_ERR = 0.20
BP_SIGMAS = (1.0, 4.0)
BP_THR = 0.10
EST = "otsu_adaptive"


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved {path}")


# ----------------------------------------------------------------------------------------
# Part 1: uniform lifetime
# ----------------------------------------------------------------------------------------
def make_uniform_sets(n_scenes):
    cfg = v3.SimConfigV3()
    ids = v3.test_ids()
    rng = np.random.default_rng(UNI_SEED)
    scenes = [v3.make_scene(rng, cfg, ids, fixed_id=ids[i % len(ids)]) for i in range(n_scenes)]
    data = {}
    for ti, tau in enumerate(UNI_TAUS):
        uni = []
        for s in scenes:
            u = dict(s)
            m = s["mask"]
            u["curves"] = v3.decay_curves_v3(np.full(int(m.sum()), tau), s["irf"], cfg)
            u["tau_gt"] = np.where(m, tau, 0.0)
            u["tau_int"] = u["tau_gt"]
            u["n_components"] = m.astype(np.uint8)
            uni.append(u)
        for lev in UNI_LEVELS:
            nrng = np.random.default_rng(UNI_SEED + 1000 * ti + lev)
            d = v3._stack_samples([v3.render_scene(u, cfg, lev, nrng) for u in uni], cfg, v3._meta(cfg))
            d["metadata"] = json.loads(d["metadata"])
            data[(tau, lev)] = d
    return data


def image_stats(est, tau_true, inten, m):
    ok = m & np.isfinite(est) & (est > 0)
    e, it = est[ok].astype(np.float64), inten[ok].astype(np.float64)
    rel = (e - tau_true) / tau_true
    c = np.corrcoef(e, it)[0, 1] if e.std() > 0 and it.std() > 0 else 0.0
    slope = np.polyfit(it, e, 1)[0] if it.std() > 0 else 0.0
    return {"spatial_std_pct": 100 * e.std() / tau_true, "mare_pct": 100 * np.median(np.abs(rel)),
            "bias_pct": 100 * np.median(rel), "corr_intensity": c,
            "intensity_trend_pct": 100 * abs(slope) * it.std() / tau_true, "invalid_frac": 1 - ok.sum() / m.sum()}


def part1(args, models, curves, test_rel_std):
    cache = PROCESSED / "hallucination_uniform_preds.npz"
    data = make_uniform_sets(args.n_scenes)
    if cache.exists():
        with np.load(cache) as z:
            preds = {k: z[k] for k in z.files}
    else:
        preds = {}
        for (tau, lev), d in data.items():
            key = f"{tau}_{lev}"
            t0 = time.perf_counter()
            b = M.run_baselines(d["counts"], d["bg_per_bin"], d["irf"], d["metadata"], ("MLE", "BinMLE5"), args.workers)
            preds[f"{key}|MLE"], preds[f"{key}|BinMLE5"] = b["MLE"][0], b["BinMLE5"][0]
            preds[f"{key}|FLIMngo pretrained"] = M.run_flimngo_pretrained(d["counts"], v3.SimConfigV3().bin_width_ns)
            for label, _, tag in MODELS:
                r = M.run_diffflim(models[tag], d["counts"], seed0=1000)
                for k in ("tau_mean", "tau_std", "tau_s0"):
                    preds[f"{key}|{label}|{k}"] = r[k]
            print(f"uniform tau {tau} ns, {lev} photons: {time.perf_counter() - t0:.0f} s", flush=True)
        np.savez_compressed(cache, **preds)
    rows, per_image = [], {}
    for (tau, lev), d in data.items():
        key = f"{tau}_{lev}"
        maps = {"MLE": preds[f"{key}|MLE"], "Binned MLE 5x5": preds[f"{key}|BinMLE5"],
                "FLIMngo pretrained": preds[f"{key}|FLIMngo pretrained"]}
        for label, _, tag in MODELS:
            maps[f"{label} mean"] = preds[f"{key}|{label}|tau_mean"]
            maps[f"{label} single sample"] = preds[f"{key}|{label}|tau_s0"]
        for name, est in maps.items():
            st = [image_stats(est[i], tau, d["intensity_gt"][i], d["mask"][i]) for i in range(len(est))]
            row = {"tau_true_ns": tau, "photons": lev, "method": name, "n_images": len(st)}
            row.update({k: float(np.median([s[k] for s in st])) for k in st[0]})
            row["frac_images_abs_corr_gt_0.3"] = float(np.mean([abs(s["corr_intensity"]) > 0.3 for s in st]))
            per_image[(tau, lev, name)] = st
            if name.startswith("DiffFLIM") and name.endswith("mean"):
                label = name[:-len(" mean")]
                tag = next(t for lab, _, t in MODELS if lab == label)
                std = preds[f"{key}|{label}|tau_std"]
                m = d["mask"]
                rel_std = [np.median(std[i][m[i]] / est[i][m[i]]) for i in range(len(est))]
                q = [factor_at(curves[tag], estimate_photon_level(d["counts"][i].sum(0), method=EST))[0] for i in range(len(est))]
                cov = [np.mean(np.abs(est[i][m[i]] - tau) <= q[i] * std[i][m[i]]) for i in range(len(est))]
                half = [np.median(q[i] * std[i][m[i]]) / tau for i in range(len(est))]
                row.update({"pred_rel_std_pct": 100 * float(np.median(rel_std)),
                            "pred_rel_std_v3test_pct": 100 * test_rel_std[(tag, lev)],
                            "sampling_noise_of_mean_pct": 100 * float(np.median(rel_std)) / np.sqrt(10),
                            "halfwidth90_rel_pct": 100 * float(np.median(half)), "coverage90": float(np.mean(cov))})
            rows.append(row)
            print(f"tau {tau:3.1f} p {lev:4d} {name:30s} spatial std {row['spatial_std_pct']:6.2f} %  "
                  f"corr(int) {row['corr_intensity']:+.2f}  trend {row['intensity_trend_pct']:5.2f} %  "
                  f"bias {row['bias_pct']:+6.2f} %" + (f"  pred std {row['pred_rel_std_pct']:.1f} % (v3 test "
                  f"{row['pred_rel_std_v3test_pct']:.1f} %) cov90 {row['coverage90']:.3f}" if "pred_rel_std_pct" in row else ""))
    write_csv(TABLES / "hallucination_uniform.csv", rows)
    plot_uniform(rows)
    plot_uniform_examples(data, preds)
    return rows


UNI_STYLE = {"MLE": ("C0", "o-"), "Binned MLE 5x5": ("C3", "v-"), "FLIMngo pretrained": ("C5", "P--"),
             "DiffFLIM v2 mean": ("C4", "D-"), "DiffFLIM v2 single sample": ("C4", "D:"),
             "DiffFLIM v3cells mean": ("C2", "s-"), "DiffFLIM v3cells single sample": ("C2", "s:")}


def plot_uniform(rows):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))
    for j, tau in enumerate(UNI_TAUS):
        for name, (c, ls) in UNI_STYLE.items():
            r = sorted((x for x in rows if x["method"] == name and x["tau_true_ns"] == tau), key=lambda x: x["photons"])
            ax[0, j].plot([x["photons"] for x in r], [x["spatial_std_pct"] for x in r], ls, color=c, label=name)
            ax[1, j].plot([x["photons"] for x in r], [x["intensity_trend_pct"] for x in r], ls, color=c, label=name)
        dm = sorted((x for x in rows if x["method"] == "DiffFLIM v2 mean" and x["tau_true_ns"] == tau), key=lambda x: x["photons"])
        ax[0, j].plot([x["photons"] for x in dm], [x["pred_rel_std_pct"] for x in dm], "x--", color="0.4",
                      label="DiffFLIM v2 predicted std (median)")
        ax[0, j].set_title(f"Uniform lifetime {tau} ns: spatial std of the estimate", fontsize=10)
        ax[1, j].set_title(f"Uniform lifetime {tau} ns: structure following the intensity", fontsize=10)
        for i in range(2):
            a = ax[i, j]
            a.set_xscale("log"); a.set_yscale("log"); a.grid(alpha=0.3, which="both")
            a.set_xlabel("Photons per foreground pixel")
        ax[0, j].set_ylabel("Spatial std inside the mask (% of tau)")
        ax[1, j].set_ylabel("|slope| x std(intensity) / tau (%)")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "hallucination_uniform.png", dpi=110)
    plt.close(fig)
    print(f"Saved {FIGURES / 'hallucination_uniform.png'}")


def plot_uniform_examples(data, preds, tau=2.5, image=0):
    cols = [("summed counts", None), ("MLE", "MLE"), ("Binned MLE 5x5", "BinMLE5"), ("FLIMngo pretrained", "FLIMngo pretrained"),
            ("DiffFLIM v2 mean", "DiffFLIM v2|tau_mean"), ("DiffFLIM v2 single sample", "DiffFLIM v2|tau_s0"),
            ("DiffFLIM v2 std", "DiffFLIM v2|tau_std")]
    fig, ax = plt.subplots(len(UNI_LEVELS), len(cols), figsize=(2.3 * len(cols), 2.4 * len(UNI_LEVELS)))
    for i, lev in enumerate(UNI_LEVELS):
        d = data[(tau, lev)]
        m = d["mask"][image]
        for j, (title, key) in enumerate(cols):
            a = ax[i, j]
            if key is None:
                im = a.imshow(d["counts"][image].sum(0), cmap="magma")
            else:
                x = np.where(m, preds[f"{tau}_{lev}|{key}"][image], np.nan)
                if key.endswith("tau_std"):
                    im = a.imshow(x, cmap="viridis", vmin=0, vmax=0.5 * tau)
                else:
                    im = a.imshow(x, cmap="coolwarm", vmin=0.5 * tau, vmax=1.5 * tau)
            plt.colorbar(im, ax=a, fraction=0.046)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{title}\n{lev} photons" if i == 0 or j == 0 else f"{lev} photons", fontsize=8)
    fig.suptitle(f"Uniform true lifetime {tau} ns (single component), test cell crop; lifetime colour range "
                 f"{0.5 * tau:.2f} to {1.5 * tau:.2f} ns, std 0 to {0.5 * tau:.2f} ns", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "hallucination_uniform_examples.png", dpi=100)
    plt.close(fig)
    print(f"Saved {FIGURES / 'hallucination_uniform_examples.png'}")


# ----------------------------------------------------------------------------------------
# Parts 2 and 3: error against uncertainty, confident errors
# ----------------------------------------------------------------------------------------
def test_files():
    return sorted((RAW_OWN / "sim_v3_test").glob("test_p*.npz"), key=lambda p: int(p.stem.split("_p")[1]))


def load_test_preds(tag):
    with np.load(PROCESSED / f"preds_{tag}_s10_n10_test.npz") as z:
        rec = {k: z[k] for k in z.files}
    est = []
    for f in test_files():
        d = load_dataset(f)
        est += [estimate_photon_level(c.sum(0), method=EST) for c in d["counts"]]
    rec["photons_est"] = np.array(est)
    rec["level"] = np.round(rec["photons"]).astype(int)
    return rec


def part2(recs):
    rows, curves_out = [], {}
    for label, _, tag in MODELS:
        rec = recs[tag]
        for q in ("tau", "int"):
            for lev in LEVELS:
                i = rec["level"] == lev
                m = rec["mask"][i]
                mean, std, gt = rec[f"{q}_mean"][i][m], rec[f"{q}_std"][i][m], rec[f"{q}_gt"][i][m]
                with np.errstate(divide="ignore", invalid="ignore"):
                    err = np.abs(mean - gt) / gt
                    rs = std / np.abs(mean)
                ok = np.isfinite(rs) & np.isfinite(err)          # drops pixels with a mean of exactly 0
                err, rs = err[ok], rs[ok]
                edges = np.quantile(rs, np.linspace(0, 1, 11))
                b = np.clip(np.searchsorted(edges, rs, side="right") - 1, 0, 9)
                curves_out[(tag, q, lev)] = ([100 * np.median(rs[b == k]) for k in range(10)],
                                             [100 * np.median(err[b == k]) for k in range(10)])
                from scipy.stats import spearmanr
                row = {"model": label, "quantity": q, "photons": lev, "n_pixels": int(ok.sum()), "n_pixels_dropped": int((~ok).sum()),
                       "spearman_err_vs_relstd": float(spearmanr(rs, err)[0]), "mare_all_pct": 100 * float(np.median(err))}
                for frac in (0.1, 0.2):
                    keep = rs <= np.quantile(rs, 1 - frac)
                    keep_or = err <= np.quantile(err, 1 - frac)
                    row[f"mare_drop{int(100 * frac)}_uncertain_pct"] = 100 * float(np.median(err[keep]))
                    row[f"mare_drop{int(100 * frac)}_oracle_pct"] = 100 * float(np.median(err[keep_or]))
                    row[f"mare_drop{int(100 * frac)}_random_pct"] = 100 * float(np.median(err))
                    row[f"rel_reduction_drop{int(100 * frac)}_pct"] = 100 * (1 - row[f"mare_drop{int(100 * frac)}_uncertain_pct"] / row["mare_all_pct"])
                    row[f"rel_reduction_drop{int(100 * frac)}_oracle_pct"] = 100 * (1 - row[f"mare_drop{int(100 * frac)}_oracle_pct"] / row["mare_all_pct"])
                    row[f"mean_err_drop{int(100 * frac)}_uncertain_pct"] = 100 * float(np.mean(err[keep]))
                row["mean_err_all_pct"] = 100 * float(np.mean(err))
                row["decile_rel_std_pct"] = " ".join(f"{x:.3g}" for x in curves_out[(tag, q, lev)][0])
                row["decile_err_pct"] = " ".join(f"{x:.3g}" for x in curves_out[(tag, q, lev)][1])
                rows.append(row)
                print(f"{label:16s} {q:3s} p {lev:4d}: spearman {row['spearman_err_vs_relstd']:.2f}  MARE "
                      f"{row['mare_all_pct']:.2f} -> {row['mare_drop10_uncertain_pct']:.2f} (drop 10 %) / "
                      f"{row['mare_drop20_uncertain_pct']:.2f} (drop 20 %); oracle {row['mare_drop10_oracle_pct']:.2f} / "
                      f"{row['mare_drop20_oracle_pct']:.2f}")
    write_csv(TABLES / "hallucination_error_vs_uncertainty.csv", rows)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    cmap = plt.get_cmap("viridis")
    for j, q in enumerate(("tau", "int")):
        a = ax[0, j]
        for k, lev in enumerate(LEVELS):
            x, y = curves_out[("v2", q, lev)]
            a.plot(x, y, "o-", color=cmap(k / 5), label=f"{lev} photons")
            x3, y3 = curves_out[("v3cells", q, lev)]
            a.plot(x3, y3, ":", color=cmap(k / 5))
        lim = [1.5, 30]
        a.plot(lim, lim, "k--", lw=0.8, label="error = predicted std")
        a.plot(lim, [0.674 * v for v in lim], "k:", lw=0.8, label="median |N(0, std)| = 0.674 std")
        a.set_xscale("log"); a.set_yscale("log"); a.grid(alpha=0.3, which="both")
        a.set_xlabel("Predicted relative std (decile median, %)"); a.set_ylabel("Actual |relative error| (decile median, %)")
        a.set_title(f"({'ab'[j]}) {'Lifetime' if q == 'tau' else 'Intensity'}: error per uncertainty decile\n"
                    "DiffFLIM v2 solid, v3cells dotted; v3 test set, 10 steps x 10 samples", fontsize=10)
        a.legend(fontsize=7)
        a = ax[1, j]
        w = 0.13
        for k, (key, lab, hatch) in enumerate((("mare_all_pct", "all pixels", ""),
                                               ("mare_drop10_uncertain_pct", "10 % most uncertain removed", ""),
                                               ("mare_drop20_uncertain_pct", "20 % most uncertain removed", ""),
                                               ("mare_drop10_oracle_pct", "10 % largest errors removed (oracle)", "//"),
                                               ("mare_drop20_oracle_pct", "20 % largest errors removed (oracle)", "//"))):
            vals = [next(r[key] for r in rows if r["model"] == "DiffFLIM v2" and r["quantity"] == q and r["photons"] == lev)
                    for lev in LEVELS]
            a.bar(np.arange(len(LEVELS)) + (k - 2) * w, vals, w, label=lab, hatch=hatch,
                  color=["0.6", "C0", "C9", "C3", "C1"][k], edgecolor="k", linewidth=0.4)
        a.set_xticks(range(len(LEVELS))); a.set_xticklabels([str(x) for x in LEVELS])
        a.set_xlabel("Photons per foreground pixel"); a.set_ylabel("Median |relative error| (%)")
        a.set_title(f"({'cd'[j]}) {'Lifetime' if q == 'tau' else 'Intensity'}: error of the remaining pixels, DiffFLIM v2", fontsize=10)
        a.legend(fontsize=7); a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "hallucination_error_vs_uncertainty.png", dpi=110)
    plt.close(fig)
    print(f"Saved {FIGURES / 'hallucination_error_vs_uncertainty.png'}")
    return rows


def part3(recs, curves):
    rows, examples = [], []
    for label, _, tag in MODELS:
        rec = recs[tag]
        q90 = np.array([factor_at(curves[tag], e)[0] for e in rec["photons_est"]])
        for lev in LEVELS + ("all",):
            idx = np.arange(len(rec["level"])) if lev == "all" else np.where(rec["level"] == lev)[0]
            tot = big = big_out = out = 0
            for i in idx:
                m = rec["mask"][i]
                gt, mean, std = rec["tau_gt"][i][m], rec["tau_mean"][i][m], rec["tau_std"][i][m]
                err = np.abs(mean - gt)
                b = err / gt > BIG_ERR
                o = err > q90[i] * std
                tot += m.sum(); big += b.sum(); big_out += (b & o).sum(); out += o.sum()
                if tag == "v2" and lev != "all":
                    examples.append((float((b & o).sum() / m.sum()), int(i)))
            rows.append({"model": label, "photons": lev, "n_images": len(idx), "n_pixels": int(tot),
                         "frac_err_gt20": big / tot, "frac_outside90": out / tot,
                         "frac_err_gt20_outside90_of_err_gt20": big_out / max(big, 1),
                         "frac_confident_errors_of_all": big_out / tot})
            r = rows[-1]
            print(f"{label:16s} p {str(lev):>4s}: error > 20 % in {100 * r['frac_err_gt20']:.2f} % of pixels; of these "
                  f"{100 * r['frac_err_gt20_outside90_of_err_gt20']:.1f} % outside the 90 % interval "
                  f"({100 * r['frac_confident_errors_of_all']:.2f} % of all pixels)")
    write_csv(TABLES / "hallucination_confident_errors.csv", rows)
    # three example crops: the v2 test images with the largest share of confident errors (different cells)
    rec, q90 = recs["v2"], np.array([factor_at(curves["v2"], e)[0] for e in recs["v2"]["photons_est"]])
    ids = load_test_ids_per_image()
    chosen, seen = [], set()
    for share, i in sorted(examples, reverse=True):
        if ids[i] not in seen:
            chosen.append(i); seen.add(ids[i])
        if len(chosen) == 3:
            break
    fig, ax = plt.subplots(3, 5, figsize=(15, 9))
    for r, i in enumerate(chosen):
        m = rec["mask"][i]
        gt, mean, std = rec["tau_gt"][i], rec["tau_mean"][i], rec["tau_std"][i]
        lo, hi = np.percentile(gt[m], [1, 99])
        relerr = np.where(m, (mean - gt) / np.where(m, gt, 1), np.nan)
        conf = m & (np.abs(mean - gt) / np.where(m, gt, 1) > BIG_ERR) & (np.abs(mean - gt) > q90[i] * std)
        panels = [("true lifetime (ns)", np.where(m, gt, np.nan), "viridis", lo, hi),
                  ("DiffFLIM v2 mean (ns)", np.where(m, mean, np.nan), "viridis", lo, hi),
                  ("relative error", relerr, "coolwarm", -0.5, 0.5),
                  ("calibrated 90 % half-width / tau", np.where(m, q90[i] * std / np.where(m, gt, 1), np.nan), "magma", 0, 0.6)]
        for j, (t, x, cm, a0, a1) in enumerate(panels):
            im = ax[r, j].imshow(x, cmap=cm, vmin=a0, vmax=a1)
            plt.colorbar(im, ax=ax[r, j], fraction=0.046)
            ax[r, j].set_title(t if r else t, fontsize=9)
        ax[r, 4].imshow(np.where(m, 0.85, 1.0), cmap="gray", vmin=0, vmax=1)
        yy, xx = np.where(conf)
        ax[r, 4].scatter(xx, yy, s=4, color="C3")
        ax[r, 4].set_title(f"confident errors (red): {100 * conf.sum() / m.sum():.1f} % of mask", fontsize=9)
        ax[r, 0].set_ylabel(f"{rec['level'][i]} photons, image {i % 40}", fontsize=9)
        for a in ax[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("DiffFLIM v2: the three test images (different cells) with the largest share of confident errors "
                 "(error > 20 % and truth outside the calibrated 90 % interval)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "hallucination_confident_errors.png", dpi=100)
    plt.close(fig)
    print(f"Saved {FIGURES / 'hallucination_confident_errors.png'}")
    return rows


def load_test_ids_per_image():
    return np.concatenate([np.asarray(load_dataset(f)["hpa_id"]).astype(str) for f in test_files()])


# ----------------------------------------------------------------------------------------
# Part 4: small structures
# ----------------------------------------------------------------------------------------
def masked_smooth(x, w, sigma):
    num = gaussian_filter(np.where(w, x, 0.0), sigma)
    den = gaussian_filter(w.astype(np.float64), sigma)
    return num / np.maximum(den, 1e-6)


def bandpass(tau, w):
    lt = np.log(np.clip(np.where(w, tau, 1.0), 1e-3, None))
    return masked_smooth(lt, w, BP_SIGMAS[0]) - masked_smooth(lt, w, BP_SIGMAS[1])


def test_set_predictions(args, models):
    cache = PROCESSED / "hallucination_test_preds.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    out = {k: [] for k in ("MLE", "BinMLE5", "FLIMngo pretrained", "v2_s0", "v3cells_s0", "v2_mean_check")}
    for f in test_files():
        d = load_dataset(f)
        meta = d["metadata"] if isinstance(d["metadata"], dict) else json.loads(d["metadata"])
        b = M.run_baselines(d["counts"], d["bg_per_bin"], d["irf"], meta, ("MLE", "BinMLE5"), args.workers)
        out["MLE"].append(b["MLE"][0]); out["BinMLE5"].append(b["BinMLE5"][0])
        out["FLIMngo pretrained"].append(M.run_flimngo_pretrained(d["counts"], float(meta["bin_width_ns"])))
        for tag in ("v2", "v3cells"):
            r = M.run_diffflim(models[tag], d["counts"], seed0=1000)
            out[f"{tag}_s0"].append(r["tau_s0"])
            if tag == "v2":
                out["v2_mean_check"].append(r["tau_mean"])
        print(f"  test predictions {f.name} done", flush=True)
    out = {k: np.concatenate(v) for k, v in out.items()}
    np.savez_compressed(cache, **out)
    return out


def part4(args, models, recs):
    tp = test_set_predictions(args, models)
    dev = np.nanmax(np.abs(tp["v2_mean_check"] - recs["v2"]["tau_mean"]))
    print(f"DiffFLIM v2 rerun vs cached mean: max difference {dev:.4f} ns (same seeds)")
    maps = {"DiffFLIM v2 mean": recs["v2"]["tau_mean"], "DiffFLIM v2 single sample": tp["v2_s0"],
            "DiffFLIM v3cells mean": recs["v3cells"]["tau_mean"], "DiffFLIM v3cells single sample": tp["v3cells_s0"],
            "MLE": tp["MLE"], "Binned MLE 5x5": tp["BinMLE5"], "FLIMngo pretrained": tp["FLIMngo pretrained"]}
    rec = recs["v2"]
    bp_true = [bandpass(rec["tau_gt"][i], rec["mask"][i]) for i in range(len(rec["mask"]))]
    rows = []
    for name, est in maps.items():
        for lev in LEVELS:
            idx = np.where(rec["level"] == lev)[0]
            acc = {"true": 0, "kept": 0, "pred": 0, "invented": 0}
            xs, ys = [], []
            for i in idx:
                m = rec["mask"][i]
                w = m & np.isfinite(est[i]) & (est[i] > 0)
                bt, bp = bp_true[i][w], bandpass(est[i], w)[w]
                tf = np.abs(bt) > BP_THR
                kept = tf & (np.sign(bp) == np.sign(bt)) & (np.abs(bp) > BP_THR / 2)
                pf = np.abs(bp) > BP_THR
                inv = pf & ((np.abs(bt) < BP_THR / 2) | (np.sign(bp) != np.sign(bt)))
                acc["true"] += tf.sum(); acc["kept"] += kept.sum(); acc["pred"] += pf.sum(); acc["invented"] += inv.sum()
                xs.append(bt); ys.append(bp)
            x, y = np.concatenate(xs), np.concatenate(ys)
            rows.append({"method": name, "photons": lev, "n_pixels": len(x), "true_feature_frac": acc["true"] / len(x),
                         "kept_frac_of_true": acc["kept"] / max(acc["true"], 1),
                         "lost_frac_of_true": 1 - acc["kept"] / max(acc["true"], 1),
                         "pred_feature_frac": acc["pred"] / len(x),
                         "invented_frac_of_pred": acc["invented"] / max(acc["pred"], 1),
                         "invented_frac_of_pixels": acc["invented"] / len(x),
                         "amplitude_ratio": float((x * y).sum() / (x * x).sum()), "corr": float(np.corrcoef(x, y)[0, 1]),
                         "rms_bp_true": float(np.sqrt((x * x).mean())), "rms_bp_pred": float(np.sqrt((y * y).mean()))})
            r = rows[-1]
            print(f"{name:30s} p {lev:4d}: kept {100 * r['kept_frac_of_true']:5.1f} %  invented {100 * r['invented_frac_of_pred']:5.1f} % "
                  f"of predicted features ({100 * r['invented_frac_of_pixels']:.2f} % of pixels)  amplitude ratio "
                  f"{r['amplitude_ratio']:.2f}  corr {r['corr']:.2f}  rms {r['rms_bp_pred']:.3f}/{r['rms_bp_true']:.3f}")
    write_csv(TABLES / "hallucination_structures.csv", rows)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for name, (c, ls) in UNI_STYLE.items():
        r = sorted((x for x in rows if x["method"] == name), key=lambda x: x["photons"])
        for a, key in zip(ax, ("kept_frac_of_true", "invented_frac_of_pixels", "amplitude_ratio")):
            a.plot([x["photons"] for x in r], [x[key] * (100 if key != "amplitude_ratio" else 1) for x in r], ls, color=c, label=name)
    for a, t, yl in zip(ax, ("(a) True features kept", "(b) Invented features", "(c) Amplitude of small structures"),
                        ("Kept features (% of true features)", "Invented features (% of foreground pixels)",
                         "Slope of estimated on true band-pass")):
        a.set_xscale("log"); a.grid(alpha=0.3, which="both"); a.set_xlabel("Photons per foreground pixel")
        a.set_ylabel(yl); a.set_title(t, fontsize=10)
    ax[1].set_yscale("log")
    ax[2].axhline(1, color="k", ls="--", lw=0.8)
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Small lifetime structures on the v3 test set: band-pass of log lifetime (Gaussian sigma "
                 f"{BP_SIGMAS[0]:g} minus {BP_SIGMAS[1]:g} pixels), feature = |band-pass| > {BP_THR}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "hallucination_structures.png", dpi=110)
    plt.close(fig)
    print(f"Saved {FIGURES / 'hallucination_structures.png'}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=30)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    t0 = time.perf_counter()
    models = {tag: M.load_diffflim(ck) for _, ck, tag in MODELS}
    curves = {tag: load_factor_curve(TABLES / f"calibration_factors_{tag}_extra_s10.csv") for _, _, tag in MODELS}
    recs = {tag: load_test_preds(tag) for _, _, tag in MODELS}
    test_rel_std = {}
    for _, _, tag in MODELS:
        r = recs[tag]
        for lev in UNI_LEVELS:
            i = r["level"] == lev
            m = r["mask"][i]
            test_rel_std[(tag, lev)] = float(np.median(r["tau_std"][i][m] / r["tau_mean"][i][m]))
    print("== Part 1: uniform lifetime"); part1(args, models, curves, test_rel_std)
    print("== Part 2: error against uncertainty"); part2(recs)
    print("== Part 3: confident errors"); part3(recs, curves)
    print("== Part 4: small structures"); part4(args, models, recs)
    print(f"Total {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
