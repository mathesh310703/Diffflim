"""Fixed evaluation protocol for the FLUTE real data (Gottlieb et al. 2023, Zenodo 8046636, CC BY 4.0).

Defined once (2026-11-23) so that every later model is evaluated identically; nothing here is tuned on
the results. A DiffFLIM model for the FLUTE time axis (56 bins, e.g. checkpoints/diffflim_flute.pt) is
passed with --ckpt/--tag/--factors; without --ckpt only the classical methods are run.

Methods
  MLE, binned MLE 5 x 5 and phasor with the periodic FLUTE forward model (src/data/flute.py) and a
  cross-estimated IRF (below); DiffFLIM (10 DDIM steps x 10 samples, overlapping 64 x 64 tiles with
  stride 32 and a linear blending ramp, seeds 1000 + tile index + part offset).

IRFs (wrapped Gaussian, estimated on the summed decay of a fluorescein file with its reference
lifetime fixed, then moved by the difference of the model-free half-maximum rise times)
  Fluorescein_hMSC   <- IRF from Fluorescein_Embryo (tau fixed at 4.0 ns)       (non-circular)
  Fluorescein_Embryo <- IRF from Fluorescein_hMSC  (tau fixed at 3.61 ns)       (non-circular)
  Embryo             <- IRF from Fluorescein_Embryo (tau 4.0 ns), same setup
  hMSC control, hMSC-ZOOM, hMSC_rotenone <- IRF from Fluorescein_hMSC (tau 3.61 ns), same setup
Background per pixel and bin: for fluorescein from the global fit with the transferred IRF (scaled by
the thinning factor); for the cell images 0 (measured background on fluorescein: 0.6 to 1.9 %).

Part a: fluorescein with a reference lifetime
  Fluorescein_hMSC thinned binomially to 10, 20, 50, 100, 500 photons per pixel (seed 5000 + level, as
  in scripts/flute_fluorescein.py), reference 3.61 ns (its own full-photon lifetime; the stated value is
  4.0 ns); Fluorescein_Embryo at full counts (about 20 photons per pixel), reference 4.0 ns (stated;
  measured 4.04 ns). Per method: median lifetime, bias and median absolute relative error against the
  reference, spatial spread (0.7413 x IQR / median), invalid pixels; for DiffFLIM also the median
  predicted relative std and the coverage of the reference by the uncalibrated (mean +- 1.645 std) and
  the calibrated (mean +- q std, q from --factors at the estimated photon level) 90 % interval.
Part b: cell images (Embryo, hMSC control, hMSC-ZOOM, hMSC_rotenone), no ground truth
  Split-half consistency: A = binomial(counts, 0.5) (seed 6000 + file index), B = counts - A (two
  independent Poisson halves); per method the median over the mask of |tau_A - tau_B| / mean(tau_A,
  tau_B); DiffFLIM uses different sampler seeds for full, A and B. Mask: Otsu threshold on the summed
  counts (flimngo_baseline.otsu_mask). Lifetime maps of all methods and DiffFLIM uncertainty maps
  (calibrated 90 % half-width). No accuracy claims.

The classical results do not depend on the model and are cached in
data/processed/real_flute_baselines.npz.

Optional --flimngo-ckpt: the FLIMngo architecture retrained on the FLUTE simulator
(scripts/train_flimngo_flute.py): 56 bins resized linearly to 256, resized bin width, non-overlapping
64 x 64 tiles, background mask computed per tile from the counts (otsu_adaptive, as in its training);
pixels outside the mask count as invalid. Example:
    python scripts/eval_real_flute.py --flimngo-ckpt checkpoints/flimngo_flute.pt --tag flimngo_flute

Optional --guide-kappa K --guide-stop S (24 November 2026): DiffFLIM with the Poisson data-consistency
guidance of src/models/guidance.py, using exactly the forward model of the MLE above for each file (the
periodic FLUTE model with the cross-estimated IRF; background from the global fit for fluorescein, 0 for
the cells). The settings are chosen on simulated validation data (scripts/select_guidance.py), never here.

Writes results/tables/real_flute_eval_<tag>.csv, results/tables/real_flute_splithalf_<tag>.csv,
results/figures/real_flute_eval_<tag>.png and results/figures/real_flute_cells_<tag>.png.

Usage:
    python scripts/eval_real_flute.py                                   # classical methods, tag "baselines"
    python scripts/eval_real_flute.py --ckpt checkpoints/diffflim_flute.pt --tag flute \
        --factors results/tables/calibration_factors_flute_s10.csv
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
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_real_celegans import factor_at, load_factor_curve  # noqa: E402
from flute_fluorescein import fit_global, run_methods  # noqa: E402
from src.baselines.flimngo_baseline import otsu_mask  # noqa: E402
from src.data.flute import (FluteConfig, binomial_thin, gaussian_irf_fine, load_flute,  # noqa: E402
                            rise_time)
from src.evaluation.calibration import estimate_photon_level  # noqa: E402
from src.paths import FIGURES, PROCESSED, TABLES  # noqa: E402

REF = {"fluorescein_hmsc": 3.61, "fluorescein_embryo": 4.0}
STATED = 4.0
HMSC_LEVELS = (10, 20, 50, 100, 500)
THIN_SEED = 5000
SPLIT_SEED = 6000
CELLS = ("embryo", "hmsc_control", "hmsc_zoom", "hmsc_rotenone")
CELL_IRF_SOURCE = {"embryo": "fluorescein_embryo", "hmsc_control": "fluorescein_hmsc",
                   "hmsc_zoom": "fluorescein_hmsc", "hmsc_rotenone": "fluorescein_hmsc"}
CLASSICAL = ("MLE", "BinMLE5", "Phasor")
TILE, STRIDE = 64, 32
PART_SEED = {"full": 0, "A": 100000, "B": 200000}
ALPHA = 0.10
PHOTON_ESTIMATOR = "otsu_adaptive"
TAU_DISPLAY_CELLS = (0.5, 3.5)
CACHE = PROCESSED / "real_flute_baselines.npz"


# ----------------------------------------------------------------------------------------
# IRFs
# ----------------------------------------------------------------------------------------
def fluorescein_irfs(cfg):
    """IRF (t0, sigma) of each fluorescein file with its reference lifetime fixed, and its rise time."""
    out = {}
    for k, tau in REF.items():
        dec = load_flute(k).sum((1, 2)).astype(float)
        g = fit_global(dec, cfg, tau=tau)
        out[k] = {"t0": g["t0"], "sigma": g["sigma"], "rise": rise_time(dec, cfg)}
    return out


def irf_for(target, source, irfs, cfg, dec_target):
    shift = rise_time(dec_target, cfg) - irfs[source]["rise"]
    return gaussian_irf_fine(cfg, irfs[source]["t0"] + shift, irfs[source]["sigma"]), shift


# ----------------------------------------------------------------------------------------
# DiffFLIM, tiled
# ----------------------------------------------------------------------------------------
def tile_weight():
    i = np.arange(TILE)
    ramp = np.minimum(np.minimum(i + 1, TILE - i) / (TILE / 2), 1.0)
    return np.outer(ramp, ramp)


def run_diffflim_tiled(model, counts, steps, n_samples, seed0, guidance=None):
    _, H, W = counts.shape
    ys = sorted(set(list(range(0, H - TILE + 1, STRIDE)) + [H - TILE]))
    xs = sorted(set(list(range(0, W - TILE + 1, STRIDE)) + [W - TILE]))
    w = tile_weight()
    acc_m, acc_v, acc_w = np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
    k = 0
    for y in ys:
        for x in xs:
            s = model.sample(counts[:, y:y + TILE, x:x + TILE], n_samples=n_samples, steps=steps, seed=seed0 + k,
                             guidance=guidance)
            acc_m[y:y + TILE, x:x + TILE] += w * s[:, 0].mean(0)
            acc_v[y:y + TILE, x:x + TILE] += w * s[:, 0].var(0, ddof=1)
            acc_w[y:y + TILE, x:x + TILE] += w
            k += 1
    return acc_m / acc_w, np.sqrt(acc_v / acc_w)


# ----------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------
def fluor_stats(tau, inten, ref, border=0):
    if border:
        tau, inten = tau[border:-border, border:-border], inten[border:-border, border:-border]
    ok = np.isfinite(tau)
    v = tau[ok]
    med = float(np.median(v))
    q1, q3 = np.percentile(v, [25, 75])
    return {"median_tau_ns": med, "bias_pct": 100 * (med - ref) / ref,
            "mare_pct": 100 * float(np.median(np.abs(v - ref) / ref)),
            "bias_vs_stated_pct": 100 * (med - STATED) / STATED,
            "spread_rel_std_pct": 100 * 0.7413 * float(q3 - q1) / med, "invalid_pct": 100 * float(1 - ok.mean()),
            "spearman_tau_intensity": float(spearmanr(v, inten[ok]).correlation)}


def split_diff(ta, tb, mask):
    ok = mask & np.isfinite(ta) & np.isfinite(tb) & (ta > 0) & (tb > 0)
    d = np.abs(ta[ok] - tb[ok]) / (0.5 * (ta[ok] + tb[ok]))
    return 100 * float(np.median(d)), int(ok.sum())


# ----------------------------------------------------------------------------------------
# Data conditions
# ----------------------------------------------------------------------------------------
def fluor_conditions():
    """[(key, condition name, counts, thinning factor p)] for part a."""
    out = []
    d = load_flute("fluorescein_hmsc")
    mean_ph = d.sum(0).mean()
    for lv in HMSC_LEVELS:
        p = lv / mean_ph
        out.append(("fluorescein_hmsc", str(lv), binomial_thin(d, p, THIN_SEED + lv), p))
    out.append(("fluorescein_embryo", "full", load_flute("fluorescein_embryo"), 1.0))
    return out


def cell_parts(i, k):
    c = load_flute(k)
    a = binomial_thin(c, 0.5, SPLIT_SEED + i)
    return {"full": c, "A": a, "B": c - a}


def baselines(cfg, irfs):
    """Classical maps for all conditions (cached)."""
    if CACHE.exists():
        z = dict(np.load(CACHE, allow_pickle=True))
        return {k: z[k] for k in z}
    out = {}
    npx = {}
    for key, cname, counts, p in fluor_conditions():
        src = "fluorescein_embryo" if key == "fluorescein_hmsc" else "fluorescein_hmsc"
        full = load_flute(key)
        npx[key] = full.shape[1] * full.shape[2]
        irf, _ = irf_for(key, src, irfs, cfg, full.sum((1, 2)).astype(float))
        g = fit_global(full.sum((1, 2)).astype(float), cfg, irf=irf)
        res = run_methods(counts, irf, cfg, g["b"] / npx[key] * p)
        for m in CLASSICAL:
            out[f"fluor|{key}|{cname}|{m}"] = res[m][0]
        out[f"fluor|{key}|{cname}|intensity"] = counts.sum(0).astype(np.float32)
        print(f"baselines {key} {cname} done", flush=True)
    for i, k in enumerate(CELLS):
        parts = cell_parts(i, k)
        irf, _ = irf_for(k, CELL_IRF_SOURCE[k], irfs, cfg, parts["full"].sum((1, 2)).astype(float))
        for part, counts in parts.items():
            res = run_methods(counts, irf, cfg, 0.0)
            for m in CLASSICAL:
                out[f"cell|{k}|{part}|{m}"] = res[m][0]
        print(f"baselines {k} done", flush=True)
    np.savez_compressed(CACHE, **out)
    return out


def make_guidance(kind, key, p, irfs, cfg, kappa, stop):
    """Guidance with the forward model of the MLE for one file (kind "fluor" or "cell")."""
    from src.data.flute import PeriodicDecayModel
    from src.models.guidance import PoissonGuidance
    full = load_flute(key)
    dec = full.sum((1, 2)).astype(float)
    if kind == "fluor":
        src = "fluorescein_embryo" if key == "fluorescein_hmsc" else "fluorescein_hmsc"
        irf, _ = irf_for(key, src, irfs, cfg, dec)
        bg = fit_global(dec, cfg, irf=irf)["b"] / (full.shape[1] * full.shape[2]) * p
    else:
        irf, _ = irf_for(key, CELL_IRF_SOURCE[key], irfs, cfg, dec)
        bg = 0.0
    return PoissonGuidance(PeriodicDecayModel(irf, cfg, tau_bounds=(0.02, 60.0)), bg, kappa, stop_frac=stop)


# ----------------------------------------------------------------------------------------
def load_flimngo_flute(path):
    import torch
    from src.models.flimngo_arch import build_flimngo
    ck = torch.load(path, map_location="cuda", weights_only=False)
    net = build_flimngo(64, ck["arch"]["bin_width"]).to("cuda").eval()
    net.load_state_dict(ck["model"])
    return net


def run_flimngo_flute(net, counts):
    import torch
    from src.evaluation.calibration import foreground_mask
    from train_flimngo_flute import flute_predict
    _, H, W = counts.shape
    out = np.full((H, W), np.nan)
    with torch.no_grad():
        for y in range(0, H, TILE):
            for x in range(0, W, TILE):
                c = counts[:, y:y + TILE, x:x + TILE]
                m = foreground_mask(c.sum(0), "otsu_adaptive")
                t = flute_predict(net, torch.from_numpy(c[None].astype(np.float32)).cuda(),
                                  torch.from_numpy(m[None]).cuda())[0].cpu().numpy()
                out[y:y + TILE, x:x + TILE] = np.where(m, t, np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--flimngo-ckpt", default="")
    ap.add_argument("--tag", default="baselines")
    ap.add_argument("--factors", default="")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--guide-kappa", type=float, default=None)
    ap.add_argument("--guide-stop", type=float, default=1.0)
    args = ap.parse_args()
    guided = args.guide_kappa is not None
    t0 = time.time()
    cfg = FluteConfig()
    irfs = fluorescein_irfs(cfg)
    for k, v in irfs.items():
        print(f"IRF from {k}: t0 {v['t0']:.3f} ns, sigma {v['sigma']:.3f} ns, rise {v['rise']:.3f} ns")
    base = baselines(cfg, irfs)
    print(f"classical methods ready [{time.time() - t0:.0f} s]", flush=True)

    model, curve = None, None
    if args.ckpt:
        from src.models.diffflim import load_model
        model = load_model(args.ckpt)[0]
        curve = load_factor_curve(args.factors, ALPHA) if args.factors else None
    fnet = load_flimngo_flute(args.flimngo_ckpt) if args.flimngo_ckpt else None

    # ---- part a
    rows, dmaps = [], {}
    for key, cname, counts, p in fluor_conditions():
        ref = REF[key]
        inten = counts.sum(0).astype(float)
        photons = float(inten.mean())
        for m in CLASSICAL:
            st = fluor_stats(base[f"fluor|{key}|{cname}|{m}"], inten, ref, border=2 if m == "BinMLE5" else 0)
            rows.append({"file": key, "condition": cname, "photons_px_mean": photons, "reference_ns": ref,
                         "method": m, **st})
        if fnet is not None:
            st = fluor_stats(run_flimngo_flute(fnet, counts), inten, ref)
            rows.append({"file": key, "condition": cname, "photons_px_mean": photons, "reference_ns": ref,
                         "method": "FLIMngo-FLUTE", **st})
        if model is not None:
            gd = make_guidance("fluor", key, p, irfs, cfg, args.guide_kappa, args.guide_stop) if guided else None
            mean, std = run_diffflim_tiled(model, counts, args.steps, args.n_samples, 1000, gd)
            st = fluor_stats(mean, inten, ref)
            est = estimate_photon_level(inten, method=PHOTON_ESTIMATOR)
            row = {"file": key, "condition": cname, "photons_px_mean": photons, "reference_ns": ref,
                   "method": f"DiffFLIM {args.tag}", **st,
                   "median_rel_std_pct": 100 * float(np.median(std / mean)),
                   "cov90_uncal": float(np.mean(np.abs(mean - ref) <= 1.645 * std)), "photons_est": est}
            if curve is not None:
                q, flag = factor_at(curve, est)
                row.update({"q90": q, "cov90_cal": float(np.mean(np.abs(mean - ref) <= q * std)),
                            "photons_est_outside_calib": int(flag)})
            rows.append(row)
            dmaps[(key, cname)] = (mean, std)
        print(f"part a {key} {cname} [{time.time() - t0:.0f} s]", flush=True)

    # ---- part b
    srows, cell_maps = [], {}
    for i, k in enumerate(CELLS):
        parts = cell_parts(i, k)
        mask = otsu_mask(parts["full"])
        maps = {m: {part: base[f"cell|{k}|{part}|{m}"] for part in parts} for m in CLASSICAL}
        if fnet is not None:
            maps["FLIMngo-FLUTE"] = {part: run_flimngo_flute(fnet, counts) for part, counts in parts.items()}
        if model is not None:
            maps["DiffFLIM"] = {}
            stds = {}
            gd = make_guidance("cell", k, 1.0, irfs, cfg, args.guide_kappa, args.guide_stop) if guided else None
            for part, counts in parts.items():
                mean, std = run_diffflim_tiled(model, counts, args.steps, args.n_samples, 1000 + PART_SEED[part], gd)
                maps["DiffFLIM"][part], stds[part] = mean, std
            q = factor_at(curve, estimate_photon_level(parts["full"].sum(0).astype(float), method=PHOTON_ESTIMATOR))[0] \
                if curve is not None else 1.645
            cell_maps[k] = {"halfwidth": q * stds["full"]}
        for m, mp in maps.items():
            diff, n = split_diff(mp["A"], mp["B"], mask)
            full = mp["full"][mask & np.isfinite(mp["full"])]
            srows.append({"file": k, "method": m if m != "DiffFLIM" else f"DiffFLIM {args.tag}", "n_mask_px": n,
                          "photons_px_median_mask": float(np.median(parts["full"].sum(0)[mask])),
                          "median_tau_full_ns": float(np.median(full)), "splithalf_median_rel_diff_pct": diff})
        cell_maps.setdefault(k, {})
        cell_maps[k].update({"intensity": parts["full"].sum(0), "mask": mask,
                             **{m: mp["full"] for m, mp in maps.items()}})
        print(f"part b {k} [{time.time() - t0:.0f} s]", flush=True)

    TABLES.mkdir(parents=True, exist_ok=True)
    for path, rr in ((TABLES / f"real_flute_eval_{args.tag}.csv", rows), (TABLES / f"real_flute_splithalf_{args.tag}.csv", srows)):
        keys = list(dict.fromkeys(k for r in rr for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rr:
                w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
        print(f"Saved {path}")
    plot_fluor(rows, FIGURES / f"real_flute_eval_{args.tag}.png")
    plot_cells(cell_maps, srows, FIGURES / f"real_flute_cells_{args.tag}.png")
    for r in rows:
        extra = f" cov90 cal {r['cov90_cal']:.3f}" if "cov90_cal" in r else ""
        print(f"{r['file']:18s} {r['condition']:>4s} {r['method']:16s} median {r['median_tau_ns']:.3f} bias {r['bias_pct']:+.1f} % "
              f"MARE {r['mare_pct']:.1f} % spread {r['spread_rel_std_pct']:.1f} %{extra}")
    for r in srows:
        print(f"{r['file']:14s} {r['method']:16s} split-half {r['splithalf_median_rel_diff_pct']:.2f} %  median tau {r['median_tau_full_ns']:.2f} ns")
    print(f"done in {time.time() - t0:.0f} s")


def plot_fluor(rows, out):
    hm = [r for r in rows if r["file"] == "fluorescein_hmsc"]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    for j, m in enumerate(methods):
        rr = sorted([r for r in hm if r["method"] == m], key=lambda r: r["photons_px_mean"])
        x = [r["photons_px_mean"] for r in rr]
        ax[0].plot(x, [r["bias_pct"] for r in rr], "o-", color=f"C{j}", label=m)
        ax[1].plot(x, [r["mare_pct"] for r in rr], "o-", color=f"C{j}", label=m)
        ax[2].plot(x, [r["spread_rel_std_pct"] for r in rr], "o-", color=f"C{j}", label=m)
        e = [r for r in rows if r["file"] == "fluorescein_embryo" and r["method"] == m]
        if e:
            ax[0].plot([e[0]["photons_px_mean"]], [e[0]["bias_pct"]], "*", ms=12, color=f"C{j}")
            ax[1].plot([e[0]["photons_px_mean"]], [e[0]["mare_pct"]], "*", ms=12, color=f"C{j}")
            ax[2].plot([e[0]["photons_px_mean"]], [e[0]["spread_rel_std_pct"]], "*", ms=12, color=f"C{j}")
    ax[0].axhline(0, color="gray", ls=":")
    for a, yl in zip(ax, ("Bias of the median (%)", "Median |relative error| (%)", "Spatial spread (%)")):
        a.set_xscale("log")
        a.set_xlabel("Photons per pixel")
        a.set_ylabel(yl)
        a.grid(alpha=0.3, which="both")
    ax[1].set_yscale("log")
    ax[2].set_yscale("log")
    ax[0].legend(fontsize=8)
    fig.suptitle("FLUTE fluorescein: lines Fluorescein_hMSC thinned (reference 3.61 ns), stars Fluorescein_Embryo "
                 "(about 20 photons, reference 4.0 ns)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def plot_cells(cell_maps, srows, out):
    cols = ["intensity"] + [m for m in ("MLE", "BinMLE5", "Phasor", "FLIMngo-FLUTE", "DiffFLIM")
                            if m in next(iter(cell_maps.values()))]
    if "halfwidth" in next(iter(cell_maps.values())):
        cols.append("halfwidth")
    fig, ax = plt.subplots(len(cell_maps), len(cols), figsize=(3.1 * len(cols), 3.0 * len(cell_maps)))
    for i, (k, mp) in enumerate(cell_maps.items()):
        for j, c in enumerate(cols):
            a = ax[i, j]
            if c == "intensity":
                im = a.imshow(mp[c], cmap="magma", vmax=np.percentile(mp[c], 99.5))
                a.set_title(f"{k}: photons", fontsize=8)
            elif c == "halfwidth":
                im = a.imshow(np.where(mp["mask"], mp[c], np.nan), cmap="cividis", vmin=0, vmax=1.5)
                a.set_title("DiffFLIM 90 % half-width (ns)", fontsize=8)
            else:
                im = a.imshow(np.where(mp["mask"], mp[c], np.nan), cmap="viridis", vmin=TAU_DISPLAY_CELLS[0],
                              vmax=TAU_DISPLAY_CELLS[1])
                sh = [r for r in srows if r["file"] == k and r["method"].startswith(c)]
                a.set_title(f"{c} (ns), split-half {sh[0]['splithalf_median_rel_diff_pct']:.1f} %" if sh else c, fontsize=8)
            a.axis("off")
            fig.colorbar(im, ax=a, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
