"""Fluorescein in the FLUTE data as a sample with a known lifetime (4.0 ns everywhere).

Part b (full photon counts):
  1. IRF, non-circular: for file A, the summed decay of all pixels is fitted with the periodic FLUTE
     model (src/data/flute.py) with tau FIXED at 4.0 ns and a wrapped Gaussian IRF (centre t0,
     width sigma), a constant background b and the intensity free (Poisson likelihood). The IRF
     obtained from A is then used for the OTHER file B: its width is kept, and its centre is moved by
     the difference of the model-free half-maximum rise times of the two summed decays. The lifetime
     of B is never assumed. Done in both directions (Embryo -> hMSC and hMSC -> Embryo).
  2. On B: global fit of the summed decay (tau, b, intensity free, IRF from A) and per-pixel fits
     (MLE, binned MLE 5 x 5, phasor with IRF correction) with the background per pixel b / n_pixels.
  Checks (reported, not used): the circular IRF of each file from itself, and a free global fit of
  tau, t0, sigma and b on each file.

Part c (thinning): Fluorescein_hMSC (about 720 photons per pixel) is thinned binomially to a mean
of 10, 20, 50, 100 and 500 photons per pixel, Fluorescein_Embryo (about 20) to 10, with fixed seeds
(5000 + level); background scaled by the same factor; same three methods with the IRF from the
other file. Truth: 4.0 ns in every pixel.

Metrics per method and condition: median lifetime, bias (median relative error), median absolute
relative error, flatness (robust spatial standard deviation 0.7413 x IQR of the lifetime map,
relative to its median), fraction of pixels without a finite estimate, and the Spearman correlation
between lifetime and intensity. Binned maps exclude a 2-pixel border.

Writes results/tables/real_flute_irf.csv, results/tables/real_flute_fluorescein.csv,
results/figures/real_flute_fluorescein_fit.png, results/figures/real_flute_fluorescein_maps.png and
results/figures/real_flute_thinning.png.

Usage: python scripts/flute_fluorescein.py
"""
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.binned import fit_binned_mle  # noqa: E402
from src.baselines.mle import fit_mle  # noqa: E402
from src.baselines.phasor import fit_phasor  # noqa: E402
from src.data.flute import (FLUORESCEIN_TAU_NS, FluteConfig, PeriodicDecayModel, binomial_thin,  # noqa: E402
                            bin_fine, gaussian_irf_fine, load_flute, periodic_decays, rise_time)
from src.paths import FIGURES, TABLES  # noqa: E402

TAU = FLUORESCEIN_TAU_NS
FILES = {"embryo": "fluorescein_embryo", "hmsc": "fluorescein_hmsc"}
OTHER = {"embryo": "hmsc", "hmsc": "embryo"}
TAU_BOUNDS = (0.1, 20.0)
THIN = {"hmsc": (10, 20, 50, 100, 500), "embryo": (10,)}
THIN_SEED = 5000
METHODS = ("MLE", "BinMLE5", "Phasor")


def model_sum(params, cfg, tau=None):
    """Expected summed decay for params = (t0, log sigma, log b, log I[, log tau])."""
    t0, ls, lb, li = params[:4]
    tau_ = np.exp(params[4]) if tau is None else tau
    d = periodic_decays([tau_], gaussian_irf_fine(cfg, t0, np.exp(ls)), cfg)[0]
    return np.exp(li) * d + np.exp(lb)


def poisson_nll(lam, c):
    return float(np.sum(lam - c * np.log(np.maximum(lam, 1e-12))))


def fit_global(dec, cfg, tau=None, irf=None, start=None):
    """Poisson fit of a summed decay. tau fixed (IRF free) or irf fixed (tau free) or both free."""
    tot = dec.sum()
    if irf is not None:                                      # IRF given: fit log tau, log b, log I
        def f(p):
            d = periodic_decays([np.exp(p[0])], irf, cfg)[0]
            return poisson_nll(np.exp(p[2]) * d + np.exp(p[1]), dec)
        r = minimize(f, [np.log(4.0), np.log(max(dec.min() * 0.1, 1.0)), np.log(tot)], method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
        return {"tau": np.exp(r.x[0]), "b": np.exp(r.x[1]), "I": np.exp(r.x[2]), "nll": r.fun}
    t_pk = np.argmax(dec) * cfg.bin_width_ns
    p0 = start if start is not None else [t_pk - 0.1, np.log(0.12), np.log(max(dec.min() * 0.1, 1.0)), np.log(tot)]
    if tau is None:
        p0 = list(p0) + [np.log(4.0)]
    best = None
    for dt0 in (-0.2, 0.0, 0.2):                             # a few starts for the IRF centre
        q = list(p0)
        q[0] += dt0
        r = minimize(lambda p: poisson_nll(model_sum(p, cfg, tau), dec), q, method="Nelder-Mead",
                     options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 20000})
        if best is None or r.fun < best.fun:
            best = r
    p = best.x
    out = {"t0": p[0], "sigma": np.exp(p[1]), "b": np.exp(p[2]), "I": np.exp(p[3]), "nll": best.fun}
    out["tau"] = np.exp(p[4]) if tau is None else tau
    return out


def map_stats(tau, inten, border=0):
    if border:
        tau, inten = tau[border:-border, border:-border], inten[border:-border, border:-border]
    ok = np.isfinite(tau)
    v = tau[ok]
    med = float(np.median(v))
    q1, q3 = np.percentile(v, [25, 75])
    rho = spearmanr(v, inten[ok]).correlation if ok.sum() > 10 else np.nan
    return {"median_tau_ns": med, "bias_pct": 100 * float(np.median((v - TAU) / TAU)),
            "mare_pct": 100 * float(np.median(np.abs(v - TAU) / TAU)),
            "flatness_rel_std_pct": 100 * 0.7413 * float(q3 - q1) / med,
            "invalid_pct": 100 * float(1 - ok.mean()), "spearman_tau_intensity": float(rho)}


def run_methods(counts, irf, cfg, b_px):
    model = PeriodicDecayModel(irf, cfg, tau_bounds=(0.05, 25.0))
    kw = {"tau_bounds": TAU_BOUNDS, "decay_fn": model}
    out = {}
    out["MLE"] = fit_mle(counts, cfg, b_px, **kw)
    out["BinMLE5"] = fit_binned_mle(counts, cfg, b_px, k=5, **kw)
    irf_binned = bin_fine(irf, cfg)
    tau_p, int_p, _, _ = fit_phasor(counts, cfg, b_px, irf=irf_binned, point_sampled=False)
    out["Phasor"] = (tau_p, int_p)
    return out


def main():
    t_start = time.time()
    cfg = FluteConfig()
    data = {k: load_flute(v) for k, v in FILES.items()}
    dec = {k: data[k].sum((1, 2)).astype(float) for k in data}
    npx = {k: data[k].shape[1] * data[k].shape[2] for k in data}
    rise = {k: rise_time(dec[k], cfg) for k in data}

    # ---- IRFs: tau fixed at 4 ns on each file (used for the OTHER file), and checks
    irf_rows, own, free = [], {}, {}
    for k in data:
        own[k] = fit_global(dec[k], cfg, tau=TAU)
        free[k] = fit_global(dec[k], cfg, tau=None)
        print(f"{k}: IRF (tau=4) t0={own[k]['t0']:.3f} sigma={own[k]['sigma']:.3f} b/bin/px={own[k]['b']/npx[k]:.4f}; "
              f"free fit tau={free[k]['tau']:.3f} t0={free[k]['t0']:.3f} sigma={free[k]['sigma']:.3f}; rise {rise[k]:.3f} ns")
    transfer = {}
    for b_key in data:                                       # B = file to analyse, A = IRF source
        a_key = OTHER[b_key]
        shift = rise[b_key] - rise[a_key]
        irf = gaussian_irf_fine(cfg, own[a_key]["t0"] + shift, own[a_key]["sigma"])
        g = fit_global(dec[b_key], cfg, irf=irf)
        transfer[b_key] = {"irf": irf, "b_px": g["b"] / npx[b_key], "global": g, "shift": shift}
        g_self = fit_global(dec[b_key], cfg, irf=gaussian_irf_fine(cfg, own[b_key]["t0"], own[b_key]["sigma"]))
        print(f"  {b_key} with IRF from {a_key}: shift {shift:+.3f} ns, global tau {g['tau']:.3f} ns "
              f"(own IRF, circular: {g_self['tau']:.3f}), b/bin/px {g['b']/npx[b_key]:.4f}")
        irf_rows.append({"file": FILES[b_key], "irf_source": FILES[a_key], "irf_t0_source_ns": own[a_key]["t0"],
                         "irf_sigma_ns": own[a_key]["sigma"], "irf_fwhm_ns": 2.3548 * own[a_key]["sigma"],
                         "rise_shift_ns": shift, "irf_t0_used_ns": own[a_key]["t0"] + shift,
                         "global_tau_ns": g["tau"], "global_bg_per_bin_per_px": g["b"] / npx[b_key],
                         "global_bg_fraction_pct": 100 * g["b"] * cfg.n_bins / dec[b_key].sum(),
                         "check_own_irf_tau_ns": g_self["tau"], "check_free_fit_tau_ns": free[b_key]["tau"],
                         "check_free_fit_sigma_ns": free[b_key]["sigma"],
                         "own_irf_t0_ns": own[b_key]["t0"], "own_irf_sigma_ns": own[b_key]["sigma"]})
    with open(TABLES / "real_flute_irf.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(irf_rows[0]))
        w.writeheader()
        for r in irf_rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})

    # ---- fit figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, k in enumerate(data):
        tr = transfer[k]
        lam = tr["global"]["I"] * periodic_decays([tr["global"]["tau"]], tr["irf"], cfg)[0] + tr["global"]["b"]
        ax[i].semilogy(cfg.time_centers, dec[k], "o", ms=3, label="measured (all pixels)")
        ax[i].semilogy(cfg.time_centers, lam, "-", label=f"fit, IRF from the other file: τ = {tr['global']['tau']:.2f} ns")
        irf_b = bin_fine(tr["irf"], cfg)
        ax[i].semilogy(cfg.time_centers, irf_b / irf_b.max() * dec[k].max(), ":", color="gray", label="transferred IRF (scaled)")
        ax[i].set_ylim(dec[k].min() * 0.3, dec[k].max() * 1.5)
        ax[i].set_title(f"{FILES[k]}")
        ax[i].set_xlabel("Time (ns)")
        ax[i].set_ylabel("Counts summed over all pixels")
        ax[i].legend(fontsize=8)
        ax[i].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "real_flute_fluorescein_fit.png", dpi=120)
    plt.close(fig)

    # ---- per-pixel methods: full photons and thinned
    rows, maps = [], {}
    for k in data:
        tr = transfer[k]
        mean_ph = data[k].sum(0).mean()
        conds = [("full", None)] + [(str(lv), lv) for lv in THIN[k]]
        for cname, lv in conds:
            if lv is None:
                counts, p = data[k], 1.0
            else:
                p = lv / mean_ph
                counts = binomial_thin(data[k], p, THIN_SEED + lv)
            res = run_methods(counts, tr["irf"], cfg, tr["b_px"] * p)
            for m in METHODS:
                tau_m, int_m = res[m]
                st = map_stats(tau_m, int_m, border=2 if m == "BinMLE5" else 0)
                rows.append({"file": FILES[k], "condition": cname, "photons_px_mean": float(counts.sum(0).mean()),
                             "method": m, **st})
                maps[(k, cname, m)] = tau_m
                print(f"{k:6s} {cname:>4s} {m:8s} median {st['median_tau_ns']:.3f} ns bias {st['bias_pct']:+.1f} % "
                      f"MARE {st['mare_pct']:.1f} % flat {st['flatness_rel_std_pct']:.1f} % rho {st['spearman_tau_intensity']:+.2f} "
                      f"[{time.time() - t_start:.0f} s]", flush=True)
    with open(TABLES / "real_flute_fluorescein.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})

    # ---- maps figure (full photons, both files)
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.6))
    for i, k in enumerate(data):
        for j, m in enumerate(METHODS):
            im = ax[i, j].imshow(maps[(k, "full", m)], vmin=3.0, vmax=5.0, cmap="viridis")
            st = [r for r in rows if r["file"] == FILES[k] and r["condition"] == "full" and r["method"] == m][0]
            ax[i, j].set_title(f"{FILES[k]}, {m}\nmedian {st['median_tau_ns']:.2f} ns, spread {st['flatness_rel_std_pct']:.1f} %",
                                fontsize=9)
            ax[i, j].axis("off")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Lifetime (ns)")
    fig.savefig(FIGURES / "real_flute_fluorescein_maps.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    plot_thinning(rows)
    print(f"done in {time.time() - t_start:.0f} s")


def plot_thinning(rows):
    """Thinning figure for Fluorescein_hMSC. Reference: the full-photon binned-MLE median of the same file
    (the file gives about 3.6 ns, not the stated 4.0 ns; see the project log of 2026-11-23)."""
    rows = [r for r in rows if r["file"] == FILES["hmsc"]]
    ref = [float(r["median_tau_ns"]) for r in rows if r["condition"] == "full" and r["method"] == "BinMLE5"][0]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for j, m in enumerate(METHODS):
        rr = sorted([r for r in rows if r["method"] == m], key=lambda r: float(r["photons_px_mean"]))
        x = [float(r["photons_px_mean"]) for r in rr]
        med = np.array([float(r["median_tau_ns"]) for r in rr])
        ax[0].plot(x, med, "o-", color=f"C{j}", label=m)
        ax[1].plot(x, 100 * (med - ref) / ref, "o-", color=f"C{j}", label=m)
        ax[2].plot(x, [float(r["flatness_rel_std_pct"]) for r in rr], "o-", color=f"C{j}", label=m)
    ax[0].axhline(TAU, color="k", ls="--", lw=1, label="stated 4.0 ns")
    ax[0].axhline(ref, color="gray", ls=":", lw=1.5, label=f"full-photon estimate {ref:.2f} ns")
    ax[1].axhline(0, color="gray", ls=":", lw=1.5)
    labels = ("Median lifetime (ns)", f"Median lifetime vs {ref:.2f} ns (%)", "Spatial spread (robust std, %)")
    for a, yl in zip(ax, labels):
        a.set_xscale("log")
        a.set_xlabel("Photons per pixel (mean; rightmost point: full data)")
        a.set_ylabel(yl)
        a.grid(alpha=0.3, which="both")
    ax[2].set_yscale("log")
    ax[0].legend(fontsize=8)
    fig.suptitle("Fluorescein_hMSC thinned binomially (uniform lifetime); IRF transferred from Fluorescein_Embryo",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "real_flute_thinning.png", dpi=120)
    plt.close(fig)
    print(f"Saved {FIGURES / 'real_flute_thinning.png'}")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        with open(TABLES / "real_flute_fluorescein.csv") as f:
            plot_thinning(list(csv.DictReader(f)))
    else:
        main()
