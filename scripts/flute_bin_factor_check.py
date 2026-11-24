"""How much lifetime information is lost when the 56 FLUTE bins are summed in groups (BIN_FACTOR)?

DiffFLIM-FLUTE (checkpoints/diffflim_flute.pt) sums pairs of bins before the condition encoder
(bin_factor 2: 56 bins of 0.223 ns -> 28 bins of 0.446 ns). Changing the grouping needs retraining, so
this supporting check (24 November 2026) estimates the effect without a network, in two ways:

  (a) Cramer-Rao bound. For a mono-exponential decay with the periodic FLUTE forward model of simulator
      v3 (flute_nadh_config, every IRF of the FLUTE library at the median peak position), the Fisher
      information per photon about log(tau) is sum_k (d p_k / d log tau)^2 / p_k, with p the binned
      decay (background 2 % of the signal, spread over the bins). The bound on the relative standard
      deviation for N photons is 1 / sqrt(N * information). Reported relative to no grouping (factor 1),
      for factors 1, 2, 4, 7, 8, 14 and lifetimes 0.4 to 6 ns. For comparison the DiffFLIM v2 time axis
      (256 bins of 0.0977 ns, factor 8, not periodic, FLIMngo IRFs) is reported in absolute terms.
  (b) MLE on equally binned data: the uniform-lifetime images of scripts/flute_uniform_check.py (layout
      `field`, 1.0 / 2.5 / 3.61 / 4.0 ns, 10 / 50 / 500 photons, the first 10 scenes, same seeds) are
      summed in groups of the factor and fitted with the per-pixel Poisson MLE and the binned MLE 5 x 5
      with the equally binned forward model. Bias, median absolute relative error and the robust spread
      of the relative error (0.7413 x IQR) over the foreground pixels.

Writes results/tables/flute_bin_factor_crb.csv and results/tables/flute_bin_factor_mle.csv.
Usage: python scripts/flute_bin_factor_check.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flute_uniform_check import SEED, build  # noqa: E402
from src.baselines.binned import bin_spatial  # noqa: E402
from src.baselines.mle import fit_mle  # noqa: E402
from src.data import simulate_v3 as v3  # noqa: E402
from src.paths import TABLES  # noqa: E402

FACTORS = (1, 2, 4, 7, 8, 14)
CRB_TAUS = (0.4, 1.0, 2.0, 2.5, 3.61, 4.0, 5.0, 6.0)
MLE_FACTORS = (1, 2, 4, 7)
TAUS = (1.0, 2.5, 3.61, 4.0)
LEVELS = (10, 50, 500)
N_SCENES = 10
BG_FRACTION = 0.02


def group(x, f, axis=0):
    """Sum groups of f consecutive bins along axis."""
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, 0)
    out = x.reshape(x.shape[0] // f, f, *x.shape[1:]).sum(1)
    return np.moveaxis(out, 0, axis)


def info_per_photon(tau, irf, cfg, f, h=1e-4):
    """Fisher information about log(tau) per signal photon, binned by f, with 2 % background."""
    lt = np.log(tau)
    p = group(v3.decay_curves_v3(np.array([tau]), irf, cfg)[:, 0], f)
    pp = group(v3.decay_curves_v3(np.array([np.exp(lt + h)]), irf, cfg)[:, 0], f)
    pm = group(v3.decay_curves_v3(np.array([np.exp(lt - h)]), irf, cfg)[:, 0], f)
    lam = p + BG_FRACTION / len(p)
    dp = (pp - pm) / (2 * h)
    return float((dp ** 2 / lam).sum())


def crb_rows():
    rows = []
    cfg = v3.flute_nadh_config()
    lo, hi = (int(round(x / cfg.bin_width_ns)) for x in cfg.irf_peak_range_ns)
    irfs = [np.roll(x["irf"], (lo + hi) // 2) for x in v3.irf_library(cfg)]
    for tau in CRB_TAUS:
        ref = np.median([info_per_photon(tau, irf, cfg, 1) for irf in irfs])
        for f in FACTORS:
            info = np.median([info_per_photon(tau, irf, cfg, f) for irf in irfs])
            rows.append({"axis": "FLUTE 56 x 0.223 ns, periodic", "tau_ns": tau, "factor": f,
                         "n_bins_after": cfg.n_bins // f, "bin_width_ns": cfg.bin_width_ns * f,
                         "crb_rel_std_pct_100ph": 100 / np.sqrt(100 * info), "std_ratio_vs_factor1": np.sqrt(ref / info)})
    cfg2 = v3.SimConfigV3()
    irfs2 = [x["irf"] for x in v3.irf_library(cfg2)]
    for tau in CRB_TAUS:
        ref = np.median([info_per_photon(tau, irf, cfg2, 1) for irf in irfs2])
        info = np.median([info_per_photon(tau, irf, cfg2, 8) for irf in irfs2])
        rows.append({"axis": "v3 256 x 0.0977 ns (DiffFLIM v2)", "tau_ns": tau, "factor": 8, "n_bins_after": 32,
                     "bin_width_ns": cfg2.bin_width_ns * 8, "crb_rel_std_pct_100ph": 100 / np.sqrt(100 * info),
                     "std_ratio_vs_factor1": np.sqrt(ref / info)})
    return rows


def mle_rows():
    cfg = v3.flute_nadh_config()
    ids = v3.test_ids()
    rng = np.random.default_rng(SEED)
    scenes = [v3.make_scene(rng, cfg, ids, fixed_id=ids[i % len(ids)]) for i in range(30)][:N_SCENES]
    rows = []
    for ti, tau in enumerate(TAUS):
        for lev in LEVELS:
            imgs = build(cfg, "field", tau, ti, lev, scenes)
            for f in MLE_FACTORS:
                for meth in ("MLE", "BinMLE5"):
                    rel = []
                    for d in imgs:
                        irf = d["irf"]
                        fn = lambda t, irf=irf: group(v3.decay_curves_v3(t, irf, cfg), f)  # noqa: E731
                        c = group(d["counts"], f)
                        bg = float(d["bg_per_bin"]) * f
                        if meth == "BinMLE5":
                            c, bg = bin_spatial(c, 5), bg * 25
                        t_hat, _ = fit_mle(c, cfg, bg, tau_bounds=(0.05, 15.0), decay_fn=fn)
                        v = t_hat[d["mask"]]
                        rel.append((v[np.isfinite(v)] - tau) / tau)
                    r = np.concatenate(rel)
                    q1, q3 = np.percentile(r, [25, 75])
                    row = {"tau_ns": tau, "photons": lev, "factor": f, "method": meth,
                           "bias_pct": 100 * float(np.median(r)), "mare_pct": 100 * float(np.median(np.abs(r))),
                           "spread_pct": 100 * 0.7413 * float(q3 - q1)}
                    rows.append(row)
                    print(f"tau {tau:4.2f} p{lev:<4d} f{f} {meth:8s} bias {row['bias_pct']:+6.2f} % "
                          f"MARE {row['mare_pct']:5.2f} % spread {row['spread_pct']:5.2f} %", flush=True)
    return rows


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved {path}")


def main():
    rows = crb_rows()
    for r in rows:
        print(f"{r['axis'][:12]:12s} tau {r['tau_ns']:4.2f} f{r['factor']:<2d} ({r['n_bins_after']:2d} bins of "
              f"{r['bin_width_ns']:.3f} ns): CRB at 100 photons {r['crb_rel_std_pct_100ph']:5.2f} %, "
              f"ratio {r['std_ratio_vs_factor1']:.3f}")
    write(TABLES / "flute_bin_factor_crb.csv", rows)
    write(TABLES / "flute_bin_factor_mle.csv", mle_rows())


if __name__ == "__main__":
    main()
