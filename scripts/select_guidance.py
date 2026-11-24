"""Choose the settings of the Poisson data-consistency guidance (src/models/guidance.py) on simulated
VALIDATION data only (24 November 2026). Real data and test sets are not used here.

Data
  (1) the paired validation set (FLUTE: data/raw/own/sim_flute_val, 40 validation cells; v3:
      data/raw/own/sim_v3_val), the first --n-val images per photon level;
  (2) a uniform-lifetime check built from the validation cells (scripts/flute_uniform_check.py with
      ids "val": scene seed 82000, layouts `field` and `cells`, tau 1.0 / 2.5 / 3.61 / 4.0 ns, 10 / 50 /
      500 photons, the first --n-scenes scenes).
Grid: no guidance, and prior precision kappa in --kappas times guidance windows (fraction of the DDIM
steps from the start, --windows; e.g. 0.5 = the first 5 of 10 steps). 10 DDIM steps x 10 samples,
sampler seed 1000 + image index (as everywhere).

Selection rule (fixed before the run): score = mean over the validation photon levels of the median
absolute relative lifetime error on the foreground (against the amplitude-weighted truth tau_gt), plus
the mean over the uniform conditions (2 layouts x 4 lifetimes x 3 levels) of |bias of the median|, both
in %. The setting with the lowest score is chosen. The uncalibrated 90 % coverage on the validation set
is reported but not used (the calibration is redone for the chosen setting).

Writes results/tables/guidance_selection_<tag>.csv (one row per setting: score, its two parts, and per
level and scene type the validation error and bias, uniform bias per lifetime) and prints the choice.

Usage:
  python scripts/select_guidance.py --ckpt checkpoints/diffflim_flute.pt --sim flute --tag flute
  python scripts/select_guidance.py --ckpt checkpoints/diffflim_v2.pt --sim v3 --tag v2
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flute_uniform_check import LEVELS as UNI_LEVELS, TAUS, build, make_guidance, make_scenes  # noqa: E402
from src.data import simulate_v3 as v3  # noqa: E402
from src.data.simulate import load_dataset  # noqa: E402
from src.models.diffflim import load_model  # noqa: E402
from src.paths import TABLES  # noqa: E402


def val_images(sim, n_val, folder=None):
    if folder is None:
        folder = v3.flute_set_dir("val") if sim == "flute" else v3.val_set_dir()
    folder = Path(folder)
    files = sorted(folder.glob("val_p*.npz"), key=lambda p: int(p.stem.split("_p")[1]))
    out = []
    for f in files:
        d = load_dataset(f)
        n = min(n_val, len(d["counts"]))
        for i in range(n):
            out.append({"level": int(round(float(d["photons"][i]))), "index": i, "counts": d["counts"][i],
                        "irf": d["irf"][i], "bg_per_bin": float(d["bg_per_bin"][i]), "tau_gt": d["tau_gt"][i],
                        "mask": d["mask"][i].astype(bool),
                        "mono": bool((d["n_components"][i][d["mask"][i].astype(bool)] == 1).all()),
                        "meta": d["metadata"]})
    return out


def run(model, d, meta, setting, i):
    g = None if setting is None else make_guidance(d, meta, setting[0], (0.0, setting[1]))
    s = model.sample(d["counts"], n_samples=10, steps=10, seed=1000 + i, guidance=g)
    return s[:, 0].mean(0), s[:, 0].std(0, ddof=1)


def name(setting):
    return "none" if setting is None else f"k{setting[0]:g}_w0-{setting[1]:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sim", choices=["flute", "v3"], default="flute")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--kappas", type=float, nargs="+", default=[0, 30, 100, 300, 1000])
    ap.add_argument("--windows", type=float, nargs="+", default=[0.3, 0.5, 0.8, 1.0])
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--n-scenes", type=int, default=8)
    ap.add_argument("--val-dir", default=None, help="validation folder (default: the set of --sim), e.g. "
                    "data/raw/own/sim_flute_v2_val for the FLUTE v2 prior")
    args = ap.parse_args()
    t0 = time.time()
    model = load_model(args.ckpt)[0]
    cfg = v3.flute_nadh_config() if args.sim == "flute" else v3.SimConfigV3()
    meta = v3._meta(cfg)
    vals = val_images(args.sim, args.n_val, args.val_dir)
    scenes, seed = make_scenes(cfg, "val", args.n_scenes)
    uni = []
    for layout in ("field", "cells"):
        for ti, tau in enumerate(TAUS):
            for lev in UNI_LEVELS:
                for d in build(cfg, layout, tau, ti, lev, scenes, seed):
                    uni.append((layout, tau, lev, d))
    levels = sorted({v["level"] for v in vals})
    print(f"{len(vals)} validation images, {len(uni)} uniform images (validation cells) [{time.time() - t0:.0f} s]")

    settings = [None] + [(k, w) for k in args.kappas for w in args.windows]
    rows = []
    for st in settings:
        row = {"setting": name(st), "kappa": "" if st is None else st[0], "window_stop": "" if st is None else st[1]}
        err, bias, hit = {}, {}, {}
        for v in vals:
            mean, std = run(model, v, v["meta"], st, v["index"])
            m = v["mask"]
            r = (mean[m] - v["tau_gt"][m]) / v["tau_gt"][m]
            for key in ((v["level"], "all"), (v["level"], "mono" if v["mono"] else "nadh")):
                err.setdefault(key, []).append(r)
            hit.setdefault(v["level"], []).append(np.abs(mean[m] - v["tau_gt"][m]) <= 1.645 * std[m])
        for (lev, sc), rr in sorted(err.items()):
            r = np.concatenate(rr)
            row[f"val_mare_{sc}_p{lev}"] = 100 * float(np.median(np.abs(r)))
            row[f"val_bias_{sc}_p{lev}"] = 100 * float(np.median(r))
        for lev in levels:
            row[f"val_cov90_uncal_p{lev}"] = float(np.mean(np.concatenate(hit[lev])))
        ub = {}
        for layout, tau, lev, d in uni:
            mean, _ = run(model, d, meta, st, len(ub.get((layout, tau, lev), [])))
            v_ = mean[d["mask"]]
            ub.setdefault((layout, tau, lev), []).append(float(np.median(v_[np.isfinite(v_)])))
        for (layout, tau, lev), meds in ub.items():
            row[f"uni_bias_{layout}_t{tau:g}_p{lev}"] = 100 * (float(np.median(meds)) / tau - 1)
        row["val_mare_mean"] = float(np.mean([row[f"val_mare_all_p{lev}"] for lev in levels]))
        row["uni_absbias_mean"] = float(np.mean([abs(v_) for k, v_ in row.items() if k.startswith("uni_bias_")]))
        row["score"] = row["val_mare_mean"] + row["uni_absbias_mean"]
        rows.append(row)
        print(f"{row['setting']:14s} score {row['score']:6.2f} = val MARE {row['val_mare_mean']:5.2f} + uniform |bias| "
              f"{row['uni_absbias_mean']:5.2f}; val MARE by level "
              + " ".join(f"{row[f'val_mare_all_p{lev}']:.1f}" for lev in levels)
              + f"; cov90 uncal " + " ".join(f"{row[f'val_cov90_uncal_p{lev}']:.2f}" for lev in levels)
              + f" [{time.time() - t0:.0f} s]", flush=True)
    best = min(rows, key=lambda r: r["score"])
    for r in rows:
        r["chosen"] = int(r is best)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = TABLES / f"guidance_selection_{args.tag}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4g}" if isinstance(r.get(k), float) else r.get(k, "")) for k in keys})
    print(f"Chosen: {best['setting']} (score {best['score']:.2f}; no guidance: {rows[0]['score']:.2f}). Saved {out}")


if __name__ == "__main__":
    main()
