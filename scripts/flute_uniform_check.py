"""Uniform-lifetime check for DiffFLIM-FLUTE on simulated FLUTE images (supporting check, 24 November 2026).

Question: is the underestimate of DiffFLIM-FLUTE on the real fluorescein files (uniform 3.61 / 4.0 ns) also
present on simulated FLUTE images with a uniform lifetime (then it comes from the model or its training
prior), or only on the real data (then it comes from the gap between simulation and the instrument)?

Images: 30 scenes from the 30 test cells, FLUTE configuration of simulator v3 (flute_nadh_config: 56 bins of
0.223 ns, periodic 12.5 ns, FLUTE IRF library), scene seed 81000; every foreground pixel gets the same single
lifetime (1.0, 2.5, 3.61 or 4.0 ns). Two layouts: `cells` (the HPA cell mask and intensity of the scene) and
`field` (every pixel foreground with the same expected intensity, like a fluorescein drop; IRF of the scene).
Photon levels 10, 50, 500 per foreground pixel; noise seed 81000 + 1000 * lifetime index + level (+ 500 for
`field`). Methods: DiffFLIM (10 DDIM steps x 10 samples, sampler seed 1000 + image index) for each given
checkpoint, MLE and binned MLE 5 x 5 with the periodic forward model and the true IRF (fit_all).

Metrics per layout, lifetime, level and method: median of the per-image median lifetime, bias of that median
(%), median absolute relative error over foreground pixels (%), spatial spread (0.7413 x IQR / median inside
the foreground, median over images, %).

Optional (--flimngo-ckpt, added after STEP 4 of the night of 24 November 2026): the FLIMngo architecture
retrained on the FLUTE simulator, with its training input mask (otsu_adaptive on the counts, as on the real
data); pixels it predicts as 0 (outside its mask) are left out of its statistics. With --ckpts none only
FLIMngo-FLUTE and no classical method is run.

Options added later the same day (guidance follow-up):
  --guide-kappa K [--guide-window A B]  DiffFLIM additionally with Poisson data-consistency guidance
        (src/models/guidance.py; mono-exponential forward model with the image's IRF and background, as
        the MLE), rows "DiffFLIM <ckpt> guided"; several K values give several rows.
  --sim v3     simulator v3 with the default (FLIMngo) configuration, 256 bins of 0.0977 ns, for
        DiffFLIM v2 (scene seed and lifetimes unchanged).
  --ids val    scenes from the 40 validation cells instead of the 30 test cells (scene seed 82000, noise
        seeds shifted by the same amount); used to choose the guidance settings, so that the test-cell
        version stays an evaluation.

Writes results/tables/flute_uniform_check.csv (or --out).

Usage: python scripts/flute_uniform_check.py [--ckpts diffflim_flute diffflim_flute_100k]
       python scripts/flute_uniform_check.py --ckpts none --flimngo-ckpt checkpoints/flimngo_flute.pt            --out results/tables/flute_uniform_check_flimngo.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.common import fit_all  # noqa: E402
from src.data import simulate_v3 as v3  # noqa: E402
from src.models.diffflim import load_model  # noqa: E402
from src.paths import CHECKPOINTS, TABLES  # noqa: E402

SEED = 81000
TAUS = (1.0, 2.5, 3.61, 4.0)
LEVELS = (10, 50, 500)
N_SCENES = 30


def build(cfg, layout, tau, ti, lev, scenes, seed=SEED):
    out = []
    for s in scenes:
        u = dict(s)
        if layout == "field":
            m = np.ones_like(s["mask"], dtype=bool)
            u["mask"] = m
            u["rel_intensity"] = np.ones(m.shape)
        m = u["mask"]
        u["curves"] = v3.decay_curves_v3(np.full(int(m.sum()), tau), s["irf"], cfg)
        u["tau_gt"] = np.where(m, tau, 0.0)
        u["tau_int"] = u["tau_gt"]
        u["n_components"] = m.astype(np.uint8)
        out.append(u)
    rng = np.random.default_rng(seed + 1000 * ti + lev + (500 if layout == "field" else 0))
    return [v3.render_scene(u, cfg, lev, rng) for u in out]


VAL_SEED = 82000


def make_scenes(cfg, ids_kind="test", n=N_SCENES):
    """Scenes of the uniform check: one per cell of the list (round robin), and the seed used."""
    ids = v3.test_ids() if ids_kind == "test" else v3.val_ids()
    seed = SEED if ids_kind == "test" else VAL_SEED
    rng = np.random.default_rng(seed)
    return [v3.make_scene(rng, cfg, ids, fixed_id=ids[i % len(ids)]) for i in range(n)], seed


def make_guidance(d, meta, kappa, window):
    from src.baselines.common import forward_setup
    from src.models.guidance import PoissonGuidance
    fn = forward_setup(meta, d["irf"])[1]["decay_fn"]
    return PoissonGuidance(fn, float(d["bg_per_bin"]), kappa, start_frac=window[0], stop_frac=window[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=["flute", "v3"], default="flute")
    ap.add_argument("--ids", choices=["test", "val"], default="test")
    ap.add_argument("--n-scenes", type=int, default=N_SCENES)
    ap.add_argument("--guide-kappa", type=float, nargs="*", default=[])
    ap.add_argument("--guide-window", type=float, nargs=2, default=(0.0, 1.0))
    ap.add_argument("--no-classical", action="store_true")
    ap.add_argument("--ckpts", nargs="+", default=["diffflim_flute", "diffflim_flute_100k"])
    ap.add_argument("--flimngo-ckpt", default=None)
    ap.add_argument("--out", default=str(TABLES / "flute_uniform_check.csv"))
    args = ap.parse_args()
    if args.ckpts == ["none"]:
        args.ckpts = []
    flim = None
    if args.flimngo_ckpt:
        import torch
        sys.path.insert(0, str(ROOT / "scripts"))
        from eval_real_flute import load_flimngo_flute
        from src.evaluation.calibration import foreground_mask
        from train_flimngo_flute import flute_predict
        flim = load_flimngo_flute(args.flimngo_ckpt)
    cfg = v3.flute_nadh_config() if args.sim == "flute" else v3.SimConfigV3()
    meta = v3._meta(cfg)
    scenes, seed = make_scenes(cfg, args.ids, args.n_scenes)
    models = {c: load_model(CHECKPOINTS / f"{c}.pt")[0] for c in args.ckpts}
    rows = []
    for layout in ("field", "cells"):
        for ti, tau in enumerate(TAUS):
            for lev in LEVELS:
                imgs = build(cfg, layout, tau, ti, lev, scenes, seed)
                est = {f"DiffFLIM {c}": [] for c in models}
                for c in models:
                    for k in args.guide_kappa:
                        est[f"DiffFLIM {c} guided k{k:g} w{args.guide_window[0]:g}-{args.guide_window[1]:g}"] = []
                classical = bool(models) and not args.no_classical
                if classical:
                    est.update({"MLE": [], "BinMLE5": []})
                if flim is not None:
                    est["FLIMngo-FLUTE"] = []
                for i, d in enumerate(imgs):
                    for c, mdl in models.items():
                        s = mdl.sample(d["counts"], n_samples=10, steps=10, seed=1000 + i)
                        est[f"DiffFLIM {c}"].append(s[:, 0].mean(0))
                        for k in args.guide_kappa:
                            s = mdl.sample(d["counts"], n_samples=10, steps=10, seed=1000 + i,
                                           guidance=make_guidance(d, meta, k, args.guide_window))
                            est[f"DiffFLIM {c} guided k{k:g} w{args.guide_window[0]:g}-{args.guide_window[1]:g}"].append(
                                s[:, 0].mean(0))
                    if classical:
                        base = fit_all(d["counts"], float(d["bg_per_bin"]), meta, d["irf"], methods=("MLE", "BinMLE5"))
                        est["MLE"].append(base["MLE"][0]); est["BinMLE5"].append(base["BinMLE5"][0])
                    if flim is not None:
                        m_in = foreground_mask(d["counts"].sum(0), "otsu_adaptive")
                        with torch.no_grad():
                            y = flute_predict(flim, torch.from_numpy(d["counts"][None].astype(np.float32)).cuda(),
                                              torch.from_numpy(m_in[None]).cuda())
                        est["FLIMngo-FLUTE"].append(y[0].cpu().numpy())
                for meth, maps in est.items():
                    meds, errs, spreads = [], [], []
                    for d, x in zip(imgs, maps):
                        v = np.asarray(x, dtype=np.float64)[d["mask"]]
                        v = v[np.isfinite(v) & (v > 0)]
                        med = np.median(v)
                        q1, q3 = np.percentile(v, [25, 75])
                        meds.append(med); spreads.append(100 * 0.7413 * (q3 - q1) / med)
                        errs.append(np.abs(v - tau) / tau)
                    med = float(np.median(meds))
                    row = {"sim": args.sim, "ids": args.ids, "layout": layout, "tau_true_ns": tau, "photons": lev, "method": meth, "n_images": len(imgs),
                           "median_tau_ns": med, "bias_pct": 100 * (med / tau - 1),
                           "mare_pct": 100 * float(np.median(np.concatenate(errs))),
                           "spread_pct": float(np.median(spreads))}
                    rows.append(row)
                    print(f"{layout:5s} tau {tau:4.2f} p{lev:<4d} {meth:28s} median {med:5.3f} bias {row['bias_pct']:+6.1f} % "
                          f"MARE {row['mare_pct']:5.1f} % spread {row['spread_pct']:5.1f} %", flush=True)
    out = Path(args.out)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
