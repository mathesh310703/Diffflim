"""Simulated FLUTE test set (data/raw/own/sim_flute_test, 40 images per level), split by scene type.

The FLUTE simulator (flute_nadh_config) mixes two kinds of scenes: mono-exponential scenes (one lifetime per
pixel) and NAD(P)H-like scenes (two components per pixel). The truth `tau_gt` is the amplitude-weighted
lifetime; a mono-exponential fit (MLE, binned MLE, phasor) follows rather the intensity-weighted lifetime
`tau_int_gt` in two-component pixels. This script reports, per photon level, scene type and method, the
median relative error and the median absolute relative error on the true foreground against both
definitions, and for mono-exponential scenes the bias in bins of the true lifetime (where the two
definitions coincide).

DiffFLIM: the cached test predictions of eval_diffflim.py (data/processed/preds_<tag>_s10_n10_test.npz,
10 DDIM steps x 10 samples); classical methods: src.baselines.common.fit_all with the periodic forward
model (identical to eval_diffflim.py); optional FLIMngo-FLUTE (--flimngo-ckpt), with its otsu_adaptive input
mask (true-foreground pixels outside the mask count with the output 0, as in eval_flimngo_flute.py).

Writes results/tables/flute_sim_by_scene.csv. Supporting check, made on 24 November 2026.

Usage: python scripts/flute_sim_by_scene.py [--tags flute flute_100k] [--flimngo-ckpt checkpoints/flimngo_flute.pt]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.baselines.common import METHODS, fit_all  # noqa: E402
from src.data import simulate_v3 as v3  # noqa: E402
from src.data.simulate import load_dataset  # noqa: E402
from src.paths import TABLES  # noqa: E402

TAU_BINS = [(0.2, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 3.5), (3.5, 4.2), (4.2, 5.2)]


def stats(est, truth):
    r = (est - truth) / truth
    r = r[np.isfinite(r)]
    return 100 * float(np.median(r)), 100 * float(np.median(np.abs(r))), int(r.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["flute", "flute_100k"])
    ap.add_argument("--flimngo-ckpt", default=None)
    ap.add_argument("--out", default=str(TABLES / "flute_sim_by_scene.csv"))
    args = ap.parse_args()
    flim = None
    if args.flimngo_ckpt:
        import torch
        from eval_real_flute import load_flimngo_flute
        from src.evaluation.calibration import foreground_mask
        from train_flimngo_flute import flute_predict
        flim = load_flimngo_flute(args.flimngo_ckpt)
    caches = {t: np.load(ROOT / f"data/processed/preds_{t}_s10_n10_test.npz") for t in args.tags}
    rows = []
    files = sorted(v3.flute_set_dir("test").glob("test_p*.npz"), key=lambda p: int(p.stem.split("_p")[1]))
    for li, f in enumerate(files):
        d = load_dataset(f)
        level = int(round(float(d["photons"][0])))
        n = len(d["counts"])
        mask = d["mask"].astype(bool)
        mono = np.array([(d["n_components"][i][mask[i]] == 1).all() for i in range(n)])
        est = {}
        for t, c in caches.items():
            sl = slice(li * n, (li + 1) * n)
            assert np.allclose(c["tau_gt"][sl], d["tau_gt"], atol=1e-4), "cache order differs from test set"
            est[f"DiffFLIM {t}"] = c["tau_mean"][sl]
        base = {m: [] for m in METHODS}
        for i in range(n):
            out = fit_all(d["counts"][i], float(d["bg_per_bin"][i]), d["metadata"], d["irf"][i])
            for m, (tau, _) in out.items():
                base[m].append(tau)
        est.update({m: np.stack(v) for m, v in base.items()})
        if flim is not None:
            m_in = np.stack([foreground_mask(x.sum(0), "otsu_adaptive") for x in d["counts"]])
            c = d["counts"].astype(np.float32)
            with torch.no_grad():
                est["FLIMngo-FLUTE"] = np.concatenate([
                    flute_predict(flim, torch.from_numpy(c[i:i + 16]).cuda(), torch.from_numpy(m_in[i:i + 16]).cuda()).cpu().numpy()
                    for i in range(0, n, 16)])
        for scene, sel in (("mono", mono), ("nadh", ~mono)):
            fg = mask[sel]
            t_amp, t_int = d["tau_gt"][sel][fg], d["tau_int_gt"][sel][fg]
            for meth, e in est.items():
                x = e[sel][fg].astype(np.float64)
                x = np.where(np.isfinite(x) & (x > 0), x, np.nan) if not meth.startswith("FLIMngo") else x
                b, a, npx = stats(x, t_amp)
                bi, ai, _ = stats(x, t_int)
                row = {"photons": level, "scene": scene, "method": meth, "n_images": int(sel.sum()), "n_pixels": npx,
                       "bias_amp_pct": b, "mare_amp_pct": a, "bias_int_pct": bi, "mare_int_pct": ai}
                if scene == "mono":
                    for lo, hi in TAU_BINS:
                        s = (t_amp >= lo) & (t_amp < hi)
                        row[f"bias_tau_{lo}_{hi}_pct"] = stats(x[s], t_amp[s])[0] if s.sum() > 50 else ""
                rows.append(row)
                print(f"p{level} {scene:4s} {meth:20s} bias {b:+6.1f} MARE {a:5.1f} | vs tau_int bias {bi:+6.1f} MARE {ai:5.1f}")
    keys = list(rows[0])
    out = Path(args.out)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r.get(k, ''):.4g}" if isinstance(r.get(k, ""), float) else r.get(k, "")) for k in keys})
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
