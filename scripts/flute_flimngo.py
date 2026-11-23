"""Pretrained FLIMngo on the FLUTE fluorescein images, with FLIMngo's own recipe for other time axes
(external/FLIMngo/demo_notebooks/predict_diff_time_dimensions.ipynb):
  1. normalise each pixel's decay to its maximum (utils.eval_utils.normalise),
  2. resize the time axis to 256 bins by linear interpolation (scipy.ndimage.zoom, order 1, as
     utils.eval_utils.resize_data_tdim),
  3. give the network the resized bin width, (56 - 1) x 0.223 ns / 255 = 0.0481 ns (resize_time);
     the network scales its output by bin_width / 0.09765625.
The 512 x 512 images are processed as 64 tiles of 64 x 64 pixels (the size used everywhere else in
this project); no mask (fluorescein fills the whole image). Conditions and thinning as in
scripts/flute_fluorescein.py (same p and seeds 5000 + level). Statistics as there (map_stats).
Diagnostic rows: the full-photon data shifted circularly so that the rise lies at 2 ns.

Writes results/tables/real_flute_flimngo.csv.

Usage: python scripts/flute_flimngo.py
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flute_fluorescein import FILES, THIN, THIN_SEED, map_stats  # noqa: E402
from src.baselines.flimngo_baseline import normalise, predict  # noqa: E402
from src.data.flute import FLUTE_BIN_NS, binomial_thin, load_flute  # noqa: E402
from src.paths import TABLES  # noqa: E402

TILE = 64


def flimngo_map(counts):
    t, h, w = counts.shape
    x = normalise(counts.astype(np.float32))
    x = zoom(x, (256 / t, 1, 1), order=1).astype(np.float32)
    bw = (t - 1) * FLUTE_BIN_NS / 255
    nh, nw = h // TILE, w // TILE
    tiles = x.reshape(256, nh, TILE, nw, TILE).transpose(1, 3, 0, 2, 4).reshape(nh * nw, 256, TILE, TILE)
    tau, _ = predict(tiles, mask=None, bin_width=bw, normalise_decays=True, batch=16)
    return tau.reshape(nh, nw, TILE, TILE).transpose(0, 2, 1, 3).reshape(h, w), bw


def main():
    t0 = time.time()
    rows = []
    for k, fname in FILES.items():
        data = load_flute(fname)
        mean_ph = data.sum(0).mean()
        for cname, lv in [("full", None)] + [(str(v), v) for v in THIN[k]]:
            counts = data if lv is None else binomial_thin(data, lv / mean_ph, THIN_SEED + lv)
            tau, bw = flimngo_map(counts)
            st = map_stats(tau, counts.sum(0).astype(float))
            rows.append({"file": fname, "condition": cname, "photons_px_mean": float(counts.sum(0).mean()),
                         "method": "FLIMngo pretrained", "bin_width_resized_ns": bw, **st})
            print(f"{k:6s} {cname:>4s} median {st['median_tau_ns']:.3f} ns spread {st['flatness_rel_std_pct']:.1f} % "
                  f"rho {st['spearman_tau_intensity']:+.2f} [{time.time() - t0:.0f} s]", flush=True)
    # Diagnostic (full photons only): the FLUTE data are periodic, so the decays can be shifted circularly
    # without changing their content; shift them so that the half-maximum rise lies at 2.0 ns, as in
    # FLIMngo's training data (IRF peak at 2 to 5 ns).
    from src.data.flute import FluteConfig, rise_time
    cfg = FluteConfig()
    for k, fname in FILES.items():
        data = load_flute(fname)
        shift = int(round((2.0 - rise_time(data.sum((1, 2)), cfg)) / FLUTE_BIN_NS))
        tau, bw = flimngo_map(np.roll(data, shift, axis=0))
        st = map_stats(tau, data.sum(0).astype(float))
        rows.append({"file": fname, "condition": f"full, rolled {shift:+d} bins (rise at 2 ns)",
                     "photons_px_mean": float(data.sum(0).mean()), "method": "FLIMngo pretrained",
                     "bin_width_resized_ns": bw, **st})
        print(f"{k:6s} rolled {shift:+d} median {st['median_tau_ns']:.3f} ns spread {st['flatness_rel_std_pct']:.1f} %", flush=True)
    out = TABLES / "real_flute_flimngo.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({kk: (f"{v:.4g}" if isinstance(v, float) else v) for kk, v in r.items()})
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
