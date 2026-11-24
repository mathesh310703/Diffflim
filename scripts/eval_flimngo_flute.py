"""The FLIMngo architecture retrained on the FLUTE simulator (checkpoints/flimngo_flute.pt) on the simulated
FLUTE test set (data/raw/own/sim_flute_test, 40 images per level, levels 10 to 500), and a combined table
with DiffFLIM-FLUTE and the classical methods from results/tables/diffflim_<tag>.csv.

FLIMngo-FLUTE: 56 bins resized linearly to 256, background mask computed from the counts
(otsu_adaptive, as in its training); pixels of the true foreground outside the computed mask count with
the network output 0 there (as for FLIMngo in training) and are reported as `invalid_frac`.
Metrics (src/evaluation/metrics.summarize): median absolute relative lifetime error and median relative
error (bias) over the true foreground pixels of all images of a level (subset all_fg).

Writes results/tables/flimngo_flute_test.csv and results/tables/flute_sim_comparison.csv.

Usage: python scripts/eval_flimngo_flute.py [--ckpt checkpoints/flimngo_flute.pt] [--diffflim-tag flute]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_real_flute import load_flimngo_flute  # noqa: E402
from src.data import simulate_v3 as v3  # noqa: E402
from src.data.simulate import load_dataset  # noqa: E402
from src.evaluation.calibration import foreground_mask  # noqa: E402
from src.evaluation.metrics import summarize  # noqa: E402
from src.paths import TABLES  # noqa: E402
from train_flimngo_flute import flute_predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/flimngo_flute.pt")
    ap.add_argument("--diffflim-tag", default="flute")
    args = ap.parse_args()
    net = load_flimngo_flute(args.ckpt)
    rows = []
    files = sorted(v3.flute_set_dir("test").glob("test_p*.npz"), key=lambda p: int(p.stem.split("_p")[1]))
    for f in files:
        d = load_dataset(f)
        c = d["counts"].astype(np.float32)
        m = np.stack([foreground_mask(x.sum(0), "otsu_adaptive") for x in d["counts"]])
        preds = []
        with torch.no_grad():
            for i in range(0, len(c), 16):
                preds.append(flute_predict(net, torch.from_numpy(c[i:i + 16]).cuda(), torch.from_numpy(m[i:i + 16]).cuda()).cpu().numpy())
        tau = np.concatenate(preds)
        st = summarize(tau, d["tau_gt"], d["mask"])
        level = int(round(float(d["photons"][0])))
        rows.append({"method": "FLIMngo-FLUTE", "subset": "all_fg", "photons": level, "tau_mare_pct": st["mare_pct"],
                     "tau_bias_pct": st["bias_pct"], "invalid_frac": float(np.mean(~m[d["mask"]]))})
        print(f"p{level}: MARE {st['mare_pct']:.2f} %  bias {st['bias_pct']:+.2f} %")
    out = TABLES / "flimngo_flute_test.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved {out}")

    dpath = TABLES / f"diffflim_{args.diffflim_tag}.csv"
    if dpath.exists():
        comb = []
        with open(dpath) as fh:
            for r in csv.DictReader(fh):
                if (r["method"] == "DiffFLIM" and r["subset"] == "all_fg") or (r["method"] != "DiffFLIM" and r["subset"] == "mle_valid"):
                    comb.append({"method": r["method"], "subset": r["subset"], "photons": r["photons"],
                                 "tau_mare_pct": r["tau_mare_pct"], "tau_bias_pct": r["tau_bias_pct"]})
        comb += [{k: r[k] for k in ("method", "subset", "photons", "tau_mare_pct", "tau_bias_pct")} for r in rows]
        out2 = TABLES / "flute_sim_comparison.csv"
        with open(out2, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(comb[0]))
            w.writeheader()
            for r in comb:
                w.writerow({k: (f"{float(v):.4g}" if k.startswith("tau") else v) for k, v in r.items()})
        print(f"Saved {out2}")


if __name__ == "__main__":
    main()
