"""Lifetime distribution of the training pixels: FLUTE simulator as trained (flute_nadh_config, lifetime
model "flute_mix") against the prepared v2 prior (flute_v2prior_config, "flute_v2"), 24 November 2026.

Scenes are drawn from the TRAINING cells only (train_ids(), nothing is written), --n-scenes per prior,
scene seed 90000. Per foreground pixel the amplitude-weighted truth tau_gt (the training target) is
collected, for all scenes and for the mono-exponential scenes alone; the share of scene types is counted
(uniform = one value, near-uniform = peak-to-peak <= 20 % of the median, NAD(P)H-like = two components).

Writes results/figures/flute_prior_lifetimes.png (histograms in log tau, pixel shares) and
results/tables/flute_prior_lifetimes.csv (shares of pixels above 3.61 / 4 / 5 ns, percentiles, scene
types). With --make-sets also the paired v2-prior sets data/raw/own/sim_flute_v2_{val,test,calib}/
(simulate_v3.make_flute_v2_sets; existing files are kept).

Usage: python scripts/flute_v2_prior_preview.py [--n-scenes 600] [--make-sets]
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import simulate_v3 as v3  # noqa: E402
from src.paths import FIGURES, TABLES  # noqa: E402

SEED = 90000


def collect(cfg, ids, n):
    rng = np.random.default_rng(SEED)
    allpx, mono, kinds = [], [], {"uniform": 0, "near-uniform": 0, "structured mono": 0, "NAD(P)H-like": 0}
    for _ in range(n):
        s = v3.make_scene(rng, cfg, ids)
        m = s["mask"]
        t = s["tau_gt"][m]
        allpx.append(t)
        if s["n_components"][m].max() >= 2:
            kinds["NAD(P)H-like"] += 1
            continue
        mono.append(t)
        r = np.ptp(t) / np.median(t)
        kinds["uniform" if r == 0 else "near-uniform" if r <= 0.2 else "structured mono"] += 1
    return np.concatenate(allpx), np.concatenate(mono), kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=600)
    ap.add_argument("--make-sets", action="store_true")
    args = ap.parse_args()
    ids = v3.train_ids(write=False)
    priors = {"as trained (flute_mix)": v3.flute_nadh_config(), "v2 prior (flute_v2)": v3.flute_v2prior_config()}
    res = {k: collect(c, ids, args.n_scenes) for k, c in priors.items()}
    rows = []
    for k, (a, mo, kinds) in res.items():
        for subset, v in (("all pixels", a), ("mono-exponential scenes", mo)):
            row = {"prior": k, "subset": subset, "n_scenes": args.n_scenes, "n_pixels": int(v.size),
                   **{f"share_ge_{x:g}ns_pct": 100 * float(np.mean(v >= x)) for x in (3.61, 4.0, 5.0)},
                   **{f"p{q}_ns": float(np.percentile(v, q)) for q in (1, 10, 50, 90, 99)}, "max_ns": float(v.max()),
                   **{f"scenes_{n.replace(' ', '_').replace('(', '').replace(')', '')}": c for n, c in kinds.items()}}
            rows.append(row)
            print(f"{k:24s} {subset:24s} >=3.61 {row['share_ge_3.61ns_pct']:5.1f} % >=4 {row['share_ge_4ns_pct']:5.1f} % "
                  f">=5 {row['share_ge_5ns_pct']:5.1f} % median {row['p50_ns']:.2f} p99 {row['p99_ns']:.2f} max {row['max_ns']:.2f}; {kinds}")
    with open(TABLES / "flute_prior_lifetimes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})

    bins = np.exp(np.linspace(np.log(0.1), np.log(8), 61))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for a, (j, title) in zip(ax, ((0, "All training pixels"), (1, "Mono-exponential scenes only"))):
        for (k, r), col in zip(res.items(), ("#2a6fdb", "#e07b00")):
            v = r[j]
            a.hist(v, bins=bins, weights=np.full(v.size, 100 / v.size), histtype="step", lw=2, color=col, label=k)
        for x in (3.61, 4.0):
            a.axvline(x, color="0.4", ls=":", lw=1)
        a.text(4.1, a.get_ylim()[1] * 0.97, "fluorescein\n3.61 / 4.0 ns", fontsize=8, color="0.3", va="top")
        a.set_xscale("log")
        a.set_xlabel("Amplitude-weighted lifetime tau_gt (ns), log scale")
        a.set_title(title, fontsize=10)
        a.grid(alpha=0.25, which="both", lw=0.5)
    ax[0].set_ylabel("Share of pixels per bin (%)")
    ax[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"FLUTE simulator: lifetime distribution of training pixels ({args.n_scenes} scenes per prior, training cells)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "flute_prior_lifetimes.png", dpi=120)
    print(f"Saved {FIGURES / 'flute_prior_lifetimes.png'} and {TABLES / 'flute_prior_lifetimes.csv'}")
    if args.make_sets:
        v3.make_flute_v2_sets()


if __name__ == "__main__":
    main()
