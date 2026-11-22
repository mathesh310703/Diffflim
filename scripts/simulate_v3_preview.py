"""Preview of one simulator-v3 sample (training IDs only).

Saves results/figures/simulate_v3_preview.png: the 4 HPA input channels of the crop, true
lifetime (amplitude-weighted), true intensity, summed counts, number of lifetime components,
3 example pixel histograms with their true expected curves, and the IRF library (used IRF bold).

Note: the figure shows 64 x 64 crops of HPA images (academic use only, cite the HPA).

Usage:
    python scripts/simulate_v3_preview.py --photons 100 --seed 3
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import simulate_v3 as v3  # noqa: E402
from src.paths import FIGURES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photons", type=float, default=100, help="mean signal photons per foreground pixel")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    cfg = v3.SimConfigV3()
    s = v3.simulate_sample(np.random.default_rng(args.seed), cfg, args.photons,
                           v3.train_ids(write=False), return_extra=True)
    tau, inten, counts, mask = s["tau_gt"], s["intensity_gt"], s["counts"], s["mask"]
    t = cfg.time_centers

    fg_tau = np.where(mask, tau, np.nan)
    picks = {"short tau": np.unravel_index(np.nanargmin(fg_tau), tau.shape),
             "long tau": np.unravel_index(np.nanargmax(fg_tau), tau.shape),
             "brightest": np.unravel_index(np.argmax(inten), tau.shape)}
    colors = ["tab:orange", "tab:cyan", "tab:red"]

    fig, ax = plt.subplots(3, 4, figsize=(17, 12.5))
    for a, c, name in zip(ax[0], range(4), ["red: microtubules", "green: protein", "blue: nucleus", "yellow: ER"]):
        a.imshow(s["channels"][..., c], cmap="gray")
        a.set_title(f"HPA {name}")
        a.axis("off")

    lo, hi = np.nanpercentile(fg_tau, [1, 99])
    panels = [(np.where(mask, tau, np.nan), f"True lifetime, amplitude-weighted (ns)", "viridis", (lo, hi)),
              (inten, "True intensity (signal photons)", "magma", None),
              (counts.sum(0), "Measured counts (sum over time)", "magma", None),
              (s["n_components"], "Number of lifetime components", "tab10", (0, 9))]
    for a, (img, title, cmap, lim) in zip(ax[1], panels):
        im = a.imshow(img, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
        fig.colorbar(im, ax=a, fraction=0.046)
        a.set_title(title, fontsize=10)
        a.axis("off")
        for (y, x), c in zip(picks.values(), colors):
            a.plot(x, y, "o", mfc="none", mec=c, ms=9, mew=2)

    for a, ((name, (y, x)), c) in zip(ax[2][:3], zip(picks.items(), colors)):
        a.step(t, counts[:, y, x], where="mid", color="gray", lw=0.8, label="measured counts")
        a.plot(t, s["expected"][:, y, x], color=c, lw=2, label="true expected curve")
        a.set_title(f"{name} ({y},{x}): tau_amp={tau[y, x]:.2f}, tau_int={s['tau_int_gt'][y, x]:.2f} ns,\n"
                    f"{s['n_components'][y, x]} comp., I={inten[y, x]:.0f}, N={int(counts[:, y, x].sum())}",
                    fontsize=9)
        a.set_xlabel("Time (ns)")
        a.legend(fontsize=8)

    a = ax[2][3]
    for x in v3.irf_library(cfg):
        a.plot(t, x["irf"] / x["irf"].max(), lw=0.7, alpha=0.5)
    a.plot(t, s["irf"] / s["irf"].max(), "k", lw=2, label=f"used: {s['irf_id']}, shift {s['irf_shift']:+d} bins")
    a.set_xlim(0, 10)
    a.set_xlabel("Time (ns)")
    a.set_title("IRF library (12, thin) and IRF used (bold)", fontsize=10)
    a.legend(fontsize=8)

    fig.suptitle(f"Simulator v3: {args.photons:g} photons per foreground pixel, mask = {mask.mean():.0%} of crop, "
                 f"seed {args.seed} (HPA crops: academic use only)")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "simulate_v3_preview.png"
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
