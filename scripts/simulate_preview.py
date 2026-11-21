"""Quick look at one simulated FLIM image.

Saves results/figures/simulate_preview_p{photons}.png with: ground-truth lifetime, ground-truth
intensity, summed measured counts, and the histograms of 3 example pixels with their true
expected decay curve.

Usage:
    python scripts/simulate_preview.py --photons 100 --seed 0
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.simulate import SimConfig, expected_counts, simulate_sample  # noqa: E402
from src.paths import FIGURES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photons", type=float, default=100, help="mean signal photons per foreground pixel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    cfg = SimConfig(size=args.size)
    s = simulate_sample(np.random.default_rng(args.seed), cfg, args.photons)
    tau, inten, counts, mask = s["tau_gt"], s["intensity_gt"], s["counts"], s["mask"]
    expected = expected_counts(tau, inten, cfg, s["bg_per_bin"])
    t = cfg.time_centers

    # Example pixels: among the foreground pixels, the shortest and longest lifetime, plus the brightest.
    tau_b = np.where(mask, tau, np.nan)
    picks = {
        "brightest": np.unravel_index(np.argmax(inten), inten.shape),
        "short tau": np.unravel_index(np.nanargmin(tau_b), tau.shape),
        "long tau": np.unravel_index(np.nanargmax(tau_b), tau.shape),
    }
    colors = ["tab:red", "tab:orange", "tab:cyan"]

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for a, img, title, cmap in [
        (ax[0, 0], tau, "Ground-truth lifetime (ns)", "viridis"),
        (ax[0, 1], inten, "Ground-truth intensity (signal photons)", "magma"),
        (ax[0, 2], counts.sum(axis=0), "Measured counts (sum over time)", "magma"),
    ]:
        im = a.imshow(img, cmap=cmap)
        fig.colorbar(im, ax=a, fraction=0.046)
        a.set_title(title)
        a.axis("off")
        a.contour(mask, levels=[0.5], colors="white", linewidths=0.8)   # foreground outline
        for (name, (y, x)), c in zip(picks.items(), colors):
            a.plot(x, y, "o", mfc="none", mec=c, ms=9, mew=2)

    for a, ((name, (y, x)), c) in zip(ax[1], zip(picks.items(), colors)):
        a.step(t, counts[:, y, x], where="mid", color="gray", lw=0.8, label="measured counts")
        a.plot(t, expected[:, y, x], color=c, lw=2, label="true expected decay")
        a.set_title(f"{name}: pixel ({y},{x})  tau={tau[y, x]:.2f} ns, "
                    f"I={inten[y, x]:.0f}, N={int(counts[:, y, x].sum())}")
        a.set_xlabel("Time (ns)")
        a.set_ylabel("Counts per bin")
        a.legend()

    fig.suptitle(f"Simulated FLIM, mean {args.photons:g} photons per foreground pixel "
                 f"(white outline = mask, {mask.mean():.0%} of image), seed {args.seed}")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"simulate_preview_p{args.photons:g}.png"
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
