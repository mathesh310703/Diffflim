"""Preview of the FLUTE configuration of simulator v3 (flute_config()) against the real FLUTE cell images.

Simulates 200 images (64 x 64, scene seed 7300, photon level log-uniform 10 to 1000 per foreground
pixel, HPA training IDs) and compares
  (a) the summed decay of each simulated image (normalised to its peak) with the summed decays of the
      four real cell files (Embryo, hMSC control, hMSC-ZOOM, hMSC_rotenone),
  (b) the distribution of photons per pixel (simulated foreground pixels against all pixels of the
      real cell files),
  (c) one simulated image (photons per pixel and lifetime truth).

Writes results/figures/real_flute_sim_preview.png (HPA-derived: academic use only).

Usage: python scripts/flute_sim_preview.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import simulate_v3 as v3  # noqa: E402
from src.data.flute import load_flute  # noqa: E402
from src.paths import FIGURES  # noqa: E402

N_SIM = 200
SEED = 7300
REAL = ("embryo", "hmsc_control", "hmsc_zoom", "hmsc_rotenone")


def main():
    cfg = v3.flute_config()
    ids = v3.train_ids()
    rng = np.random.default_rng(SEED)
    sims = [v3.simulate_sample(rng, cfg, float(np.exp(rng.uniform(np.log(10), np.log(1000)))), ids) for _ in range(N_SIM)]
    t = cfg.time_centers
    fig, ax = plt.subplots(1, 4, figsize=(18, 4), gridspec_kw={"width_ratios": [1.3, 1.3, 0.8, 0.8]})
    for s in sims[:60]:
        d = s["counts"].sum((1, 2)).astype(float)
        ax[0].semilogy(t, d / d.max(), color="gray", lw=0.5, alpha=0.4)
    ax[0].plot([], [], color="gray", lw=1, label="simulated images (60 of 200)")
    real_px = []
    for i, k in enumerate(REAL):
        a = load_flute(k)
        d = a.sum((1, 2)).astype(float)
        ax[0].semilogy(t, d / d.max(), lw=2, color=f"C{i}", label=f"real: {k}")
        real_px.append(a.sum(0).ravel())
    ax[0].set_ylim(3e-3, 1.5)
    ax[0].set_xlabel("Time (ns)")
    ax[0].set_ylabel("Summed decay / peak")
    ax[0].set_title("Decays: FLUTE simulator vs real cell images")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")

    bins = np.logspace(0, np.log10(5000), 60)
    sim_px = np.concatenate([s["counts"][:, s["mask"]].sum(0) for s in sims])
    ax[1].hist(sim_px, bins=bins, density=True, histtype="step", lw=2, color="gray", label="simulated (foreground)")
    for i, (k, px) in enumerate(zip(REAL, real_px)):
        ax[1].hist(px, bins=bins, density=True, histtype="step", lw=1.3, color=f"C{i}", label=f"real: {k}")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Photons per pixel")
    ax[1].set_ylabel("Density")
    ax[1].set_title("Photons per pixel")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)

    s = sims[3]
    im = ax[2].imshow(s["counts"].sum(0), cmap="magma")
    ax[2].set_title(f"Simulated image, {s['photons']:.0f} photons\nper fg pixel (counts)", fontsize=9)
    fig.colorbar(im, ax=ax[2], shrink=0.7)
    im = ax[3].imshow(np.where(s["mask"], s["tau_gt"], np.nan), cmap="viridis")
    ax[3].set_title(f"Lifetime truth (ns)\nIRF {s['irf_id']}", fontsize=9)
    fig.colorbar(im, ax=ax[3], shrink=0.7)
    for a in ax[2:]:
        a.axis("off")
    fig.tight_layout()
    out = FIGURES / "real_flute_sim_preview.png"
    fig.savefig(out, dpi=110)
    print(f"Saved {out}; simulated fg photons median {np.median(sim_px):.0f}, real cell files median "
          + ", ".join(f"{k} {np.median(px):.0f}" for k, px in zip(REAL, real_px)))


if __name__ == "__main__":
    main()
