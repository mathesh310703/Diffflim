"""Load one FLIMngo sample, print its shapes and save a quick-look figure.

Saves results/figures/inspect_sample.png with: intensity image, ground-truth lifetime map,
and the decay curve of the brightest pixel.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.flimngo import BIN_WIDTH_NS, load_h5  # noqa: E402
from src.paths import FIGURES, RAW_FLIMNGO  # noqa: E402


def main():
    files = sorted((RAW_FLIMNGO / "h5").glob("*.h5"))
    if not files:
        sys.exit("No .h5 files found. Run scripts/download_flimngo.py first.")

    decay, tau = load_h5(files[0])
    print(f"File:  {files[0].name}")
    print(f"decay: {decay.shape}  (t, x, y)  min={decay.min():.3f} max={decay.max():.3f}")
    print(f"tau:   {tau.shape}  (x, y)     lifetimes {tau[tau > 0].min():.2f}-{tau.max():.2f} ns")

    intensity = decay.sum(axis=0)
    x, y = np.unravel_index(np.argmax(intensity), intensity.shape)
    t = np.arange(decay.shape[0]) * BIN_WIDTH_NS

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].imshow(intensity, cmap="gray"); ax[0].set_title("Summed decay (intensity)")
    im = ax[1].imshow(tau, cmap="viridis"); ax[1].set_title("Ground-truth lifetime (ns)")
    fig.colorbar(im, ax=ax[1])
    ax[2].plot(t, decay[:, x, y]); ax[2].set_xlabel("Time (ns)"); ax[2].set_title(f"Decay at pixel ({x},{y})")
    for a in ax[:2]:
        a.axis("off")
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "inspect_sample.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
