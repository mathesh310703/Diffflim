"""Inspection of the FLUTE example data (data/raw/flute/, Gottlieb et al. 2023, Zenodo 8046636, CC BY 4.0).

For each file: shape, dtype, whether the values are integers, photons per pixel (summed over the
56 time bins), the summed decay (peak bin, level before the rise and at the end, both relative to
the peak), and a Poisson check: for horizontally adjacent pixels, var(x_i - x_j) / mean(x_i + x_j)
per time bin, pooled over the image (1 for Poisson counts with equal expectation; > 1 where the
two pixels differ in their expected counts, e.g. at cell edges, or for over-dispersed data).

Writes results/tables/real_flute_inspection.csv and results/figures/real_flute_decays.png.

Usage: python scripts/inspect_flute.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.flute import FLUTE_BIN_NS, FLUTE_FILES, FLUTE_PERIOD_NS, load_flute  # noqa: E402
from src.paths import FIGURES, TABLES  # noqa: E402


def inspect(name):
    a = load_flute(name)
    t, h, w = a.shape
    tot = a.sum(0)
    dec = a.sum((1, 2)).astype(float)
    pk = int(dec.argmax())
    rise = int(np.argmax(dec > 0.5 * dec.max()))           # first bin above half maximum
    pre = dec[:max(rise - 1, 1)].mean() / dec.max()        # bins clearly before the rise
    end = dec[-3:].mean() / dec.max()
    d = np.diff(a.astype(float), axis=2)
    ssum = a[:, :, 1:].astype(float) + a[:, :, :-1]
    disp = float((d ** 2).sum() / ssum.sum())
    return a, {
        "file": FLUTE_FILES[name], "key": name, "shape_T_H_W": f"{t}x{h}x{w}", "dtype": "uint16",
        "integer_values": bool(np.all(a == np.round(a))), "max_count_per_bin": int(a.max()),
        "photons_px_median": float(np.median(tot)), "photons_px_p10": float(np.percentile(tot, 10)),
        "photons_px_p90": float(np.percentile(tot, 90)), "photons_px_max": int(tot.max()),
        "photons_px_mean": float(tot.mean()), "zero_pixels_pct": 100 * float((tot == 0).mean()),
        "peak_bin": pk, "peak_ns": pk * FLUTE_BIN_NS, "halfmax_rise_bin": rise,
        "pre_rise_level_pct_of_peak": 100 * pre, "end_level_pct_of_peak": 100 * end,
        "poisson_dispersion_adjacent": disp,
    }, dec, tot


def main():
    rows, decs, tots = [], {}, {}
    for name in FLUTE_FILES:
        _, r, dec, tot = inspect(name)
        rows.append(r)
        decs[name], tots[name] = dec, tot
        print(", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    out = TABLES / "real_flute_inspection.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved {out}")

    tb = np.arange(len(next(iter(decs.values())))) * FLUTE_BIN_NS
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, (k, dec) in enumerate(decs.items()):
        ax[0].semilogy(tb, dec / dec.max(), "-o", ms=2.5, lw=1.2, color=f"C{i}", label=FLUTE_FILES[k])
        ax[1].hist(tots[k].ravel(), bins=np.logspace(0, np.log10(4000), 60), histtype="step", lw=1.3, color=f"C{i}",
                   label=FLUTE_FILES[k])
    t0 = tb[int(np.argmax(decs["fluorescein_hmsc"]))]
    ax[0].semilogy(tb[tb >= t0], np.exp(-(tb[tb >= t0] - t0) / 4.0), "k--", lw=1, label="exp(−t / 4 ns)")
    ax[0].set_xlabel(f"Time (ns), 56 bins of {FLUTE_BIN_NS} ns, period {FLUTE_PERIOD_NS} ns")
    ax[0].set_ylabel("Summed decay / peak")
    ax[0].set_ylim(3e-3, 1.5)
    ax[0].set_title("Summed decay of each file")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Photons per pixel (sum over 56 bins)")
    ax[1].set_ylabel("Number of pixels")
    ax[1].set_title("Photons per pixel")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "real_flute_decays.png", dpi=120)
    print(f"Saved {FIGURES / 'real_flute_decays.png'}")


if __name__ == "__main__":
    main()
