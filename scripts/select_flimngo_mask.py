"""Choose a recall-oriented background mask for FLIMngo's input, on the calibration sets only.

The mask "otsu_adaptive" (src/evaluation/calibration.py::foreground_mask) was tuned for the
photon-level estimate, where precision matters, and keeps only 74 to 81 % of the true foreground
of the v3 test set. FLIMngo sets every pixel outside its mask to zero, so missed foreground pixels
lose their data. Here candidate masks from the summed counts are compared on the two v3
calibration sets (74 cells, 1200 images):

  frac     smoothed sum > f * P99(smoothed sum)        (sigma, f)
  frac+fill  the same, holes filled                     (sigma, f)
  otsu_k   otsu_adaptive smoothing, threshold * k       (k)
  dilate   otsu_adaptive mask dilated by r pixels       (r)

Rule: recall (fraction of true foreground pixels kept, pooled per photon level) >= 0.95 at EVERY
level; among those, the highest mean fraction of true background excluded (over the six levels).
The test set is not used for the choice; it is only reported for the chosen rule.

Writes results/tables/flimngo_mask_selection.csv (all candidates, calibration sets) and prints the
chosen rule with its test-set numbers.

Usage:
    python scripts/select_flimngo_mask.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.simulate import load_dataset  # noqa: E402
from src.evaluation.calibration import _otsu_threshold, foreground_mask  # noqa: E402
from src.paths import RAW_OWN, TABLES  # noqa: E402

LEVELS = (10, 20, 50, 100, 500, 1000)
SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0)
FRACS = (0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20)
KS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
RADII = (1, 2, 3)
MIN_RECALL = 0.95


def load_sums(folders, prefix):
    sums, masks, levels = [], [], []
    for folder in folders:
        for path in sorted(folder.glob(f"{prefix}_p*.npz"), key=lambda p: int(p.stem.split("_p")[1])):
            d = load_dataset(path)
            sums.append(d["counts"].sum(1, dtype=np.float64))
            masks.append(d["mask"])
            levels.append(np.round(d["photons"]).astype(int))
    return np.concatenate(sums), np.concatenate(masks), np.concatenate(levels)


def adaptive_sigma(s):
    p99 = np.percentile(s, 99)
    return 1.0 if p99 < 30 else (0.5 if p99 < 300 else 0.0)


def candidates(s):
    """All candidate masks for one summed-count image s (H, W): {name: mask}."""
    out = {}
    for sig in SIGMAS:
        sm = gaussian_filter(s, sig) if sig else s
        ref = np.percentile(sm, 99)
        for f in FRACS:
            m = sm > f * ref
            out[f"frac sigma={sig} f={f}"] = m
            out[f"frac+fill sigma={sig} f={f}"] = binary_fill_holes(m)
    sig = adaptive_sigma(s)
    sm = gaussian_filter(s, sig) if sig else s
    thr = _otsu_threshold(sm.ravel())
    for k in KS:
        out[f"otsu_k k={k}"] = sm > k * thr
    base = sm > thr
    for r in RADII:
        out[f"dilate r={r}"] = binary_dilation(base, iterations=r)
    return out


def score(sums, masks, levels, mask_fn):
    """Per level: recall, background excluded, precision; plus the 5th percentile of per-image recall."""
    acc = {}
    for s, fg, lev in zip(sums, masks, levels):
        for name, m in mask_fn(s).items():
            a = acc.setdefault(name, {l: np.zeros(4) for l in LEVELS} | {"img_recall": []})
            a[lev] += [(m & fg).sum(), fg.sum(), (~m & ~fg).sum(), (~fg).sum()]
            a["img_recall"].append((m & fg).sum() / max(fg.sum(), 1))
            acc[name] = a
    rows = []
    for name, a in acc.items():
        r = {"candidate": name}
        for l in LEVELS:
            tp, p, tn, n = a[l]
            r[f"recall_{l}"] = tp / p
            r[f"bg_excluded_{l}"] = tn / n
        r["min_recall"] = min(r[f"recall_{l}"] for l in LEVELS)
        r["mean_bg_excluded"] = float(np.mean([r[f"bg_excluded_{l}"] for l in LEVELS]))
        r["p5_image_recall"] = float(np.percentile(a["img_recall"], 5))
        rows.append(r)
    return rows


def main():
    cal_dirs = [RAW_OWN / "sim_v3_calib", RAW_OWN / "sim_v3_calib_extra"]
    sums, masks, levels = load_sums(cal_dirs, "calib")
    print(f"Calibration: {len(sums)} images, true foreground {masks.mean():.1%} of pixels")
    rows = score(sums, masks, levels, candidates)
    ok = [r for r in rows if r["min_recall"] >= MIN_RECALL]
    best = max(ok, key=lambda r: r["mean_bg_excluded"])
    rows.sort(key=lambda r: (-int(r["min_recall"] >= MIN_RECALL), -r["mean_bg_excluded"]))

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "flimngo_mask_selection.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved {out} ({len(rows)} candidates, {len(ok)} with recall >= {MIN_RECALL} at every level)")

    print("\nBest 8 candidates (calibration sets):")
    for r in rows[:8]:
        print(f"  {r['candidate']:28s} min recall {r['min_recall']:.3f}  bg excluded {r['mean_bg_excluded']:.3f}  "
              f"p5 image recall {r['p5_image_recall']:.3f}")
    for name in ("otsu_adaptive (reference)",):
        ref = score(sums, masks, levels, lambda s: {name: foreground_mask(s, "otsu_adaptive")})[0]
        print(f"  {name:28s} min recall {ref['min_recall']:.3f}  bg excluded {ref['mean_bg_excluded']:.3f}")
    print(f"\nChosen: {best['candidate']}")
    print("  calibration: " + "  ".join(f"{l}: recall {best[f'recall_{l}']:.3f} / bg excl. {best[f'bg_excluded_{l}']:.3f}"
                                         for l in LEVELS))

    tsums, tmasks, tlevels = load_sums([RAW_OWN / "sim_v3_test"], "test")
    for method in ("otsu_adaptive", "flimngo_recall"):
        try:
            t = score(tsums, tmasks, tlevels, lambda s: {method: foreground_mask(s, method)})[0]
        except ValueError:
            print(f"  ({method} not yet in foreground_mask)")
            continue
        print(f"  test, {method}: " + "  ".join(f"{l}: {t[f'recall_{l}']:.3f}/{t[f'bg_excluded_{l}']:.3f}" for l in LEVELS))


if __name__ == "__main__":
    main()
