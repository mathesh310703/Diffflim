"""Error metrics for lifetime / intensity maps, computed on foreground pixels only.

Relative error per pixel: (estimate - truth) / truth.
  * mare  = median of |relative error|   (robust "typical error", in %)
  * bias  = median of relative error      (robust signed bias, in %)
  * invalid = fraction of foreground pixels where the estimate is NaN/inf (e.g. 0 photons).
    Invalid pixels are left out of mare/bias, so always look at this column too.
"""
import numpy as np


def relative_errors(est, gt, mask):
    """Flattened relative errors over mask pixels, plus the number of invalid estimates."""
    est, gt = np.asarray(est, dtype=np.float64)[mask], np.asarray(gt, dtype=np.float64)[mask]
    valid = np.isfinite(est)
    return (est[valid] - gt[valid]) / gt[valid], int((~valid).sum()), int(mask.sum())


def summarize(est, gt, mask):
    """est, gt: (N, H, W) or (H, W); mask: same shape, bool. Returns dict of metrics in %."""
    rel, n_invalid, n_total = relative_errors(est, gt, mask)
    return {
        "mare_pct": 100 * float(np.median(np.abs(rel))) if rel.size else np.nan,
        "bias_pct": 100 * float(np.median(rel)) if rel.size else np.nan,
        "invalid_frac": n_invalid / max(n_total, 1),
        "n_pixels": n_total,
    }


# ----------------------------------------------------------------------------------------
# Pixel subsets and ground-truth roughness (added for simulator v3)
# ----------------------------------------------------------------------------------------
BRIGHT_FRAC = 0.20


def bright_mask(intensity_gt, mask, frac=BRIGHT_FRAC):
    """"Bright" foreground pixels: true intensity at least `frac` of the crop's intensity RANGE
    above its dimmest foreground pixel, (I - I_min,fg) / (I_max - I_min,fg) >= frac, per image.

    Range-based on purpose: in simulator v3 (FLIMngo intensity mapping) every foreground pixel has
    at least 28 % of the crop maximum, so a threshold relative to the maximum alone ("I >= 20 % of
    max") selects all foreground pixels. intensity_gt, mask: (N, H, W) or (H, W)."""
    inten = np.asarray(intensity_gt, dtype=np.float64)
    m = np.asarray(mask, bool)
    lo = np.where(m, inten, np.inf).min(axis=(-2, -1), keepdims=True)
    hi = np.where(m, inten, -np.inf).max(axis=(-2, -1), keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (inten - lo) / (hi - lo)
    return m & (rel >= frac)


def local_jitter(tau_gt, mask):
    """Per-pixel lifetime jitter of the ground truth: |tau - median of tau over the 3 x 3
    foreground neighbourhood (incl. the pixel)|. Returns (absolute in ns, relative to tau),
    NaN outside the mask. This is the part of the true map that a method which averages over
    neighbouring pixels cannot recover."""
    tau = np.where(mask, np.asarray(tau_gt, dtype=np.float64), np.nan)
    pad = np.pad(tau, [(0, 0)] * (tau.ndim - 2) + [(1, 1), (1, 1)], constant_values=np.nan)
    H, W = tau.shape[-2:]
    shifts = [pad[..., dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.stack(shifts), axis=0)
    absj = np.abs(tau - med)
    return absj, absj / tau


def jitter_summary(tau_gt, mask):
    """Median absolute (ns) and relative (%) local jitter over the mask pixels."""
    a, r = local_jitter(tau_gt, mask)
    m = np.asarray(mask, bool)
    return {"jitter_median_ns": float(np.median(a[m])), "jitter_median_pct": 100 * float(np.median(r[m]))}
