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
