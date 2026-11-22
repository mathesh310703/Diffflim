"""Split-conformal calibration of per-pixel uncertainty (model- and simulator-agnostic).

Input is always a "prediction record": a dict of arrays for N images,
    {"<q>_mean", "<q>_std", "<q>_gt"}  for each quantity q (here "tau", "int"), shape (N, H, W)
    "mask"         (N, H, W) bool   pixels that count (foreground)
    "photons_est"  (N,)             photon level ESTIMATED from the counts (estimate_photon_level)
Nothing here knows about the simulator or the model, so the same code calibrates any future
model / simulator version as long as it produces such a record.

Method (split conformal, Vovk / Angelopoulos & Bates):
    score s_i = |truth - mean| / std              on every calibration foreground pixel
    q(alpha)  = the ceil((n + 1)(1 - alpha))-th smallest score   (finite-sample correction)
    interval  = mean +/- q(alpha) * std            -> P(truth inside) >= 1 - alpha
Variants:
    (a) global:    one q per quantity and alpha, from all calibration pixels;
    (b) per-level: one q per photon level; for a test image q is read off a curve in log photons
                   at the image's ESTIMATED photon level: linear interpolation (default) or, with
                   smooth=True, a least-squares quadratic in log photons through the per-level
                   factors (clamped to the calibrated range).

Caveat: pixels within one image are strongly correlated (same cells, same noise draw), so the
guarantee holds for a random pixel on average over images, not for every single image. The
data are not i.i.d. pixels; with 20 calibration images per level the effective sample size is
closer to the number of images than to the number of pixels.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

QUANTITIES = ("tau", "int")
STD_FLOOR = 1e-6   # avoids division by zero for a (degenerate) zero spread


# ----------------------------------------------------------------------------------------
# Photon-level estimate (uses only the measured counts)
# ----------------------------------------------------------------------------------------
def _otsu_threshold(values, n_bins=128):
    """Otsu's threshold: the split that maximises the between-class variance."""
    hist, edges = np.histogram(values, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = ((hist * centers).sum() - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    return centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def estimate_photon_level(counts_sum, smooth=1.5):
    """Mean photons per foreground pixel, estimated from the summed counts (H, W) alone.

    Foreground = Otsu threshold on log1p of the smoothed photon image; the level is the mean raw
    count inside it (includes the small background, ~2% in the v1 simulator).
    """
    smoothed = gaussian_filter(np.asarray(counts_sum, dtype=np.float64), smooth)
    fg = np.log1p(smoothed) > _otsu_threshold(np.log1p(smoothed).ravel())
    if not fg.any():
        return float(np.mean(counts_sum))
    return float(np.mean(np.asarray(counts_sum)[fg]))


# ----------------------------------------------------------------------------------------
# Conformal scores and quantiles
# ----------------------------------------------------------------------------------------
def scores(rec, q, idx=None):
    """Normalised errors |truth - mean| / std on mask pixels (optionally only images idx)."""
    sl = slice(None) if idx is None else idx
    m = rec["mask"][sl]
    err = np.abs(rec[f"{q}_gt"][sl] - rec[f"{q}_mean"][sl])[m]
    return err / np.maximum(rec[f"{q}_std"][sl][m], STD_FLOOR)


def conformal_quantile(s, alpha):
    """ceil((n+1)(1-alpha))-th smallest score; +inf if n is too small for this alpha."""
    s = np.sort(np.asarray(s))
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return float(s[k - 1]) if k <= len(s) else float("inf")


def fit(calib, alphas=(0.10, 0.32), smooth=False):
    """Fit global and per-level factors. calib also needs "photons" (nominal level, used only
    to group calibration images). Returns a dict that `predict_q` understands."""
    levels = np.unique(calib["photons"])
    out = {"alphas": list(alphas), "levels": levels.tolist(), "global": {}, "per_level": {}, "n": {},
           "scores": {}, "smooth": smooth}
    for q in QUANTITIES:
        s_all = scores(calib, q)
        out["scores"][q] = {"all": s_all}
        out["n"][(q, "all")] = len(s_all)
        for a in alphas:
            out["global"][(q, a)] = conformal_quantile(s_all, a)
        for lev in levels:
            s_lev = scores(calib, q, calib["photons"] == lev)
            out["scores"][q][float(lev)] = s_lev
            out["n"][(q, float(lev))] = len(s_lev)
            for a in alphas:
                out["per_level"][(q, a, float(lev))] = conformal_quantile(s_lev, a)
    # x-position of each level for interpolation = median ESTIMATED level of its calib images,
    # so calibration and test use the same (estimated) photon scale.
    out["level_x"] = [float(np.median(calib["photons_est"][calib["photons"] == lev])) for lev in levels]
    return out


def level_curve_points(cal, q, alpha):
    """(log estimated photon level, per-level factor) for every calibration level."""
    qs = [cal["per_level"].get((q, alpha, float(l))) or _q_from_scores(cal, q, alpha, float(l))
          for l in cal["levels"]]
    return np.log(cal["level_x"]), np.array(qs)


def _q_from_scores(cal, q, alpha, key):
    s = cal["scores"][q][key]
    return conformal_quantile(s, alpha)


def predict_q(cal, q, alpha, variant, photons_est=None):
    """Scale factor(s) for quantity q at miscoverage alpha.

    variant: "none"      -> Gaussian z (what mean +/- z*std promises without calibration)
             "global"    -> one conformal factor
             "per_level" -> factor per image, interpolated in log(estimated photons)
    Any alpha works (factors not stored in `fit` are recomputed from the stored scores).
    """
    if variant == "none":
        z = norm.ppf(1 - alpha / 2)
        return z if photons_est is None else np.full(len(photons_est), z)
    if variant == "global":
        g = cal["global"].get((q, alpha)) or _q_from_scores(cal, q, alpha, "all")
        return g if photons_est is None else np.full(len(photons_est), g)
    if variant == "per_level":
        x, qs = level_curve_points(cal, q, alpha)
        xe = np.log(np.asarray(photons_est, dtype=np.float64))
        if cal.get("smooth"):
            coef = np.polyfit(x, qs, deg=min(2, len(x) - 1))
            return np.polyval(coef, np.clip(xe, x.min(), x.max()))
        return np.interp(xe, x, qs)
    raise ValueError(variant)


# ----------------------------------------------------------------------------------------
# Evaluation of intervals
# ----------------------------------------------------------------------------------------
def interval_stats(rec, q, factor, idx=None):
    """Coverage and mean relative width (2 * factor * std / truth) on mask pixels.
    factor: scalar or per-image array (N,)."""
    sl = slice(None) if idx is None else idx
    m = rec["mask"][sl]
    f = np.broadcast_to(np.asarray(factor, dtype=np.float64).reshape(-1, 1, 1) if np.ndim(factor) else factor,
                        m.shape)
    half = (f * np.maximum(rec[f"{q}_std"][sl], STD_FLOOR))[m]
    err = np.abs(rec[f"{q}_gt"][sl] - rec[f"{q}_mean"][sl])[m]
    truth = np.abs(rec[f"{q}_gt"][sl])[m]
    return float(np.mean(err <= half)), float(np.mean(2 * half / np.maximum(truth, STD_FLOOR)))
