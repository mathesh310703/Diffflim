"""Per-pixel Poisson maximum-likelihood fit of lifetime and intensity.

Forward model: exactly the simulator's (src/data/simulate.py):
    lambda_t(tau, I) = I * d_t(tau) + b,     d = decay_curves(tau)  (IRF-convolved, truncated,
                                                                     sums to 1 over the window)
Poisson negative log-likelihood (dropping terms that do not depend on tau, I):
    NLL(tau, I) = sum_t [ lambda_t - c_t * log(lambda_t) ]

Fitting, vectorised over all pixels:
  1. Intensity starting value I0 = N - b*T (total counts minus expected background). Because
     sum_t d_t = 1, this makes sum_t lambda_t = N, and with b = 0 it is the exact MLE of I.
  2. tau: evaluate NLL(tau, I0) on a log-spaced grid of lifetimes (default 0.1-8 ns), take the
     best grid point and refine it with a parabola through its neighbours (sub-grid accuracy).
  3. I: a few Newton steps on the exact MLE condition  sum_t c_t d_t / (I d_t + b) = 1  at the
     fitted tau (only differs from I0 when b > 0).

Other forward models (e.g. simulator v3 with its per-image IRF) are passed as `decay_fn`:
a function tau (any shape) -> normalised decays (T, *shape). The default is decay_curves(., cfg).

`fit_pixel_scalar` does the same fit for one pixel with scipy.optimize.minimize_scalar; it is
slow but simple, and the tests use it to check the grid version.
"""
import numpy as np
from scipy.optimize import minimize_scalar

from src.data.simulate import decay_curves

TAU_BOUNDS = (0.1, 8.0)
_EPS = 1e-12   # keeps log() finite in bins where the model is exactly 0 (only if b = 0)


def _intensity_start(counts_flat, bg_per_bin):
    n_total = counts_flat.sum(axis=1)
    return np.maximum(n_total - bg_per_bin * counts_flat.shape[1], 1e-3), n_total


def _refine_intensity(counts_flat, d, bg_per_bin, i0, n_iter=8):
    """Newton steps for the exact MLE of I at fixed decay shapes d (P, T)."""
    if bg_per_bin <= 0:
        return counts_flat.sum(axis=1).astype(np.float64)   # exact: I = N
    I = i0.copy()
    for _ in range(n_iter):
        lam = I[:, None] * d + bg_per_bin
        f = (counts_flat * d / lam).sum(axis=1) - 1.0           # score (decreasing in I)
        fp = -(counts_flat * d**2 / lam**2).sum(axis=1)
        I = np.maximum(I - f / np.minimum(fp, -1e-12), 0.0)
    return I


def fit_mle(counts, cfg, bg_per_bin, tau_bounds=TAU_BOUNDS, n_grid=200, chunk=256, decay_fn=None):
    """Fit every pixel of one image.

    counts: (T, H, W) integer histogram.  Returns tau (H, W) ns and intensity (H, W) photons.
    Pixels with zero counts get tau = NaN (lifetime undefined) and intensity = 0.
    """
    curves = decay_fn if decay_fn is not None else (lambda tau: decay_curves(tau, cfg))
    T, H, W = counts.shape
    c = counts.reshape(T, -1).T.astype(np.float32)               # (P, T)
    i0, n_total = _intensity_start(c, bg_per_bin)

    log_grid = np.linspace(np.log(tau_bounds[0]), np.log(tau_bounds[1]), n_grid)
    D = curves(np.exp(log_grid)).T.astype(np.float32)            # (G, T)

    # NLL(tau_g) for every pixel; sum_t lambda_t = I0 + bT does not depend on tau -> dropped.
    nll = np.empty((c.shape[0], n_grid), dtype=np.float64)
    for s in range(0, c.shape[0], chunk):
        cc, ii = c[s:s + chunk], i0[s:s + chunk].astype(np.float32)
        lam = ii[:, None, None] * D[None] + np.float32(bg_per_bin) + _EPS   # (p, G, T)
        nll[s:s + chunk] = -np.einsum("pt,pgt->pg", cc, np.log(lam))

    # Best grid point + parabolic refinement in log(tau).
    k = np.clip(np.argmin(nll, axis=1), 1, n_grid - 2)
    rows = np.arange(len(k))
    f0, f1, f2 = nll[rows, k - 1], nll[rows, k], nll[rows, k + 1]
    curv = f0 - 2 * f1 + f2
    h = log_grid[1] - log_grid[0]
    offset = np.where(curv > 0, 0.5 * h * (f0 - f2) / np.where(curv > 0, curv, 1.0), 0.0)
    log_tau = log_grid[k] + np.clip(offset, -h, h)
    # if the true minimum is the first/last grid point, stay on the bound
    best = np.argmin(nll, axis=1)
    log_tau = np.where(best == 0, log_grid[0], np.where(best == n_grid - 1, log_grid[-1], log_tau))
    tau = np.exp(log_tau)

    d_fit = curves(tau).T                                        # (P, T)
    intensity = _refine_intensity(c.astype(np.float64), d_fit, bg_per_bin, i0)

    tau[n_total == 0] = np.nan
    intensity[n_total == 0] = 0.0
    return tau.reshape(H, W), intensity.reshape(H, W)


def fit_pixel_scalar(counts_1d, cfg, bg_per_bin, tau_bounds=TAU_BOUNDS, decay_fn=None):
    """Reference fit of one pixel (T,) with minimize_scalar. Same likelihood as fit_mle."""
    curves = decay_fn if decay_fn is not None else (lambda tau: decay_curves(tau, cfg))
    c = counts_1d.astype(np.float64)
    i0, _ = _intensity_start(c[None], bg_per_bin)

    def nll(log_tau):
        lam = i0[0] * curves(np.array([np.exp(log_tau)]))[:, 0] + bg_per_bin + _EPS
        return -(c * np.log(lam)).sum()

    res = minimize_scalar(nll, bounds=np.log(tau_bounds), method="bounded", options={"xatol": 1e-5})
    tau = float(np.exp(res.x))
    intensity = _refine_intensity(c[None], curves(np.array([tau])).T, bg_per_bin, i0)[0]
    return tau, float(intensity)
