"""Poisson data-consistency guidance for DDIM sampling (no retraining; added 24 November 2026).

Idea (DPS, Chung et al. 2023; decomposed / "x0-space" variants such as DDS and DynamicDPS): at every
DDIM step the network's estimate of the clean maps x0 = (normalised log tau, normalised log1p I) is
moved towards a higher Poisson likelihood of the MEASURED counts, and the DDIM update then uses the
moved x0 (the noise estimate is recomputed from it). The network is not differentiated; only the
forward model is.

Forward model (per pixel, the same one the classical baselines use for the data set at hand):
    lambda_k = I * d_k(tau) + b,     log L = sum_k [ c_k log lambda_k - lambda_k ]
with d(tau) the normalised mono-exponential decay convolved with the image's IRF (periodic for FLUTE),
b the background per bin and c the counts in the full time resolution (not the binned network input).
The decay model is passed as any numpy function tau -> (T, n) (e.g. decay_curves_v3 with the image's
IRF, or src.data.flute.PeriodicDecayModel); it is tabulated once on a log-spaced lifetime grid and
interpolated linearly in log tau on the GPU, together with its derivative d d / d log tau.

Update (one Fisher-scoring step per DDIM step and channel, in normalised units theta):
    theta <- theta + clip( g / (F + kappa), -max_step, max_step )
    g = sum_k (c_k / lambda_k - 1) d lambda_k / d theta      (score)
    F = sum_k (d lambda_k / d theta)^2 / lambda_k            (Fisher information, diagonal)
This is the posterior mode of a one-step Laplace approximation that combines the likelihood with a
Gaussian prior centred on the network's x0 with precision kappa (normalised units). kappa -> infinity:
no guidance; kappa = 0: a full step towards the per-pixel MLE. In lifetime terms the prior standard
deviation is about tau_scale / sqrt(kappa) in log tau (tau_scale = 2.39 for the tau range 0.1 to 12 ns:
kappa 100 -> 24 %, 1000 -> 7.6 %). Because F grows with the photon number, the data automatically get
more weight at high photon levels. Guidance is applied at the DDIM step indices i with
start_frac * n_steps <= i < stop_frac * n_steps (default 0 and 1: every step; stop_frac < 1 leaves the
last steps to the network alone). The prior precision kappa is the same at every step.

Limitation: the likelihood is mono-exponential. On multi-exponential pixels (NAD(P)H-like scenes, real
cells) it pulls the lifetime towards the mono-exponential fit, i.e. towards an intensity-weighted
lifetime, not towards the amplitude-weighted lifetime the models are trained to predict.
"""
import numpy as np
import torch

from src.models.diffflim import INT_LOG_CENTER, INT_LOG_SCALE, _tau_consts

GRID = (0.02, 60.0)     # ns, covers the clamp range of x0 (+-1.5 in normalised units: 0.03 to 40 ns)
N_GRID = 2048
LAM_EPS = 1e-6


class PoissonGuidance:
    """Guidance specification for one forward model (one IRF, one background level).

    decay_fn: tau (1-D numpy array, ns) -> normalised decays (T, len(tau)).
    bg_per_bin: expected background counts per pixel and time bin.
    kappa: prior precision in normalised units (None or inf: no guidance).
    """

    def __init__(self, decay_fn, bg_per_bin, kappa, start_frac=0.0, stop_frac=1.0, max_step=0.5,
                 channels=("tau", "int")):
        self.decay_fn = decay_fn
        self.bg = float(bg_per_bin)
        self.kappa = float(kappa)
        self.start_frac = float(start_frac)
        self.stop_frac = float(stop_frac)
        self.max_step = float(max_step)
        self.channels = tuple(channels)
        self._table = {}

    def table(self, device):
        if device not in self._table:
            lg = np.linspace(np.log(GRID[0]), np.log(GRID[1]), N_GRID)
            D = np.asarray(self.decay_fn(np.exp(lg)), dtype=np.float64).T          # (G, T)
            dD = np.gradient(D, lg, axis=0)
            self._table[device] = (torch.tensor(D, dtype=torch.float32, device=device),
                                   torch.tensor(dD, dtype=torch.float32, device=device),
                                   float(lg[0]), float(lg[1] - lg[0]))
        return self._table[device]

    def bind(self, counts, tau_range):
        """counts: torch (T, H, W) on the model device -> callable(x0, i, n_steps) -> x0."""
        D, dD, g0, h = self.table(counts.device)
        C = counts.float().permute(1, 2, 0).contiguous()                           # (H, W, T)
        c_tau, s_tau = _tau_consts(tau_range)
        G = D.shape[0]
        guide = self

        def step(x0, i, n_steps):
            if not guide.start_frac * n_steps <= i < guide.stop_frac * n_steps:
                return x0
            with torch.autocast(x0.device.type, enabled=False):
                x = x0.float().clone()
                lt = x[:, 0] * s_tau + c_tau                                        # log tau (B, H, W)
                pos = ((lt - g0) / h).clamp(0, G - 1.0001)
                j = pos.floor().long()
                w = (pos - j)[..., None]
                d = D[j] * (1 - w) + D[j + 1] * w                                    # (B, H, W, T)
                dd = dD[j] * (1 - w) + dD[j + 1] * w
                e = torch.exp(x[:, 1] * INT_LOG_SCALE + INT_LOG_CENTER)
                inten = (e - 1).clamp(min=0)[..., None]
                lam = inten * d + guide.bg + LAM_EPS
                r = C / lam - 1.0
                if "tau" in guide.channels:
                    dl = inten * dd * s_tau
                    g, F = (r * dl).sum(-1), (dl * dl / lam).sum(-1)
                    x[:, 0] += (g / (F + guide.kappa).clamp(min=1e-6)).clamp(-guide.max_step, guide.max_step)
                if "int" in guide.channels:
                    dl = d * (INT_LOG_SCALE * e)[..., None]
                    g, F = (r * dl).sum(-1), (dl * dl / lam).sum(-1)
                    x[:, 1] += (g / (F + guide.kappa).clamp(min=1e-6)).clamp(-guide.max_step, guide.max_step)
            return x.to(x0.dtype)

        return step


def guidance_from_args(kappa, start_frac):
    """None when guidance is off (kappa None / negative / inf)."""
    return None if kappa is None or kappa < 0 or not np.isfinite(kappa) else (float(kappa), float(start_frac))
