"""Phasor analysis: per-pixel phase lifetime and intensity.

For a histogram f(t) the phasor at angular frequency omega = 2*pi / period is
    g = sum_t f(t) cos(omega t) / sum_t f(t),     s = sum_t f(t) sin(omega t) / sum_t f(t).
The period is the laser repetition period = the 25 ns TCSPC window (40 MHz).

Steps:
  1. Background subtraction: f = counts - b (b = expected background counts per bin).
  2. IRF correction: the measured phasor is (decay phasor) x (IRF phasor) as complex numbers,
     so divide by the IRF phasor (computed from the bin-integrated IRF, as you would from a
     measured IRF).
  3. Phase lifetime of a mono-exponential:  tau = s / (g * omega).
Intensity = total counts - b * T.

Not corrected (small, but known): the simulator truncates the decay at the end of the window
instead of wrapping it around; for tau = 5 ns this shifts the phase lifetime by well under 1%.
At low photon counts g can be ~0 or negative, which gives huge or negative tau; those values are
kept as they are (they are real phasor failures), only g = 0 or no net signal gives NaN.
"""
import numpy as np

from src.data.simulate import irf_curve


def phasor_coordinates(f, cfg):
    """Phasor (g, s) of histograms f with time on axis 0. Returns arrays of shape f.shape[1:]."""
    omega = 2 * np.pi / cfg.period_ns
    t = cfg.time_centers.reshape(-1, *([1] * (f.ndim - 1)))
    total = f.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (f * np.cos(omega * t)).sum(axis=0) / total
        s = (f * np.sin(omega * t)).sum(axis=0) / total
    return g, s


def fit_phasor(counts, cfg, bg_per_bin, irf=None):
    """counts: (T, H, W). Returns tau (H, W) ns, intensity (H, W) photons, and g, s (corrected)."""
    f = counts.astype(np.float64) - bg_per_bin                   # background subtraction
    g, s = phasor_coordinates(f, cfg)

    irf = irf_curve(cfg) if irf is None else irf
    g_irf, s_irf = phasor_coordinates(irf[:, None], cfg)
    omega = 2 * np.pi / cfg.period_ns
    # (errstate: pixels with zero net signal give NaN/inf here; they end up as tau = NaN)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (g + 1j * s) / (g_irf[0] + 1j * s_irf[0])            # IRF correction (complex division)
        g, s = z.real, z.imag
        tau = s / (g * omega)
    tau[~np.isfinite(tau)] = np.nan
    intensity = f.sum(axis=0)
    return tau, intensity, g, s
