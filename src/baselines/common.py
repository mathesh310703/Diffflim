"""Run the classical baselines on one image, with the forward model that matches the data set.

  * simulate.py data (v1/v2): Gaussian-IRF model of simulate.decay_curves, tau bounds 0.1-8 ns.
  * simulate_v3 data: the image's own (jittered) IRF and the v3 point-sampled model, as a
    MONO-exponential (a simplification: v3 pixels have up to 4 components), tau bounds 0.05-15 ns;
    the phasor uses the exact inversion for point-sampled decays.

`fit_all` takes only plain arrays/dicts, so it can be sent to worker processes.
"""
from src.baselines.binned import fit_binned_mle
from src.baselines.mle import TAU_BOUNDS, fit_mle
from src.baselines.phasor import fit_phasor
from src.data.simulate import config_from_metadata

METHODS = ("MLE", "Phasor", "BinMLE3", "BinMLE5")
TAU_BOUNDS_V3 = (0.05, 15.0)


def forward_setup(meta, irf=None):
    """meta: dataset metadata dict; irf: the image's IRF (v3 only)."""
    cfg = config_from_metadata(meta)
    if meta.get("simulator") == "v3":
        from src.data.simulate_v3 import SimConfigV3, decay_curves_v3
        # periodic: FLUTE configuration (2026-11-23), decays of earlier pulses wrap into the window
        cfg3 = SimConfigV3(size=meta["size"], n_bins=meta["n_bins"], bin_width_ns=meta["bin_width_ns"],
                           periodic=bool(meta.get("periodic", False)))
        return cfg, {"decay_fn": lambda tau: decay_curves_v3(tau, irf, cfg3), "irf": irf,
                     "tau_bounds": TAU_BOUNDS_V3, "point_sampled": True}
    return cfg, {"decay_fn": None, "irf": None, "tau_bounds": TAU_BOUNDS, "point_sampled": False}


def fit_all(counts, bg_per_bin, meta, irf=None, methods=METHODS):
    """Returns {method: (tau (H, W), intensity (H, W))}."""
    cfg, fwd = forward_setup(meta, irf)
    kw = {"tau_bounds": fwd["tau_bounds"], "decay_fn": fwd["decay_fn"]}
    out = {}
    for m in methods:
        if m == "MLE":
            out[m] = fit_mle(counts, cfg, bg_per_bin, **kw)
        elif m == "Phasor":
            out[m] = fit_phasor(counts, cfg, bg_per_bin, irf=fwd["irf"], point_sampled=fwd["point_sampled"])[:2]
        elif m.startswith("BinMLE"):
            out[m] = fit_binned_mle(counts, cfg, bg_per_bin, k=int(m[len("BinMLE"):]), **kw)
        else:
            raise ValueError(m)
    return out
