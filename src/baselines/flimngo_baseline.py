"""FLIMngo pretrained baseline (lifetime only).

Source: S. Kapsiani, N. F. Läubli, E. N. Ward, A. Fernandez-Villegas, B. Mazumder, C. F. Kaminski,
G. S. Kaminski Schierle, "Deep learning for fluorescence lifetime predictions enables
high-throughput in vivo imaging", J. Am. Chem. Soc. (2025), doi:10.1021/jacs.5c03749.
Code and weights: https://github.com/SofiaKapsiani/FLIMngo (local read-only copy in
external/FLIMngo; network class from backbone/flimngo_net.py, weights
flimngo_pretrained_v13102024.pth). Nothing is copied: the network class is imported from the
local copy, and the preprocessing below re-implements utils/eval_utils.py::normalise.

Input conventions, taken from their README, demo notebooks and
data_simulation/notebook3_lifetime_simulation.ipynb:
  - 256 time bins, layout (T, Y, X); bin width passed to the network (output is multiplied by
    bin_width / 0.09765625).
  - each pixel's decay is divided by its own maximum over time (normalise(); their training data
    were saved with exactly this normalisation, decay_noisy / decay_noisy.max());
  - background pixels are exactly zero ("the background should be masked");
  - square images with even size; the network has no IRF input.
The network accepts 64 x 64 directly: `imageSize` only sets the decoder's resize targets
(64/4, 64/2, 64), which match the three stride-2 stages of the encoder.

Preprocessing options (for the sensitivity analysis):
  mask:      "gt"   -> zero every pixel outside the ground-truth foreground (as in their simulated data)
             "otsu" -> zero pixels below an Otsu threshold on the log summed counts (photon-count only)
             "none" -> keep the background counts
  normalise: True (default, as in their pipeline) / False (raw counts)
  pad_to:    None (run at the crop size) or e.g. 256 (zero-pad the crop to 256 x 256, as trained)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FLIMNGO_DIR = ROOT / "external" / "FLIMngo"
WEIGHTS = FLIMNGO_DIR / "flimngo_pretrained_v13102024.pth"
DEFAULT_BIN_WIDTH = 0.09765625   # bin width of FLIMngo's training data (ns)


def _flimngo_class():
    if str(FLIMNGO_DIR) not in sys.path:
        sys.path.insert(0, str(FLIMNGO_DIR))
    from backbone.flimngo_net import FLIMngo   # noqa: E402  (read-only import)
    return FLIMngo


class _Opt:
    def __init__(self, image_size, bin_width):
        self.model = "flimngo"
        self.cpu = False
        self.imageSize = image_size
        self.n_in_channels = 256
        self.width_multiple = 0.5
        self.bin_width = bin_width


_CACHE = {}


def load_flimngo(image_size=64, bin_width=DEFAULT_BIN_WIDTH, device=None):
    """FLIMngo network with pretrained weights, eval mode. Cached per (size, bin width, device)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (image_size, float(bin_width), str(device))
    if key not in _CACHE:
        net = _flimngo_class()(_Opt(image_size, bin_width))
        net.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
        _CACHE[key] = net.to(device).eval()
    return _CACHE[key]


def normalise(counts):
    """Per-pixel max normalisation along time, as utils/eval_utils.py::normalise. counts (T, H, W)."""
    mx = counts.max(0, keepdims=True)
    mx[mx == 0] = 1
    return counts / mx


def otsu_mask(counts):
    from src.evaluation.calibration import _otsu_threshold
    s = np.log1p(counts.sum(0).astype(np.float64))
    return s > _otsu_threshold(s.ravel())


def preprocess(counts, mask=None, normalise_decays=True, pad_to=None):
    """counts (T, H, W) -> float32 (T, S, S) network input."""
    x = counts.astype(np.float32)
    if mask is not None:
        x = np.where(mask[None], x, 0.0)
    if normalise_decays:
        x = normalise(x)
    if pad_to and pad_to > x.shape[-1]:
        p = pad_to - x.shape[-1]
        x = np.pad(x, ((0, 0), (p // 2, p - p // 2), (p // 2, p - p // 2)))
    return x


@torch.no_grad()
def predict(counts, mask=None, bin_width=DEFAULT_BIN_WIDTH, normalise_decays=True, pad_to=None,
            batch=8, device=None):
    """Lifetime maps (N, H, W) in ns for counts (N, T, H, W).

    mask: (N, H, W) bool or None (pixels outside are set to zero before the network).
    Returns (tau, seconds_per_image) where the time covers only the network forward pass
    (preprocessing on the CPU excluded), with the GPU synchronised.
    """
    n, t, h, w = counts.shape
    assert t == 256 and h == w and h % 2 == 0, "FLIMngo needs 256 time bins and an even square image"
    size = pad_to if pad_to and pad_to > h else h
    net = load_flimngo(size, bin_width, device)
    dev = next(net.parameters()).device
    out = np.zeros((n, h, w), np.float32)
    elapsed = 0.0
    off = (size - h) // 2
    for i in range(0, n, batch):
        x = np.stack([preprocess(counts[j], None if mask is None else mask[j], normalise_decays, pad_to)
                      for j in range(i, min(i + batch, n))])
        xt = torch.from_numpy(x).to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = net(xt)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0
        y = y.reshape(len(x), size, size).float().cpu().numpy()
        out[i:i + len(x)] = y[:, off:off + h, off:off + w]
    return out, elapsed / n
