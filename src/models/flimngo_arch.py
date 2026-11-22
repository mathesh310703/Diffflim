"""FLIMngo network architecture, randomly initialised, for training on our own data.

Source: S. Kapsiani, N. F. Läubli, E. N. Ward, A. Fernandez-Villegas, B. Mazumder, C. F. Kaminski,
G. S. Kaminski Schierle, "Deep learning for fluorescence lifetime predictions enables
high-throughput in vivo imaging", J. Am. Chem. Soc. (2025), doi:10.1021/jacs.5c03749.
Code: https://github.com/SofiaKapsiani/FLIMngo (local read-only copy in external/FLIMngo).
The network class `FLIMngo` is imported unchanged from external/FLIMngo/backbone/flimngo_net.py
(YOLOv5-style encoder/decoder, width multiple 0.5, 256 input channels = time bins, one output
channel = lifetime in ns after a ReLU, scaled by bin_width / 0.09765625).

Training recipe from their train.py: MSE loss between the predicted and true lifetime map in ns
over ALL pixels (background target 0, input 0), AdamW with lr 0.01 and weight decay 1e-8,
batch size 10, constant learning rate, model selection on a validation set (early stopping).
Input preprocessing from their notebooks (utils/eval_utils.py::normalise): background pixels set
to zero, each pixel's decay divided by its maximum over time.

No intensity head: the per-pixel max normalisation removes the absolute photon count from the
input, so an intensity output would need an extra input channel, i.e. a change of FLIMngo's
preprocessing. The model is therefore kept lifetime-only, as published.
"""
import torch

from src.baselines.flimngo_baseline import DEFAULT_BIN_WIDTH, _flimngo_class, _Opt


def build_flimngo(image_size=64, bin_width=DEFAULT_BIN_WIDTH, seed=0):
    """FLIMngo network with fresh random weights (PyTorch default initialisation, as in their code)."""
    torch.manual_seed(seed)
    return _flimngo_class()(_Opt(image_size, bin_width))


def normalise_batch(counts, mask):
    """GPU version of FLIMngo's preprocessing. counts (B, 256, H, W) float, mask (B, H, W) bool
    -> background zeroed, each pixel divided by its maximum over time."""
    x = counts * mask[:, None].to(counts.dtype)
    mx = x.amax(1, keepdim=True)
    return x / torch.where(mx > 0, mx, torch.ones_like(mx))


def predict(net, counts, mask):
    """counts (B, 256, H, W), mask (B, H, W) -> lifetime (B, H, W) in ns."""
    y = net(normalise_batch(counts, mask))
    return y.reshape(counts.shape[0], *counts.shape[-2:])
