"""Run every lifetime method on a batch of v3-type images (used by scripts/hallucination_test.py and
scripts/eval_flimngo_protocol.py).

All functions take counts of shape (N, 256, H, W) and return lifetime maps (N, H, W) in ns.
  * classical baselines (MLE, phasor, binned MLE 5 x 5): v3 mono-exponential forward model with
    each image's own IRF (src/baselines/common.py), in worker processes;
  * FLIMngo pretrained (own preprocessing) and FLIMngo retrained on v3 (checkpoints/flimngo_v3.pt),
    both with the recall-oriented mask computed from the counts (`flimngo_recall`), as in the main
    comparison (scripts/eval_compare_v2.py);
  * DiffFLIM: 10 DDIM steps x 10 samples, seeds seed0 + i; returns mean, standard deviation and the
    first sample of lifetime and intensity.
"""
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from src.baselines import flimngo_baseline as pre
from src.baselines.common import fit_all
from src.evaluation.calibration import foreground_mask
from src.paths import CHECKPOINTS

FLIMNGO_MASK = "flimngo_recall"


def _fit_image(args):
    counts, bg, meta, irf, methods = args
    return fit_all(counts, bg, meta, irf, methods)


def run_baselines(counts, bg, irfs, meta, methods=("MLE", "BinMLE5"), workers=12):
    """{method: (tau (N, H, W), intensity (N, H, W))}."""
    jobs = [(counts[i], float(bg[i]), meta, irfs[i], tuple(methods)) for i in range(len(counts))]
    with ProcessPoolExecutor(workers) as pool:
        res = list(pool.map(_fit_image, jobs, chunksize=2))
    return {m: (np.stack([r[m][0] for r in res]).astype(np.float32), np.stack([r[m][1] for r in res]).astype(np.float32))
            for m in methods}


def flimngo_masks(counts):
    return np.stack([foreground_mask(c.sum(0), FLIMNGO_MASK) for c in counts])


def run_flimngo_pretrained(counts, bin_width, masks=None):
    masks = flimngo_masks(counts) if masks is None else masks
    return pre.predict(counts, masks, bin_width=bin_width)[0]


def load_flimngo_retrained(path=CHECKPOINTS / "flimngo_v3.pt"):
    from src.models.flimngo_arch import build_flimngo
    ck = torch.load(path, map_location="cuda", weights_only=False)
    net = build_flimngo(ck["arch"]["image_size"], ck["arch"]["bin_width"]).cuda()
    net.load_state_dict(ck["model"])
    return net.eval()


@torch.no_grad()
def run_flimngo_retrained(counts, net=None, masks=None, batch=8):
    from src.models.flimngo_arch import predict as predict_arch
    net = load_flimngo_retrained() if net is None else net
    masks = flimngo_masks(counts) if masks is None else masks
    out = np.zeros((len(counts),) + counts.shape[-2:], np.float32)
    for i in range(0, len(counts), batch):
        c = torch.from_numpy(counts[i:i + batch].astype(np.float32)).cuda()
        m = torch.from_numpy(masks[i:i + batch]).cuda()
        out[i:i + batch] = predict_arch(net, c, m).float().cpu().numpy()
    return out


def load_diffflim(ckpt_name):
    from src.models.diffflim import load_model
    return load_model(CHECKPOINTS / ckpt_name, "cuda")[0]


def run_diffflim(model, counts, seed0=1000, steps=10, n_samples=10):
    """{tau_mean, tau_std, tau_s0, int_mean, int_std, int_s0}, each (N, H, W)."""
    keys = ("tau_mean", "tau_std", "tau_s0", "int_mean", "int_std", "int_s0")
    out = {k: [] for k in keys}
    t0 = time.perf_counter()
    for i in range(len(counts)):
        s = model.sample(counts[i], n_samples=n_samples, steps=steps, seed=seed0 + i)
        for j, q in enumerate(("tau", "int")):
            out[f"{q}_mean"].append(s[:, j].mean(0))
            out[f"{q}_std"].append(s[:, j].std(0, ddof=1))
            out[f"{q}_s0"].append(s[0, j])
    out = {k: np.stack(v).astype(np.float32) for k, v in out.items()}
    out["sec_per_image"] = (time.perf_counter() - t0) / max(len(counts), 1)
    return out
