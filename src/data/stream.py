"""PyTorch IterableDataset that streams freshly simulated training samples (nothing saved).

Each item: {"binned": (32, H, W) float32 time-binned counts,
            "target": (2, H, W) float32 normalised [log tau, log1p intensity],
            "mask":   (H, W) bool foreground}
Every DataLoader worker gets its own seed, so workers never produce the same samples.
"""
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from src.data.simulate import SimConfig, training_samples
from src.models.diffflim import bin_counts, normalize_targets


class SimStream(IterableDataset):
    def __init__(self, cfg=None, photon_range=(10, 1000), seed=0):
        self.cfg = SimConfig() if cfg is None else cfg
        self.photon_range = photon_range
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        worker = info.id if info is not None else 0
        for s in training_samples(self.cfg, self.photon_range, seed=self.seed * 1000 + worker):
            yield {
                "binned": bin_counts(s["counts"]),
                "target": normalize_targets(s["tau_gt"], s["intensity_gt"]).astype(np.float32),
                "mask": s["mask"],
            }
