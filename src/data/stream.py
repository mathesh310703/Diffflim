"""PyTorch IterableDataset that streams freshly simulated training samples (nothing saved).

Each item: {"binned": (32, H, W) float32 time-binned counts,
            "target": (2, H, W) float32 normalised [log tau, log1p intensity],
            "mask":   (H, W) bool foreground}
Every DataLoader worker gets its own seed, so workers never produce the same samples.

simulator="v2": src/data/simulate.py (blob cells). simulator="v3": src/data/simulate_v3.py
(HPA crops, FLIMngo lifetimes); the training IDs are computed once in the main process and
passed in, so all workers use the same list and never touch test or calibration IDs. In v3 the
lifetime is undefined outside the foreground (stored as 0); the target there is set to the centre
of the normalised range (0), and the loss weights those pixels with bg_weight only.
"""
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from src.models.diffflim import DEFAULT_TAU_RANGE, bin_counts, normalize_targets


class SimStream(IterableDataset):
    def __init__(self, cfg=None, photon_range=(10, 1000), seed=0, simulator="v2", ids=None,
                 tau_range=DEFAULT_TAU_RANGE):
        self.cfg = cfg
        self.photon_range = photon_range
        self.seed = seed
        self.simulator = simulator
        self.ids = None if ids is None else list(ids)
        self.tau_range = tuple(tau_range)

    def __iter__(self):
        info = get_worker_info()
        worker = info.id if info is not None else 0
        seed = self.seed * 1000 + worker
        if self.simulator == "v3":
            from src.data import simulate_v3 as v3
            gen = v3.training_samples(self.cfg, self.photon_range, seed=seed, ids=self.ids)
        else:
            from src.data.simulate import training_samples
            gen = training_samples(self.cfg, self.photon_range, seed=seed)
        geo_centre = float(np.sqrt(self.tau_range[0] * self.tau_range[1]))
        for s in gen:
            tau = np.where(s["mask"], s["tau_gt"], geo_centre) if self.simulator == "v3" else s["tau_gt"]
            yield {
                "binned": bin_counts(s["counts"]),
                "target": normalize_targets(tau, s["intensity_gt"], self.tau_range).astype(np.float32),
                "mask": s["mask"],
            }
