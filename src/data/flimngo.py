"""Loader for the FLIMngo simulated dataset (.h5 files)."""
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# FLIMngo constants (from the dataset card)
N_TIME_BINS = 256
BIN_WIDTH_NS = 0.0977
LASER_MHZ = 40

RAW_KEY = "tpsf"       # photon decay data, shape (t, x, y)
LABEL_KEY = "tau_maps"  # ground-truth lifetime map in ns, shape (x, y)


def load_h5(path):
    """Return (decay, tau) as numpy arrays: decay (t, x, y), tau (x, y)."""
    with h5py.File(path, "r") as f:
        decay = np.asarray(f[RAW_KEY], dtype=np.float32)
        tau = np.asarray(f[LABEL_KEY], dtype=np.float32)
    return decay, tau


def read_split(split_file):
    """Read a split file (one .h5 path per line)."""
    return [Path(p) for p in Path(split_file).read_text().split() if p]


class FLIMngoDataset(Dataset):
    """Each item: decay (T, H, W) tensor with time bins as channels, tau (1, H, W) tensor."""

    def __init__(self, files):
        self.files = list(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        decay, tau = load_h5(self.files[i])
        # FLIMngo's own loader permutes x/y; we keep the same orientation
        decay = torch.from_numpy(decay).permute(0, 2, 1)
        tau = torch.from_numpy(tau).permute(1, 0).unsqueeze(0)
        return decay, tau
