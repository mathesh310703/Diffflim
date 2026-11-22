"""Check that the FLIMngo wrapper reproduces FLIMngo's own demo (needs external/FLIMngo).

demo_notebooks/predict_simulated.ipynb stores the MSE between prediction and ground truth for
three simulated samples; the wrapper must give the same values on the same files.

Run:  python tests/test_flimngo.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines.flimngo_baseline import FLIMNGO_DIR, predict  # noqa: E402

SAMPLES = FLIMNGO_DIR / "sample_data" / "simulated_data"
# (sample, photons) -> MSE printed in their notebook
REFERENCE = {("sample1", 10): 0.360, ("sample1", 100): 0.157, ("sample2", 10): 0.131,
             ("sample2", 100): 0.081, ("sample3", 10): 0.183, ("sample3", 100): 0.090}


def _read_tiff(path):
    return np.stack([np.array(p, dtype=np.float32) for p in ImageSequence.Iterator(Image.open(path))])


def test_reproduces_demo_mse():
    for (s, p), ref in REFERENCE.items():
        gt = _read_tiff(SAMPLES / f"{s}_ground_truth.tiff")[0]
        x = _read_tiff(SAMPLES / f"{s}_photon_counts_{p}.tiff")
        tau, _ = predict(x[None], bin_width=0.09765625, batch=1)
        mse = float(np.mean((np.nan_to_num(tau[0]) - np.nan_to_num(gt)) ** 2))
        print(f"  {s} {p:3d} photons: MSE {mse:.3f} (notebook {ref:.3f})")
        assert abs(mse - ref) < 0.005


def test_accepts_64():
    x = np.random.default_rng(0).poisson(0.2, (2, 256, 64, 64)).astype(np.float32)
    tau, _ = predict(x)
    assert tau.shape == (2, 64, 64) and np.all(np.isfinite(tau))


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        print(fn.__name__)
        fn()
    print(f"All {len(tests)} tests passed.")
