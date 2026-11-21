"""Sanity tests for the DiffFLIM model (fast; uses the GPU if available).

Run:  python tests/test_diffflim.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.simulate import SimConfig, simulate_sample  # noqa: E402
from src.models.diffflim import (  # noqa: E402
    DiffFLIM, bin_counts, cosine_alphas_cumprod, denormalize_targets, normalize_targets,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_normalisation_roundtrip():
    tau = np.array([[0.5, 1.0], [2.5, 5.0]], dtype=np.float32)
    inten = np.array([[0.0, 10.0], [100.0, 3000.0]], dtype=np.float32)
    y = normalize_targets(tau, inten)
    assert y.shape == (2, 2, 2)
    assert np.isclose(y[0].min(), -1) and np.isclose(y[0].max(), 1)      # tau range -> [-1, 1]
    assert y[1].min() >= -1 and y[1].max() <= 1.01
    t2, i2 = denormalize_targets(torch.as_tensor(y))
    assert np.allclose(t2.numpy(), tau, rtol=1e-5) and np.allclose(i2.numpy(), inten, rtol=1e-4, atol=1e-4)


def test_binning():
    c = np.random.default_rng(0).poisson(1.0, (256, 4, 4)).astype(np.uint16)
    b = bin_counts(c)
    assert b.shape == (32, 4, 4) and np.allclose(b.sum(0), c.sum(0))
    assert np.allclose(bin_counts(torch.as_tensor(c.astype(np.int32))).numpy(), b)


def test_schedule():
    ab = cosine_alphas_cumprod(1000)
    assert ab.shape == (1000,) and torch.all(ab[1:] < ab[:-1]) and 0.99 < ab[0] <= 1 and ab[-1] < 1e-3


def test_forward_and_overfit():
    """Loss is finite and goes down when overfitting one small batch; sampling gives finite maps."""
    torch.manual_seed(0)
    cfg = SimConfig(size=32)
    s = [simulate_sample(np.random.default_rng(i), cfg, 200) for i in range(4)]
    binned = torch.as_tensor(np.stack([bin_counts(x["counts"]) for x in s]), device=DEVICE)
    target = torch.as_tensor(np.stack([normalize_targets(x["tau_gt"], x["intensity_gt"]) for x in s]),
                             device=DEVICE)
    mask = torch.as_tensor(np.stack([x["mask"] for x in s]), device=DEVICE)
    model = DiffFLIM(base=32, cond_ch=32).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(150):
        loss = model.loss(binned, target, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"  loss {np.mean(losses[:10]):.3f} -> {np.mean(losses[-10:]):.3f}")
    assert np.isfinite(losses).all() and np.mean(losses[-10:]) < 0.5 * np.mean(losses[:10])

    out = model.eval().sample(s[0]["counts"], n_samples=3, steps=10, seed=0)
    assert out.shape == (3, 2, 32, 32) and np.isfinite(out).all() and (out[:, 0] > 0).all()
    again = model.sample(s[0]["counts"], n_samples=3, steps=10, seed=0)
    assert np.allclose(out, again, rtol=1e-3, atol=1e-3)                 # seeded -> reproducible


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        print(fn.__name__)
        fn()
    print(f"All {len(tests)} tests passed.")
