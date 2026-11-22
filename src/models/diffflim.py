"""DiffFLIM v1: conditional diffusion model for lifetime + intensity maps from low-photon TCSPC data.

Pipeline
    counts (256, H, W)  --bin 8x-->  binned (32, H, W)  --CondEncoder-->  features (C, H, W)
    target x0 = [norm log(tau), norm log1p(intensity)]   (2, H, W), each roughly in [-1, 1]
    DDPM (1000 steps, cosine schedule) trains a U-Net, given concat(x_t, features) and the time
    step t, to predict either
      * "v"   (default): v = sqrt(ab) * eps - sqrt(1 - ab) * x0   (Salimans & Ho 2022), or
      * "eps": the noise eps added to x0.
    Why v: with eps-prediction the loss hardly penalises a wrong x0 at high noise levels
    (x0 = (x_t - sqrt(1-ab) eps) / sqrt(ab) and sqrt(ab) ~ 0.01 there), so the eps model learned
    to copy the signal from x_t instead of reading it from the counts, and DDIM sampling from
    pure noise failed (v1_eps run: 48% tau error). At high noise v ~ -x0, so the v-loss forces
    the network to predict the maps from the counts.
    Sampling: DDIM, 50 deterministic steps; different starting noise -> different samples.
    N samples per image -> mean = prediction, std = uncertainty.

Everything is small on purpose (fits an 8 GB laptop GPU): base 64 channels, 3 U-Net levels.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------
# Target normalisation (fixed constants, so the model never sees dataset statistics)
# ----------------------------------------------------------------------------------------
# log(tau): tau in TAU_RANGE ns -> [-1, 1]. v1 (simulate.py data) used (0.5, 5); models store their
# own range in the checkpoint config (DiffFLIM v2 on simulator-v3 data: (0.1, 12)).
DEFAULT_TAU_RANGE = (0.5, 5.0)
TAU_LOG_CENTER = 0.5 * (math.log(0.5) + math.log(5.0))
TAU_LOG_SCALE = 0.5 * (math.log(5.0) - math.log(0.5))


def _tau_consts(tau_range):
    lo, hi = tau_range
    return 0.5 * (math.log(lo) + math.log(hi)), 0.5 * (math.log(hi) - math.log(lo))
# log1p(intensity): intensity in [0, ~3000] photons -> log1p in [0, 8] -> [-1, 1]
INT_LOG_CENTER = 4.0
INT_LOG_SCALE = 4.0

BIN_FACTOR = 8  # 256 time bins -> 32

# DDIM steps used by default for evaluation and calibration (2026-11-22: 10 instead of 50; on the v3
# test set 10 steps gave the same error and, after recalibration, the same coverage at 1/5 of the time).
DEFAULT_SAMPLE_STEPS = 10


def _lib(x):
    return torch if torch.is_tensor(x) else np


def normalize_targets(tau, intensity, tau_range=DEFAULT_TAU_RANGE):
    """tau, intensity (..., H, W) -> (..., 2, H, W) normalised target (numpy or torch)."""
    xp = _lib(tau)
    c, sc = _tau_consts(tau_range)
    y_tau = (xp.log(tau) - c) / sc
    y_int = (xp.log1p(intensity) - INT_LOG_CENTER) / INT_LOG_SCALE
    return xp.stack([y_tau, y_int], -3)


def denormalize_targets(y, tau_range=DEFAULT_TAU_RANGE):
    """(..., 2, H, W) normalised -> (tau, intensity), each (..., H, W), physical units."""
    xp = _lib(y)
    c, sc = _tau_consts(tau_range)
    tau = xp.exp(y[..., 0, :, :] * sc + c)
    intensity = xp.expm1(y[..., 1, :, :] * INT_LOG_SCALE + INT_LOG_CENTER)
    return tau, (intensity.clamp(min=0) if xp is torch else np.maximum(intensity, 0))


def bin_counts(counts, factor=BIN_FACTOR):
    """Sum groups of `factor` time bins: (..., 256, H, W) -> (..., 256/factor, H, W) float32."""
    *lead, T, H, W = counts.shape
    if _lib(counts) is torch:
        return counts.float().reshape(*lead, T // factor, factor, H, W).sum(-3)
    return counts.astype(np.float32).reshape(*lead, T // factor, factor, H, W).sum(-3)


# ----------------------------------------------------------------------------------------
# Network blocks
# ----------------------------------------------------------------------------------------
def _norm(ch):
    return nn.GroupNorm(min(32, ch // 4), ch)


def timestep_embedding(t, dim):
    """Standard sinusoidal embedding of integer time steps t (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv x2 with the time embedding added in between, plus a skip."""

    def __init__(self, cin, cout, temb_dim):
        super().__init__()
        self.block1 = nn.Sequential(_norm(cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1))
        self.temb = nn.Linear(temb_dim, cout)
        self.block2 = nn.Sequential(_norm(cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.block1(x) + self.temb(temb)[:, :, None, None]
        return self.block2(h) + self.skip(x)


class CondEncoder(nn.Module):
    """log1p(binned counts) (32 ch) + log1p(total counts) (1 ch) -> feature map (out_ch, H, W)."""

    def __init__(self, n_bins=256 // BIN_FACTOR, out_ch=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bins + 1, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, out_ch, 3, padding=1),
        )

    def forward(self, binned):
        total = binned.sum(1, keepdim=True)
        x = torch.cat([torch.log1p(binned), torch.log1p(total)], 1) / 4.0   # keep inputs ~[0, 2]
        return self.net(x)


class UNet(nn.Module):
    """3-level U-Net (64 -> 32 -> 16 px at 64x64 input), 2 ResBlocks per level."""

    def __init__(self, in_ch, out_ch=2, base=64, mults=(1, 2, 2)):
        super().__init__()
        chs = [base * m for m in mults]
        temb_dim = base * 4
        self.base = base
        self.temb = nn.Sequential(nn.Linear(base, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim))
        self.inc = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        c = chs[0]
        for i, ch in enumerate(chs):
            self.down.append(nn.ModuleList([ResBlock(c, ch, temb_dim), ResBlock(ch, ch, temb_dim)]))
            c = ch
            self.downsample.append(nn.Conv2d(c, c, 3, stride=2, padding=1) if i < len(chs) - 1 else nn.Identity())

        self.mid = nn.ModuleList([ResBlock(c, c, temb_dim), ResBlock(c, c, temb_dim)])

        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(chs))):
            ch = chs[i]
            self.up.append(nn.ModuleList([ResBlock(c + ch, ch, temb_dim), ResBlock(ch, ch, temb_dim)]))
            c = ch
            self.upsample.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                               nn.Conv2d(c, c, 3, padding=1)) if i > 0 else nn.Identity())

        self.out = nn.Sequential(_norm(c), nn.SiLU(), nn.Conv2d(c, out_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)   # start by predicting eps = 0
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.base))
        h = self.inc(x)
        skips = []
        for blocks, down in zip(self.down, self.downsample):
            for b in blocks:
                h = b(h, temb)
            skips.append(h)
            h = down(h)
        for b in self.mid:
            h = b(h, temb)
        for blocks, up in zip(self.up, self.upsample):
            h = torch.cat([h, skips.pop()], 1)
            for b in blocks:
                h = b(h, temb)
            h = up(h)
        return self.out(h)


# ----------------------------------------------------------------------------------------
# Diffusion model
# ----------------------------------------------------------------------------------------
def cosine_alphas_cumprod(n_steps, s=0.008):
    """Nichol & Dhariwal cosine schedule: alpha_bar(t) for t = 1..n_steps (index 0..n_steps-1)."""
    x = torch.linspace(0, n_steps, n_steps + 1, dtype=torch.float64)
    f = torch.cos(((x / n_steps) + s) / (1 + s) * math.pi / 2) ** 2
    betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0).float()


class DiffFLIM(nn.Module):
    def __init__(self, base=64, cond_ch=64, n_steps=1000, bg_weight=0.1, prediction="v",
                 tau_range=DEFAULT_TAU_RANGE):
        super().__init__()
        assert prediction in ("v", "eps")
        self.tau_range = tuple(float(x) for x in tau_range)
        self.config = {"base": base, "cond_ch": cond_ch, "n_steps": n_steps, "bg_weight": bg_weight,
                       "prediction": prediction, "tau_range": list(self.tau_range)}
        self.prediction = prediction
        self.n_steps = n_steps
        self.bg_weight = bg_weight
        self.encoder = CondEncoder(out_ch=cond_ch)
        self.unet = UNet(in_ch=2 + cond_ch, out_ch=2, base=base)
        self.register_buffer("alphas_cumprod", cosine_alphas_cumprod(n_steps))

    def network(self, x_t, t, features):
        """Raw network output (v or eps, depending on self.prediction)."""
        return self.unet(torch.cat([x_t, features], 1), t)

    def predict_x0_eps(self, x_t, t, features):
        """Network output converted to (x0, eps) estimates. t: (B,) long."""
        ab = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        out = self.network(x_t, t, features).float()
        if self.prediction == "v":
            return ab.sqrt() * x_t - (1 - ab).sqrt() * out, (1 - ab).sqrt() * x_t + ab.sqrt() * out
        return (x_t - (1 - ab).sqrt() * out) / ab.sqrt(), out

    def loss(self, binned, target, mask):
        """MSE on v (or eps); foreground pixels weight 1, others bg_weight."""
        B = target.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=target.device)
        ab = self.alphas_cumprod[t].view(B, 1, 1, 1)
        noise = torch.randn_like(target)
        x_t = ab.sqrt() * target + (1 - ab).sqrt() * noise
        out = self.network(x_t, t, self.encoder(binned)).float()
        goal = ab.sqrt() * noise - (1 - ab).sqrt() * target if self.prediction == "v" else noise
        w = torch.where(mask[:, None], 1.0, self.bg_weight)
        return (w * (out - goal) ** 2).sum() / (w.sum() * target.shape[1])

    @torch.no_grad()
    def ddim_sample(self, binned, n_steps=DEFAULT_SAMPLE_STEPS, generator=None, clip=1.5):
        """Deterministic DDIM (eta = 0). binned (B, 32, H, W) -> normalised x0 (B, 2, H, W)."""
        B, _, H, W = binned.shape
        features = self.encoder(binned)                  # computed once, reused at every step
        x = torch.randn((B, 2, H, W), generator=generator, device=binned.device)
        ts = torch.linspace(self.n_steps - 1, 0, n_steps, device=binned.device).round().long()
        for i, t in enumerate(ts):
            ab = self.alphas_cumprod[t]
            ab_prev = self.alphas_cumprod[ts[i + 1]] if i + 1 < n_steps else torch.tensor(1.0, device=x.device)
            x0, _ = self.predict_x0_eps(x, t.repeat(B), features)
            x0 = x0.clamp(-clip, clip)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()  # noise estimate consistent with clipped x0
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x

    @torch.no_grad()
    def sample(self, counts, n_samples=10, steps=DEFAULT_SAMPLE_STEPS, seed=None, batch=None):
        """counts (256, H, W) numpy/torch -> samples (n_samples, 2, H, W) numpy, physical units:
        channel 0 = tau (ns), channel 1 = intensity (signal photons)."""
        device = self.alphas_cumprod.device
        c = torch.as_tensor(np.asarray(counts, dtype=np.float32), device=device)
        binned = bin_counts(c)[None]
        g = torch.Generator(device=device)
        g.manual_seed(int(seed) if seed is not None else int(torch.randint(0, 2**31, (1,))))
        batch = batch or n_samples
        out = []
        for s in range(0, n_samples, batch):
            k = min(batch, n_samples - s)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                y = self.ddim_sample(binned.expand(k, -1, -1, -1), steps, generator=g)
            tau, inten = denormalize_targets(y.float(), self.tau_range)
            out.append(torch.stack([tau, inten], 1).cpu().numpy())
        return np.concatenate(out)


def load_model(path, device="cuda", use_ema=True):
    """Load a checkpoint written by scripts/train_diffflim.py (EMA weights by default)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = {"prediction": "eps", **ckpt["config"]}   # checkpoints without the key are eps models
    model = DiffFLIM(**config).to(device)
    model.load_state_dict(ckpt["ema" if use_ema else "model"])
    return model.eval(), ckpt
