"""Simulator v3: FLIMngo-style semi-synthetic TCSPC data from HPA images, with REAL photon counts.

Follows external/FLIMngo/data_simulation (README, utils_simulation.py, notebooks 1-3), with the
differences listed at the end of this docstring.

Per sample (64 x 64 by default):
  1. Structure: a random crop of a Human Protein Atlas sample (4 channels: red = microtubules,
     green = protein, blue = nucleus, yellow = ER), random 90-degree rotation and flip. Only
     samples from the given ID list are used (train or test, see hpa_splits()).
  2. Foreground mask (FLIMngo): pixels whose 4-channel sum is > 0. Crops with less than
     min_fg_frac foreground are redrawn (FLIMngo: 5 %).
  3. Lifetimes (FLIMngo `tau_channel`): each channel c gets a base lifetime U(0.1, 10) ns and a
     spread U(0.2, 2) ns; tau_c(pixel) = base + U(0, 1) * spread, independently per pixel.
     Channel c contributes to a pixel only where its own channel value is > 0.
  4. Amplitudes (FLIMngo `calculate_ratio`): Perlin noise p(x, y) in [0, 1] (3-5 octaves,
     persistence 0.3-0.9). The k-th contributing channel (in the order R, G, B, Y) gets amplitude
     p (1 - p)^k, the last one the remainder (1 - p)^(n-1); the amplitudes sum to 1.
  5. Decay: sum_c a_c exp(-t / tau_c), point-sampled at t_k = k * dt (k = 0..255), discretely
     convolved with the IRF, truncated to the 256 bins and normalised to sum 1.
  6. Intensity (FLIMngo): the 4-channel sum, min-max normalised over the crop, mapped linearly to
     [min, min + delta] photons with min ~ U(10, 60), delta ~ U(10, 40) (FLIMngo "low" setting),
     0 outside the mask; then rescaled so that the mean over the foreground equals `photons`
     (photon level = mean signal photons per foreground pixel, as in simulate.py).
  7. Counts: intensity * decay + constant background (bg_fraction * photons / T per bin, as in
     simulate.py; FLIMngo has no background), Poisson-sampled to uint16.

Ground truth
  tau_gt       AMPLITUDE-weighted mean lifetime sum_c a_c tau_c, exactly as FLIMngo
               (notebook 3: `ground_truth_tau = sum(ratios * tau_values)`). 0 outside the mask.
  tau_int_gt   intensity-weighted mean lifetime sum a tau^2 / sum a tau (extra, for analysis;
               this is what the mean photon arrival time measures).
  intensity_gt expected signal photons per pixel (background excluded). 0 outside the mask.
  mask         foreground (4-channel sum > 0).

IRFs (FLIMngo notebook 2): all 12 of the FLIMngo library, in the same order: 6 simulated
skew-normal IRFs (with/without secondary peak; FWHM about 200, 400, 800 ps) and 6 experimental
IRFs (in-house PMT (.sdt, read with `sdtfile`), DLT, DL4FLI, FPFLI (MATLAB v7.3, read with `h5py`),
in-house APD, flimfit). The library is fixed in IRF_LIBRARY so that data sets stay reproducible. Laser jitter: the IRF is shifted to earlier times by
1 + U{0..early-1} bins or to later times by U{0..late-1} bins (50/50), as in notebook 3. Each IRF
is normalised to sum 1 after shifting. The IRF used is stored with every sample.

Differences to FLIMngo
  * 64 x 64 random crops (FLIMngo: 256 x 256 sliding-window crops), plus rotation/flip.
  * Real photon counts are kept (FLIMngo normalises each decay to max 1), intensity is not
    rounded, a small constant background is added, and the photon level is set per foreground
    pixel.
  * IRF_DLT is resampled physically (5 ps channels -> 0.0977 ns bins, first 25 ns); FLIMngo
    averages all 10000 channels into 256 bins. The other experimental IRFs already have 256
    points and are used bin for bin, as in FLIMngo.
  * Perlin noise uses scipy.ndimage.zoom (bilinear) instead of skimage.transform.resize.
  * In the simulated IRFs the secondary peak uses `sigma` (not `sigma_s`), as in the FLIMngo code.

Discretisation note: because the decay is point-sampled at the start of each bin, the mean
arrival time of a mono-exponential is dt / (exp(dt / tau) - 1) ~ tau - dt/2 (about 0.05 ns less
than tau), plus the IRF mean; this is a property of the FLIMngo model and is checked in the tests.
"""
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.ndimage import zoom
from scipy.stats import norm, skewnorm

from src.data.simulate import sample_counts, save_dataset
from src.paths import RAW, RAW_OWN, ROOT, SPLITS

HPA_DIR = RAW / "hpa" / "train"
CHANNELS = ("red", "green", "blue", "yellow")
IRF_DIR = ROOT / "external" / "FLIMngo" / "data_simulation" / "data" / "irfs"

TEST_IDS_FILE = SPLITS / "hpa_test.txt"
TRAIN_IDS_FILE = SPLITS / "hpa_train.txt"
CALIB_IDS_FILE = SPLITS / "hpa_calib.txt"
# Frozen record of the 246 IDs DiffFLIM v2 was trained on (copy of hpa_train.txt as written at the
# start of its run, 2026-11-21 21:31), and the complete samples downloaded later that v2 never saw.
# The latter are used as EXTRA CALIBRATION cells and are excluded from train_ids() by default;
# a future model may only train on them if this is stated in the project log.
TRAIN_IDS_V2_FILE = SPLITS / "hpa_train_diffflim_v2.txt"
CALIB_EXTRA_V2_FILE = SPLITS / "hpa_calib_extra_v2.txt"
N_TEST_IDS = 30
N_CALIB_IDS = 20
SPLIT_SEED = 2026
CALIB_SEED = 2027

TEST_PHOTON_LEVELS = (10, 20, 50, 100, 500, 1000)
TEST_SEED = 31000        # scenes of the paired test set (noise: TEST_SEED + level)
CALIB_SET_SEED = 41000   # scenes of the paired calibration set (noise: CALIB_SET_SEED + level)
CALIB_EXTRA_SET_SEED = 51000   # scenes of the extra paired calibration set (unseen-by-v2 cells)

# (id, source, parameters, laser jitter (early, late) in bins) -- FLIMngo notebook 2
IRF_LIBRARY = (
    ("IRF_simulated_s", "sim", dict(sigma=0.3, loc=2, skewness=4, loc_secondary=3.2, ratio=0.025), (8, 25)),
    ("IRF_simulated", "sim", dict(sigma=0.3, loc=2, skewness=4, loc_secondary=3.2, ratio=0.0), (8, 25)),
    ("IRF_simulated_broad_s", "sim", dict(sigma=0.6, loc=2, skewness=4, loc_secondary=4.1, ratio=0.05), (8, 25)),
    ("IRF_simulated_broad", "sim", dict(sigma=0.6, loc=2, skewness=4, loc_secondary=4.1, ratio=0.0), (8, 25)),
    ("IRF_simulated_narrow_s", "sim", dict(sigma=0.15, loc=2, skewness=3, loc_secondary=2.7, ratio=0.025), (8, 25)),
    ("IRF_simulated_narrow", "sim", dict(sigma=0.15, loc=2, skewness=3, loc_secondary=2.7, ratio=0.0), (8, 25)),
    ("IRF_PMT", "file", "IRF_inhouse_PMT.sdt", (8, 25)),
    ("IRF_DLT", "file", "IRF_DLT.dat", (8, 25)),
    ("IRF_DL4FLI", "file", "IRF_DL4FLI.mat", (8, 25)),
    ("IRF_FPFLI", "file", "IRF_FPFLI.mat", (8, 25)),
    ("IRF_APD", "file", "IRF_inhouse_APD.csv", (35, 3)),
    ("IRF_flimfit", "file", "IRF_flimfit_py.txt", (35, 3)),
)
EXCLUDED_IRFS = {}   # all FLIMngo IRFs are used


@dataclass
class SimConfigV3:
    size: int = 64
    n_bins: int = 256
    bin_width_ns: float = 0.0977
    tau_base_range: tuple = (0.1, 10.0)      # FLIMngo tau_range
    tau_spread_range: tuple = (0.2, 2.0)     # FLIMngo tau_dist
    min_photons_range: tuple = (10.0, 60.0)  # FLIMngo "low": photon_range_min (sets the contrast only)
    delta_photons_range: tuple = (10.0, 40.0)  # FLIMngo "low": photon_range_max
    perlin_octaves: tuple = (3, 5)           # inclusive
    perlin_persistence: tuple = (0.3, 0.9)
    bg_fraction: float = 0.02                # as simulate.py
    min_fg_frac: float = 0.05                # FLIMngo: > 5 % non-background pixels
    max_crop_tries: int = 50

    @property
    def time_starts(self):
        """t_k = k * dt, where FLIMngo samples the exponential."""
        return np.arange(self.n_bins) * self.bin_width_ns

    @property
    def time_centers(self):
        return (np.arange(self.n_bins) + 0.5) * self.bin_width_ns

    @property
    def period_ns(self):
        return self.n_bins * self.bin_width_ns


# ----------------------------------------------------------------------------------------
# HPA images and the fixed train/test split
# ----------------------------------------------------------------------------------------
def _sample_ids(hpa_dir):
    by_id = {}
    for p in Path(hpa_dir).glob("*.png"):
        sid, _, ch = p.stem.rpartition("_")
        by_id.setdefault(sid, set()).add(ch)
    return by_id


def complete_ids(hpa_dir=HPA_DIR, verify=True):
    """IDs with all 4 channel PNGs present (and fully decodable if verify)."""
    out = []
    for sid, chans in sorted(_sample_ids(hpa_dir).items()):
        if not set(CHANNELS) <= chans:
            continue
        if verify:
            try:
                for c in CHANNELS:
                    with Image.open(Path(hpa_dir) / f"{sid}_{c}.png") as im:
                        im.load()
            except Exception:
                continue
        out.append(sid)
    return out


def _read_ids(path):
    return [s for s in Path(path).read_text().split() if s]


def test_ids(hpa_dir=HPA_DIR):
    """The FIXED list of 30 test sample IDs. Created once (seed SPLIT_SEED), never rewritten."""
    if TEST_IDS_FILE.exists():
        return _read_ids(TEST_IDS_FILE)
    ids = complete_ids(hpa_dir)
    if len(ids) < 2 * N_TEST_IDS:
        raise RuntimeError(f"Only {len(ids)} complete HPA samples; need at least {2 * N_TEST_IDS}.")
    chosen = sorted(np.random.default_rng(SPLIT_SEED).choice(ids, N_TEST_IDS, replace=False).tolist())
    TEST_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TEST_IDS_FILE.write_text("\n".join(chosen) + "\n")
    return chosen


def calib_ids(hpa_dir=HPA_DIR):
    """The FIXED list of N_CALIB_IDS calibration sample IDs, drawn once (seed CALIB_SEED) from the
    training IDs of that moment, written to data/splits/hpa_calib.txt and never rewritten.
    Calibration IDs are excluded from training."""
    if CALIB_IDS_FILE.exists():
        return _read_ids(CALIB_IDS_FILE)
    test = set(test_ids(hpa_dir))
    pool = [i for i in complete_ids(hpa_dir) if i not in test]
    chosen = sorted(np.random.default_rng(CALIB_SEED).choice(pool, N_CALIB_IDS, replace=False).tolist())
    CALIB_IDS_FILE.write_text("\n".join(chosen) + "\n")
    return chosen


def diffflim_v2_train_ids():
    """The 246 training IDs DiffFLIM v2 was trained on (frozen record, never rewritten)."""
    return _read_ids(TRAIN_IDS_V2_FILE)


def calib_extra_v2_ids(hpa_dir=HPA_DIR):
    """Complete samples that DiffFLIM v2 never saw (not in its training list) and that are neither
    test nor calibration IDs. Created once from the samples complete at that moment, written to
    data/splits/hpa_calib_extra_v2.txt and never rewritten."""
    if CALIB_EXTRA_V2_FILE.exists():
        return _read_ids(CALIB_EXTRA_V2_FILE)
    excluded = set(diffflim_v2_train_ids()) | set(test_ids(hpa_dir)) | set(calib_ids(hpa_dir))
    ids = [i for i in complete_ids(hpa_dir) if i not in excluded]
    CALIB_EXTRA_V2_FILE.write_text("\n".join(ids) + "\n")
    return ids


def held_out_ids(hpa_dir=HPA_DIR, include_calib_extra=True):
    """IDs that training must not use: test, calibration and (if it exists) extra calibration."""
    held = set(test_ids(hpa_dir)) | set(calib_ids(hpa_dir))
    if include_calib_extra and CALIB_EXTRA_V2_FILE.exists():
        held |= set(_read_ids(CALIB_EXTRA_V2_FILE))
    return held


def train_ids(hpa_dir=HPA_DIR, write=True, allow_calib_extra=False):
    """All complete samples that are neither test, calibration nor extra-calibration samples,
    computed now (new downloads are added to training automatically). Written to
    data/splits/hpa_train.txt. allow_calib_extra=True also returns the extra calibration cells;
    only use it deliberately and note it in the project log (the extra calibration set then
    stops being unseen data)."""
    held_out = held_out_ids(hpa_dir, include_calib_extra=not allow_calib_extra)
    ids = [i for i in complete_ids(hpa_dir) if i not in held_out]
    if write:
        TRAIN_IDS_FILE.write_text("\n".join(ids) + "\n")
    return ids


_HPA_CACHE = {}


def load_hpa(sid, hpa_dir=HPA_DIR):
    """Decoded sample as uint8 (512, 512, 4) in channel order R, G, B, Y. Cached in memory only
    (the images are never written anywhere else)."""
    key = (str(hpa_dir), sid)
    if key not in _HPA_CACHE:
        _HPA_CACHE[key] = np.stack(
            [np.asarray(Image.open(Path(hpa_dir) / f"{sid}_{c}.png").convert("L")) for c in CHANNELS], -1)
    return _HPA_CACHE[key]


# ----------------------------------------------------------------------------------------
# IRFs
# ----------------------------------------------------------------------------------------
def simulate_irf(cfg, sigma, loc, skewness, loc_secondary, ratio):
    """FLIMngo notebook 2 `simulate_IRF` (secondary peak uses `sigma`, as in their code)."""
    t = cfg.time_starts
    irf = skewnorm.pdf(t, skewness, loc=loc, scale=sigma) + ratio * norm.pdf(t, loc=loc_secondary, scale=sigma)
    return irf


def _load_irf_file(name, cfg):
    path = IRF_DIR / name
    if name == "IRF_DLT.dat":
        # 10000 channels of 5 ps ("irf_5ps.dat"); integrate into our bins over the first window
        y = np.loadtxt(path)[:, 1]
        t = (np.arange(len(y)) + 0.5) * 0.005
        keep = t < cfg.period_ns
        return np.bincount((t[keep] / cfg.bin_width_ns).astype(int), weights=y[keep], minlength=cfg.n_bins)[:cfg.n_bins]
    if name.endswith(".sdt"):
        import sdtfile                                        # in-house PMT IRF, 256 x 0.0977 ns
        y = np.moveaxis(sdtfile.SdtFile(path).data[0], -1, 0).astype(np.float64)
    elif name == "IRF_FPFLI.mat":
        import h5py                                           # MATLAB v7.3 = HDF5
        with h5py.File(path, "r") as f:
            y = f["irf"][:, 0]
    elif name.endswith(".mat"):
        y = sio.loadmat(path)["irf_whole"][:, 0]
    elif name.endswith(".csv"):
        y = np.loadtxt(path, delimiter=",")
    else:
        y = np.loadtxt(path)
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) != cfg.n_bins:
        raise ValueError(f"{name}: {len(y)} points, expected {cfg.n_bins}")
    return y


@lru_cache(maxsize=4)
def _irf_library(n_bins, bin_width_ns):
    cfg = SimConfigV3(n_bins=n_bins, bin_width_ns=bin_width_ns)
    lib = []
    for irf_id, src, par, jitter in IRF_LIBRARY:
        y = simulate_irf(cfg, **par) if src == "sim" else _load_irf_file(par, cfg)
        y = np.clip(y, 0, None)
        lib.append({"id": irf_id, "irf": y / y.sum(), "jitter": jitter})
    return lib


def irf_library(cfg=None):
    """List of {"id", "irf" (T,) summing to 1, "jitter" (early, late)}."""
    cfg = SimConfigV3() if cfg is None else cfg
    return _irf_library(cfg.n_bins, cfg.bin_width_ns)


def jitter_irf(irf, jitter, rng):
    """FLIMngo laser jitter. Returns (shifted IRF normalised to sum 1, signed shift in bins)."""
    T = len(irf)
    if rng.random() <= 0.5:
        s = 1 + int(rng.integers(0, jitter[0]))
        out = np.concatenate([irf[s:], np.zeros(s)])
        shift = -s
    else:
        s = int(rng.integers(0, jitter[1]))
        out = np.concatenate([np.zeros(s), irf[:T - s]])
        shift = s
    return out / out.sum(), shift


# ----------------------------------------------------------------------------------------
# Decay model
# ----------------------------------------------------------------------------------------
def multi_exp_curves(amps, taus, irf, cfg):
    """amps, taus: (C, P). Returns normalised IRF-convolved decays (T, P), each summing to 1.

    Equals conv(sum_c a_c exp(-t_k / tau_c), irf)[:T] (FLIMngo: scipy.signal.convolve, 'full',
    truncated). Because exp(-k dt / tau) = r^k with r = exp(-dt / tau), each component's
    convolution obeys y[n] = r * y[n-1] + irf[n]; this recursion over the 256 bins replaces both
    the exponentials and the convolution and is several times faster.
    """
    amps = np.asarray(amps, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    r = np.where(taus > 0, np.exp(-cfg.bin_width_ns / np.where(taus > 0, taus, 1.0)), 0.0)
    y = np.zeros_like(r)
    out = np.empty((cfg.n_bins, r.shape[1]))
    for n in range(cfg.n_bins):
        y *= r
        y += irf[n]
        np.einsum("cp,cp->p", amps, y, out=out[n])
    out = np.clip(out, 0.0, None)
    return out / out.sum(axis=0, keepdims=True)


def decay_curves_v3(tau, irf, cfg):
    """Mono-exponential version of the v3 forward model: tau any shape -> (T, *shape), sums to 1.
    Used by the MLE baselines on v3 data."""
    tau = np.asarray(tau, dtype=np.float64)
    flat = tau.reshape(1, -1)
    out = multi_exp_curves(np.ones_like(flat), flat, irf, cfg)
    return out.reshape(cfg.n_bins, *tau.shape)


# ----------------------------------------------------------------------------------------
# Ground-truth maps
# ----------------------------------------------------------------------------------------
def perlin(rng, h, w, octaves, persistence):
    """FLIMngo `getPerlin`: sum of upsampled random grids, weighted persistence^(octaves - i),
    normalised to [0, 1]. Bilinear upsampling with scipy instead of skimage."""
    out = np.zeros((h, w))
    for i in range(octaves):
        gh, gw = max(int(h / 2 ** i), 1), max(int(w / 2 ** i), 1)
        z = rng.random((gh, gw))
        up = zoom(z, (h / gh, w / gw), order=1, mode="nearest", grid_mode=True)[:h, :w]
        out += up * persistence ** (octaves - i)
    out -= out.min()
    return out / max(out.max(), 1e-12)


def channel_amplitudes(present, p):
    """FLIMngo `calculate_ratio`, vectorised. present (C, H, W) bool, p (H, W) -> amps (C, H, W)."""
    n = present.sum(0)                                   # contributing channels per pixel
    k = np.cumsum(present, axis=0) - 1                   # rank of each channel among them
    amp = np.where(k < (n - 1)[None], p[None] * (1 - p[None]) ** k, (1 - p[None]) ** (n - 1)[None])
    return np.where(present, amp, 0.0)


def _random_crop(rng, cfg, ids, hpa_dir, fixed_id=None):
    s = cfg.size
    for _ in range(cfg.max_crop_tries):
        sid = fixed_id if fixed_id is not None else ids[int(rng.integers(len(ids)))]
        img = load_hpa(sid, hpa_dir)
        y, x = (int(v) for v in rng.integers(0, np.array(img.shape[:2]) - s + 1))
        k, flip = int(rng.integers(4)), bool(rng.random() < 0.5)
        crop = np.rot90(img[y:y + s, x:x + s], k)
        crop = crop[:, ::-1] if flip else crop
        if (crop.astype(np.int32).sum(-1) > 0).mean() >= cfg.min_fg_frac:
            break
    return np.ascontiguousarray(crop), sid, (y, x, k, int(flip))


def make_scene(rng, cfg, ids, hpa_dir=HPA_DIR, fixed_id=None):
    """Everything about one image except the photon level and the Poisson noise: crop, mask,
    lifetimes, amplitudes, relative intensity (foreground mean 1), IRF and the normalised decay
    curves of the foreground pixels."""
    crop, sid, crop_info = _random_crop(rng, cfg, ids, hpa_dir, fixed_id)
    H, W = crop.shape[:2]
    chans = np.moveaxis(crop, -1, 0).astype(np.float64)                 # (4, H, W)
    present = chans > 0
    mask = present.any(0)

    # lifetimes per channel (base + per-pixel uniform spread), FLIMngo tau_channel
    base = rng.uniform(*cfg.tau_base_range, size=4)
    spread = rng.uniform(*cfg.tau_spread_range, size=4)
    taus = np.where(present, base[:, None, None] + rng.random((4, H, W)) * spread[:, None, None], 0.0)

    # amplitudes from Perlin noise, FLIMngo calculate_ratio
    octaves = int(rng.integers(cfg.perlin_octaves[0], cfg.perlin_octaves[1] + 1))
    p = perlin(rng, H, W, octaves, rng.uniform(*cfg.perlin_persistence))
    amps = channel_amplitudes(present, p)

    tau_gt = (amps * taus).sum(0)                                         # amplitude-weighted
    with np.errstate(invalid="ignore", divide="ignore"):
        tau_int = np.where(mask, (amps * taus ** 2).sum(0) / tau_gt, 0.0)  # intensity-weighted

    # intensity: FLIMngo min-max scaling of the channel sum; foreground mean 1 (scaled later)
    s = chans.sum(0)
    norm_s = (s - s.min()) / max(s.max() - s.min(), 1e-12)
    lo = rng.uniform(*cfg.min_photons_range)
    rel = np.where(mask, lo + norm_s * rng.uniform(*cfg.delta_photons_range), 0.0)
    rel /= rel[mask].mean()

    # IRF with laser jitter
    lib = irf_library(cfg)
    j = int(rng.integers(len(lib)))
    irf, shift = jitter_irf(lib[j]["irf"], lib[j]["jitter"], rng)
    curves = multi_exp_curves(amps[:, mask], taus[:, mask], irf, cfg)    # (T, P_fg)
    return {"mask": mask, "rel_intensity": rel, "curves": curves, "tau_gt": tau_gt, "tau_int": tau_int,
            "n_components": present.sum(0), "irf": irf, "irf_id": lib[j]["id"], "irf_shift": shift,
            "hpa_id": sid, "crop": crop_info, "channels": crop, "amps": amps, "taus": taus}


def render_scene(scene, cfg, photons, rng, return_extra=False):
    """Scale a scene to `photons` (mean signal per foreground pixel), add background, sample."""
    mask = scene["mask"]
    inten = scene["rel_intensity"] * photons
    bg = cfg.bg_fraction * photons / cfg.n_bins
    expected = np.full((cfg.n_bins,) + mask.shape, bg)
    expected[:, mask] += inten[mask][None] * scene["curves"]
    counts = sample_counts(expected, rng)
    out = {"counts": counts, "tau_gt": scene["tau_gt"].astype(np.float32), "intensity_gt": inten.astype(np.float32),
           "mask": mask, "bg_per_bin": bg, "photons": float(photons),
           "tau_int_gt": scene["tau_int"].astype(np.float32),
           "n_components": scene["n_components"].astype(np.uint8),
           "irf": scene["irf"].astype(np.float32), "irf_id": scene["irf_id"], "irf_shift": scene["irf_shift"],
           "hpa_id": scene["hpa_id"], "crop": np.array(scene["crop"], dtype=np.int32)}
    if return_extra:
        out.update(expected=expected, channels=scene["channels"], amps=scene["amps"], taus=scene["taus"])
    return out


def simulate_sample(rng, cfg, photons, ids, hpa_dir=HPA_DIR, return_extra=False):
    """One image (scene + rendering with the same rng). Returns dict like simulate.simulate_sample
    plus irf, irf_id, irf_shift, hpa_id, crop, tau_int_gt, n_components. With return_extra: also
    'expected', 'channels', 'amps', 'taus'."""
    return render_scene(make_scene(rng, cfg, ids, hpa_dir), cfg, photons, rng, return_extra)


def training_samples(cfg=None, photon_range=(10, 1000), seed=None, ids=None, hpa_dir=HPA_DIR,
                     allow_calib_extra=False):
    """Endless generator of training samples from TRAIN IDs only (log-uniform photon level)."""
    cfg = SimConfigV3() if cfg is None else cfg
    ids = train_ids(hpa_dir, write=False, allow_calib_extra=allow_calib_extra) if ids is None else list(ids)
    if set(ids) & held_out_ids(hpa_dir, include_calib_extra=not allow_calib_extra):
        raise ValueError("Training IDs overlap with the fixed HPA test or (extra) calibration IDs.")
    rng = np.random.default_rng(seed)
    lo, hi = photon_range
    while True:
        yield simulate_sample(rng, cfg, float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), ids, hpa_dir)


# ----------------------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------------------
def _stack_samples(samples, cfg, meta):
    stack = lambda k, dt=None: np.stack([np.asarray(x[k], dtype=dt) for x in samples])  # noqa: E731
    return {
        "counts": stack("counts"), "tau_gt": stack("tau_gt"), "intensity_gt": stack("intensity_gt"),
        "mask": stack("mask"), "tau_int_gt": stack("tau_int_gt"), "n_components": stack("n_components"),
        "photons": np.array([x["photons"] for x in samples], dtype=np.float32),
        "bg_per_bin": stack("bg_per_bin", np.float32), "irf": stack("irf"),
        "irf_id": np.array([x["irf_id"] for x in samples]), "irf_shift": stack("irf_shift", np.int32),
        "hpa_id": np.array([x["hpa_id"] for x in samples]), "crop": stack("crop"),
        "time_ns": cfg.time_centers.astype(np.float32), "metadata": json.dumps(meta),
    }


def _meta(cfg, **extra):
    return {**asdict(cfg), "simulator": "v3", "irf_library": [x[0] for x in IRF_LIBRARY],
            "irf_excluded": EXCLUDED_IRFS,
            "tau_gt": "amplitude-weighted mean lifetime sum a_i tau_i (FLIMngo); 0 outside mask",
            "intensity_gt": "expected signal photons per pixel (background excluded)",
            "photons_definition": "mean signal photons per foreground (mask) pixel",
            "counts_shape": "(N, T, H, W)", **extra}


def generate_dataset(n, photons, ids, cfg=None, seed=0, hpa_dir=HPA_DIR, id_source="test"):
    """n independent samples (scene and noise from one rng)."""
    cfg = SimConfigV3() if cfg is None else cfg
    rng = np.random.default_rng(seed)
    samples = [simulate_sample(rng, cfg, photons, ids, hpa_dir) for _ in range(n)]
    return _stack_samples(samples, cfg, _meta(cfg, seed=seed, photons=photons, hpa_ids=id_source))


def generate_paired_dataset(n, photons, ids, scene_seed, noise_seed, cfg=None, hpa_dir=HPA_DIR, id_source="test"):
    """n samples whose scenes (crop, lifetimes, amplitudes, relative intensity, IRF) depend only on
    scene_seed, so that data sets at different photon levels contain the same images; only the
    photon level and the Poisson noise (noise_seed) differ."""
    cfg = SimConfigV3() if cfg is None else cfg
    scene_rng, noise_rng = np.random.default_rng(scene_seed), np.random.default_rng(noise_seed)
    # every ID is used (round robin over a shuffled list): n = 40 over 30 test IDs -> 10 IDs twice
    order = [ids[i] for i in scene_rng.permutation(len(ids))]
    samples = [render_scene(make_scene(scene_rng, cfg, ids, hpa_dir, fixed_id=order[i % len(order)]),
                            cfg, photons, noise_rng) for i in range(n)]
    return _stack_samples(samples, cfg, _meta(cfg, scene_seed=scene_seed, noise_seed=noise_seed, photons=photons,
                                              hpa_ids=id_source, paired=True))


def test_set_dir():
    return RAW_OWN / "sim_v3_test"


def calib_set_dir():
    return RAW_OWN / "sim_v3_calib"


def _make_paired_set(out_dir, prefix, ids, id_source, levels, n_per_level, seed, force):
    paths = []
    for level in levels:
        path = Path(out_dir) / f"{prefix}_p{level}.npz"
        if path.exists() and not force:
            print(f"{path} exists, skipping")
        else:
            save_dataset(path, generate_paired_dataset(n_per_level, level, ids, scene_seed=seed,
                                                       noise_seed=seed + level, id_source=id_source))
            print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
        paths.append(path)
    return paths


def make_test_set(out_dir=None, levels=TEST_PHOTON_LEVELS, n_per_level=40, seed=TEST_SEED, force=False):
    """Fixed, PAIRED v3 test set from the fixed TEST IDs only: test_p{level}.npz. All levels contain
    the same 40 scenes (scene seed TEST_SEED); the noise uses TEST_SEED + level."""
    return _make_paired_set(test_set_dir() if out_dir is None else out_dir, "test", test_ids(), "test",
                            levels, n_per_level, seed, force)


def calib_extra_set_dir():
    return RAW_OWN / "sim_v3_calib_extra"


def make_calib_extra_set(out_dir=None, levels=TEST_PHOTON_LEVELS, n_per_level=100, seed=CALIB_EXTRA_SET_SEED,
                         force=False):
    """Extra PAIRED calibration set from the cells DiffFLIM v2 never saw: calib_p{level}.npz."""
    return _make_paired_set(calib_extra_set_dir() if out_dir is None else out_dir, "calib", calib_extra_v2_ids(),
                            "calib_extra_v2", levels, n_per_level, seed, force)


def make_calib_set(out_dir=None, levels=TEST_PHOTON_LEVELS, n_per_level=100, seed=CALIB_SET_SEED, force=False):
    """Fixed, PAIRED v3 calibration set from the fixed CALIBRATION IDs only: calib_p{level}.npz."""
    return _make_paired_set(calib_set_dir() if out_dir is None else out_dir, "calib", calib_ids(), "calib",
                            levels, n_per_level, seed, force)


if __name__ == "__main__":
    import sys
    tr = train_ids()
    print(f"HPA: {len(tr)} train IDs, {len(calib_ids())} calibration IDs, {len(test_ids())} test IDs")
    make_test_set(force="--force" in sys.argv)
    make_calib_set(force="--force" in sys.argv)
