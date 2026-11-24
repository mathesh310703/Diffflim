"""Train the FLIMngo architecture on the FLUTE configuration of simulator v3 (fair deep-learning baseline
for DiffFLIM-FLUTE; 2026-11-23/24).

Same as scripts/train_flimngo.py (FLIMngo's train.py: MSE on the lifetime in ns over all pixels, target 0
outside the true foreground, AdamW lr 0.01, weight decay 1e-8, constant learning rate, float32, batch
16, background mask computed from the counts with "otsu_adaptive"), with these FLUTE-specific parts:
  * data: simulate_v3.flute_nadh_config() (56 bins of 0.223 ns, periodic, FLUTE IRF set, NAD(P)H-like
    and mono-exponential scenes), photon level log-uniform 10 to 1000, the SAME training IDs as
    DiffFLIM-FLUTE (read from checkpoints/diffflim_flute.pt, else train_ids());
  * input format: FLIMngo needs 256 time bins. As in FLIMngo's repository
    (utils/eval_utils.py: resize_data_tdim = scipy.ndimage.zoom(order=1) on the time axis, and
    resize_time for the new bin width; demo_notebooks/predict_diff_time_dimensions.ipynb), the 56 bins
    are interpolated linearly to 256 with the first and last bins aligned (torch interpolate,
    align_corners=True, identical to zoom order 1), and the network gets the resized bin width
    (56 - 1) x 0.223 / 255 = 0.0481 ns;
  * validation: the paired FLUTE validation set (data/raw/own/sim_flute_val, 40 validation cells,
    40 images at each of 10, 20, 50, 100, 500 photons) every --val-every minutes; the checkpoint with
    the lowest MEDIAN RELATIVE ERROR on the true foreground is kept (selection on the MSE gave the
    retrained FLIMngo v3 an output offset, see the project log of 2026-11-22); the MSE is logged too.

Writes checkpoints/{name}.pt (best validation median relative error), checkpoints/{name}_last.pt,
logs/{name}.csv, logs/{name}_val.csv. A file named STOP in the project root stops the run cleanly.

Usage:
    python scripts/train_flimngo_flute.py --minutes 1 --name flimngo_flute_smoke
    python scripts/train_flimngo_flute.py --minutes 150 --name flimngo_flute
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import simulate_v3 as v3  # noqa: E402
from src.data.simulate import load_dataset  # noqa: E402
from src.evaluation.calibration import foreground_mask  # noqa: E402
from src.models.flimngo_arch import build_flimngo, predict  # noqa: E402
from src.paths import CHECKPOINTS, LOGS  # noqa: E402

MASK_METHOD = "otsu_adaptive"
N_IN = 56
BIN_WIDTH_RESIZED = (N_IN - 1) * 0.223 / 255
STOP_FILE = ROOT / "STOP"


def resize_to_256(counts):
    """(B, 56, H, W) float -> (B, 256, H, W), linear interpolation with aligned end points
    (= scipy.ndimage.zoom(order=1) on the time axis, FLIMngo's resize_data_tdim)."""
    b, t, h, w = counts.shape
    x = counts.permute(0, 2, 3, 1).reshape(b * h * w, 1, t)
    x = F.interpolate(x, size=256, mode="linear", align_corners=True)
    return x.reshape(b, h, w, 256).permute(0, 3, 1, 2).contiguous()


def flute_predict(net, counts, mask):
    return predict(net, resize_to_256(counts), mask)


def to_item(s):
    counts = s["counts"]
    return {"counts": torch.from_numpy(counts.astype(np.int16)),
            "mask": torch.from_numpy(foreground_mask(counts.sum(0), MASK_METHOD)),
            "target": torch.from_numpy(np.where(s["mask"], s["tau_gt"], 0).astype(np.float32)),
            "true_mask": torch.from_numpy(s["mask"])}


class FluteStream(IterableDataset):
    def __init__(self, ids, seed, photon_range=(10, 1000)):
        self.ids, self.seed, self.photon_range = list(ids), seed, photon_range

    def __iter__(self):
        info = get_worker_info()
        worker = info.id if info is not None else 0
        for s in v3.training_samples(v3.flute_nadh_config(), self.photon_range, seed=self.seed * 1000 + worker,
                                     ids=self.ids):
            yield to_item(s)


def load_val():
    files = sorted(v3.flute_set_dir("val").glob("val_p*.npz"), key=lambda p: int(p.stem.split("_p")[1]))
    items = []
    for f in files:
        d = load_dataset(f)
        for i in range(len(d["counts"])):
            items.append(to_item({"counts": d["counts"][i], "mask": d["mask"][i], "tau_gt": d["tau_gt"][i]}))
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


@torch.no_grad()
def validate(net, val, device):
    net.eval()
    losses, rel = [], []
    for i in range(0, len(val["counts"]), 16):
        c = val["counts"][i:i + 16].to(device).float()
        y = flute_predict(net, c, val["mask"][i:i + 16].to(device))
        t = val["target"][i:i + 16].to(device)
        tm = val["true_mask"][i:i + 16].to(device)
        losses.append(torch.mean((y - t) ** 2).item())
        rel.append((torch.abs(y - t)[tm] / t[tm]).cpu())
    net.train()
    return float(np.mean(losses)), float(torch.cat(rel).median()) * 100


def save(path, net, opt, step, elapsed, args, ids, val):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "step": step, "elapsed_s": elapsed,
                "args": vars(args), "train_ids": list(ids), "val": val, "mask_method": MASK_METHOD,
                "arch": {"image_size": 64, "bin_width": BIN_WIDTH_RESIZED, "n_in": N_IN, "resize": "linear to 256"}},
               tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=150)
    ap.add_argument("--name", default="flimngo_flute")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=1e-8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val-every", type=float, default=10, help="minutes")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if STOP_FILE.exists():
        print(f"{STOP_FILE} exists: remove it to start training.")
        sys.exit(1)
    device = "cuda"
    ref = CHECKPOINTS / "diffflim_flute.pt"
    if ref.exists():
        ids = list(torch.load(ref, map_location="cpu", weights_only=False)["args"]["train_ids"])
        source = "checkpoints/diffflim_flute.pt"
    else:
        ids, source = v3.train_ids(), "train_ids()"
    assert not set(ids) & v3.held_out_ids(), "training IDs overlap held-out IDs"
    net = build_flimngo(64, BIN_WIDTH_RESIZED, seed=args.seed).to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"FLIMngo architecture (FLUTE): {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters, "
          f"{len(ids)} training IDs from {source}, input 56 bins resized to 256 (bin width {BIN_WIDTH_RESIZED:.4f} ns)",
          flush=True)
    val = load_val()
    print(f"Validation set: {len(val['counts'])} images (sim_flute_val)", flush=True)
    loader = DataLoader(FluteStream(ids, args.seed), batch_size=args.batch, num_workers=args.workers,
                        pin_memory=True, prefetch_factor=4, persistent_workers=True)
    flog = open(LOGS / f"{args.name}.csv", "w", newline="")
    vlog = open(LOGS / f"{args.name}_val.csv", "w", newline="")
    w, wv = csv.writer(flog), csv.writer(vlog)
    w.writerow(["step", "elapsed_min", "loss", "it_per_s", "data_wait_frac"])
    wv.writerow(["step", "elapsed_min", "val_mse", "val_mare_pct", "best"])
    best = float("inf")
    step, t_start = 0, time.perf_counter()
    next_val = args.val_every
    wait, busy, losses, t_win = 0.0, 0.0, [], time.perf_counter()
    t_fetch = time.perf_counter()
    stopped = False
    vl, vm = float("nan"), float("nan")
    for batch in loader:
        t_got = time.perf_counter()
        wait += t_got - t_fetch
        c = batch["counts"].to(device, non_blocking=True).float()
        y = flute_predict(net, c, batch["mask"].to(device, non_blocking=True))
        loss = torch.mean((y - batch["target"].to(device, non_blocking=True)) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        losses.append(loss.item())
        t_fetch = time.perf_counter()
        busy += t_fetch - t_got
        elapsed = (t_fetch - t_start) / 60
        if step % args.log_every == 0:
            dt = t_fetch - t_win
            w.writerow([step, f"{elapsed:.2f}", f"{np.mean(losses):.5f}", f"{args.log_every / dt:.2f}",
                        f"{wait / (wait + busy):.2f}"])
            flog.flush()
            if step % (args.log_every * 10) == 0:
                print(f"step {step:7d}  {elapsed:6.1f} min  loss {np.mean(losses):.4f}  {args.log_every / dt:5.1f} it/s",
                      flush=True)
            wait, busy, losses, t_win = 0.0, 0.0, [], time.perf_counter()
            stopped = STOP_FILE.exists()
        done = elapsed >= args.minutes or stopped
        if elapsed >= next_val or done:
            vl, vm = validate(net, val, device)
            is_best = vm < best
            if is_best:
                best = vm
                save(CHECKPOINTS / f"{args.name}.pt", net, opt, step, elapsed * 60, args, ids, {"mse": vl, "mare_pct": vm})
            wv.writerow([step, f"{elapsed:.2f}", f"{vl:.5f}", f"{vm:.3f}", int(is_best)])
            vlog.flush()
            print(f"  validation at {elapsed:.1f} min: MSE {vl:.4f} ns^2, median rel. error {vm:.2f}%"
                  f"{' (best, saved)' if is_best else ''}", flush=True)
            next_val += args.val_every
            t_fetch = time.perf_counter()
        if done:
            break
    save(CHECKPOINTS / f"{args.name}_last.pt", net, opt, step, elapsed * 60, args, ids, {"mse": vl, "mare_pct": vm})
    status = "stopped" if stopped else "completed"
    (LOGS / f"{args.name}_status.txt").write_text(f"{status} at step {step}, {elapsed:.1f} min, best val median rel. error {best:.2f} %\n")
    print(f"Done ({status}): {step} steps in {elapsed:.1f} min. Best validation median rel. error {best:.2f} %", flush=True)


if __name__ == "__main__":
    main()
