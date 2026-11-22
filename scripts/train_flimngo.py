"""Train the FLIMngo architecture from scratch on simulator-v3 data (src/models/flimngo_arch.py).

Fair-comparison setup: the same 246 HPA training IDs as DiffFLIM v2
(data/splits/hpa_train_diffflim_v2.txt), the same photon range (10-1000, log-uniform), the same
64 x 64 crops and the same simulator, streamed fresh (nothing saved).

Following FLIMngo's train.py: MSE on the lifetime in ns over all pixels (target 0 outside the true
foreground), AdamW, lr 0.01, weight decay 1e-8, constant learning rate, float32, best checkpoint
chosen on a validation set. Input preprocessing as in FLIMngo, but with the background mask
COMPUTED from the counts (src/evaluation/calibration.py::foreground_mask, "otsu_adaptive"), in
training and at evaluation, because the true mask is not available for real data and DiffFLIM
does not get it either. True foreground pixels outside the computed mask have zero input; their
target is still the true lifetime (the network may fill them in from the neighbourhood).

Deviations from their train.py: batch 16 instead of 10 (same images per step as DiffFLIM v2), a
fixed time budget instead of epochs (data are streamed), validation every --val-every minutes on
64 fixed images from TRAINING cells (their validation set was a random 20 % of their training
images) instead of every epoch.

Writes checkpoints/{name}.pt (best validation loss), checkpoints/{name}_last.pt,
       logs/{name}.csv (training), logs/{name}_val.csv (validation).

Usage:
    python scripts/train_flimngo.py --minutes 2 --name flimngo_smoke      # smoke test
    python scripts/train_flimngo.py --minutes 180 --name flimngo_v3
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import simulate_v3 as v3  # noqa: E402
from src.evaluation.calibration import foreground_mask  # noqa: E402
from src.models.flimngo_arch import build_flimngo, predict  # noqa: E402
from src.paths import CHECKPOINTS, LOGS  # noqa: E402

MASK_METHOD = "otsu_adaptive"


def to_item(s):
    counts = s["counts"]
    return {
        "counts": torch.from_numpy(counts.astype(np.int16)),          # max count per bin << 32767
        "mask": torch.from_numpy(foreground_mask(counts.sum(0), MASK_METHOD)),
        "target": torch.from_numpy(np.where(s["mask"], s["tau_gt"], 0).astype(np.float32)),
        "true_mask": torch.from_numpy(s["mask"]),
    }


class FlimngoStream(IterableDataset):
    def __init__(self, ids, seed, photon_range=(10, 1000)):
        self.ids, self.seed, self.photon_range = list(ids), seed, photon_range

    def __iter__(self):
        info = get_worker_info()
        worker = info.id if info is not None else 0
        for s in v3.training_samples(None, self.photon_range, seed=self.seed * 1000 + worker, ids=self.ids):
            yield to_item(s)


def make_val_set(ids, n=64, seed=777):
    rng = np.random.default_rng(seed)
    cfg = v3.SimConfigV3()
    items = []
    for _ in range(n):
        photons = float(np.exp(rng.uniform(np.log(10), np.log(1000))))
        items.append(to_item(v3.simulate_sample(rng, cfg, photons, ids)))
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


@torch.no_grad()
def validate(net, val, device):
    net.eval()
    losses, rel = [], []
    for i in range(0, len(val["counts"]), 16):
        c = val["counts"][i:i + 16].to(device).float()
        m = val["mask"][i:i + 16].to(device)
        t = val["target"][i:i + 16].to(device)
        tm = val["true_mask"][i:i + 16].to(device)
        y = predict(net, c, m)
        losses.append(torch.mean((y - t) ** 2).item())
        rel.append((torch.abs(y - t)[tm] / t[tm]).cpu())
    net.train()
    return float(np.mean(losses)), float(torch.cat(rel).median()) * 100


def save(path, net, opt, step, elapsed, args, ids, val_loss):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "step": step, "elapsed_s": elapsed,
                "args": vars(args), "train_ids": list(ids), "val_loss": val_loss,
                "mask_method": MASK_METHOD, "arch": {"image_size": 64, "bin_width": v3.SimConfigV3().bin_width_ns}},
               tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=180)
    ap.add_argument("--name", default="flimngo_v3")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=1e-8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val-every", type=float, default=10, help="minutes")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    ids = v3.diffflim_v2_train_ids()
    assert not set(ids) & v3.held_out_ids(), "training IDs overlap held-out IDs"
    net = build_flimngo(64, v3.SimConfigV3().bin_width_ns, seed=args.seed).to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"FLIMngo architecture: {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters, "
          f"{len(ids)} training IDs (DiffFLIM v2 list), batch {args.batch}, lr {args.lr}, {args.workers} workers",
          flush=True)
    t0 = time.perf_counter()
    val = make_val_set(ids)
    print(f"Validation set: {len(val['counts'])} images from training cells ({time.perf_counter() - t0:.0f} s)",
          flush=True)

    loader = DataLoader(FlimngoStream(ids, args.seed), batch_size=args.batch, num_workers=args.workers,
                        pin_memory=True, prefetch_factor=4, persistent_workers=True)
    LOGS.mkdir(exist_ok=True); CHECKPOINTS.mkdir(exist_ok=True)
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
    for batch in loader:
        t_got = time.perf_counter()
        wait += t_got - t_fetch
        c = batch["counts"].to(device, non_blocking=True).float()
        m = batch["mask"].to(device, non_blocking=True)
        t = batch["target"].to(device, non_blocking=True)
        y = predict(net, c, m)
        loss = torch.mean((y - t) ** 2)
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
                print(f"step {step:7d}  {elapsed:6.1f} min  loss {np.mean(losses):.4f}  "
                      f"{args.log_every / dt:5.1f} it/s  data wait {wait / (wait + busy):.0%}", flush=True)
            wait, busy, losses, t_win = 0.0, 0.0, [], time.perf_counter()
        done = elapsed >= args.minutes
        if elapsed >= next_val or done:
            vl, vm = validate(net, val, device)
            is_best = vl < best
            if is_best:
                best = vl
                save(CHECKPOINTS / f"{args.name}.pt", net, opt, step, elapsed * 60, args, ids, vl)
            wv.writerow([step, f"{elapsed:.2f}", f"{vl:.5f}", f"{vm:.3f}", int(is_best)])
            vlog.flush()
            print(f"  validation at {elapsed:.1f} min: MSE {vl:.4f} ns^2, median rel. error {vm:.2f}%"
                  f"{' (best, saved)' if is_best else ''}", flush=True)
            next_val += args.val_every
            t_fetch = time.perf_counter()
        if done:
            break
    save(CHECKPOINTS / f"{args.name}_last.pt", net, opt, step, elapsed * 60, args, ids, vl)
    print(f"Done: {step} steps in {elapsed:.1f} min ({step / elapsed / 60:.1f} it/s). Best validation MSE {best:.4f}. "
          f"Peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
