"""Train DiffFLIM v1 on simulated data generated on the fly, for a fixed wall-clock time.

Saves checkpoints/{name}.pt every --ckpt-every minutes and at the end; logs to logs/{name}.csv.
At the end, a quick sampling check runs on one test image (100 photons) and is compared to MLE.

Usage:
    python scripts/train_diffflim.py --minutes 2 --name diffflim_smoke     # smoke test
    python scripts/train_diffflim.py --minutes 60                          # main run
    python scripts/train_diffflim.py --simulator v3 --steps 186116 --snapshot-at-step 92168:3h         --name diffflim_v3cells --ckpt-every 30 --workers 16              # fixed number of steps

--steps N stops after N optimiser steps instead of after --minutes (the time limit is then not
used). --snapshot-at-step N:SUFFIX keeps a copy of the checkpoint after step N as
checkpoints/{name}_{SUFFIX}.pt (the time-based --snapshot-at still works).
"""
import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.stream import SimStream  # noqa: E402
from src.models.diffflim import DEFAULT_SAMPLE_STEPS, DiffFLIM  # noqa: E402
from src.paths import CHECKPOINTS, LOGS  # noqa: E402


def save_checkpoint(path, model, ema, opt, step, elapsed, args):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                "config": model.config, "step": step, "elapsed_s": elapsed, "args": vars(args)}, tmp)
    tmp.replace(path)   # atomic: a crash never leaves a half-written checkpoint


def quick_check(ema, device, simulator="v2"):
    """Sample one test image at 100 photons and compare tau error to MLE (foreground pixels)."""
    from src.baselines.common import fit_all
    from src.data.simulate import load_dataset
    from src.evaluation.metrics import summarize
    if simulator == "v3":
        from src.data.simulate_v3 import test_set_dir
    else:
        from src.data.simulate import test_set_dir

    path = test_set_dir() / "test_p100.npz"
    if not path.exists():
        print("No test set, skipping sampling check.")
        return
    d = load_dataset(path)
    counts, tau_gt, int_gt, m = d["counts"][0], d["tau_gt"][0], d["intensity_gt"][0], d["mask"][0]
    t0 = time.perf_counter()
    s = ema.sample(counts, n_samples=4, seed=0)
    print(f"Sampling check: samples {s.shape}, finite={np.isfinite(s).all()}, "
          f"{time.perf_counter() - t0:.1f} s for 4 samples x {DEFAULT_SAMPLE_STEPS} DDIM steps")
    base = fit_all(counts, float(d["bg_per_bin"][0]), d["metadata"], d["irf"][0] if "irf" in d else None,
                   methods=("MLE", "BinMLE5"))
    rows = [("DiffFLIM", s[:, 0].mean(0), s[:, 1].mean(0))] + [(k, t, I) for k, (t, I) in base.items()]
    for name, tau, inten in rows:
        print(f"  {name:8s} tau err {summarize(tau, tau_gt, m)['mare_pct']:.2f}%, "
              f"intensity err {summarize(inten, int_gt, m)['mare_pct']:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--name", default="diffflim_v1")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--prediction", choices=["v", "eps"], default="v")
    ap.add_argument("--simulator", choices=["v2", "v3"], default="v2", help="training data generator")
    ap.add_argument("--tau-range", type=float, nargs=2, default=None,
                    help="lifetime normalisation range in ns (default: 0.5 5 for v2, 0.1 12 for v3)")
    ap.add_argument("--snapshot-at", type=float, nargs="*", default=[],
                    help="also keep copies of the checkpoint at these training minutes (e.g. 180)")
    ap.add_argument("--steps", type=int, default=0,
                    help="stop after this many optimiser steps (0: use --minutes); the time limit is then ignored")
    ap.add_argument("--snapshot-at-step", nargs="*", default=[],
                    help="STEP:SUFFIX, keep checkpoints/{name}_{SUFFIX}.pt after STEP steps (e.g. 92168:3h)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ckpt-every", type=float, default=10, help="minutes")
    ap.add_argument("--log-every", type=int, default=50, help="steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    tau_range = tuple(args.tau_range) if args.tau_range else ((0.1, 12.0) if args.simulator == "v3" else (0.5, 5.0))
    model = DiffFLIM(base=args.base, prediction=args.prediction, tau_range=tau_range).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DiffFLIM: {n_params / 1e6:.2f} M parameters, batch {args.batch}, {args.workers} data workers")

    ids = None
    if args.simulator == "v3":
        from src.data import simulate_v3 as v3
        # computed once; excludes test, second-test, calibration and extra-calibration IDs. This is the
        # only place where data/splits/hpa_train.txt is written (record of the latest training run).
        ids = v3.train_ids(write=True)
        args.train_ids = list(ids)              # stored in every checkpoint (vars(args)) as the record
        print(f"Simulator v3: {len(ids)} training IDs (written to {v3.TRAIN_IDS_FILE.name}; held out: "
              f"{len(v3.held_out_ids())} test, second-test and calibration IDs)")
    stream = SimStream(seed=args.seed, simulator=args.simulator, ids=ids, tau_range=tau_range)
    loader = DataLoader(stream, batch_size=args.batch, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers > 0 else None)

    CHECKPOINTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINTS / f"{args.name}.pt"
    log_f = open(LOGS / f"{args.name}.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "elapsed_min", "loss", "lr", "it_per_s", "data_wait_frac"])

    t_start = time.perf_counter()
    t_log, t_ckpt, wait, losses, step = t_start, t_start, 0.0, [], 0
    pending_snapshots = sorted(args.snapshot_at)
    step_snapshots = {int(x.split(":")[0]): x.split(":")[1] for x in args.snapshot_at_step}
    t_fetch = time.perf_counter()
    for batch in loader:
        wait += time.perf_counter() - t_fetch
        elapsed = time.perf_counter() - t_start
        if (args.steps and step >= args.steps) or (not args.steps and elapsed > args.minutes * 60):
            break

        binned = batch["binned"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(binned, target, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        decay = min(args.ema, (1 + step) / (10 + step))   # EMA warms up over the first steps
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.lerp_(p, 1 - decay)
        losses.append(loss.item())
        step += 1

        if step % args.log_every == 0:
            now = time.perf_counter()
            ips = args.log_every / (now - t_log)
            log.writerow([step, f"{(now - t_start) / 60:.2f}", f"{np.mean(losses):.5f}", f"{lr:.2e}",
                          f"{ips:.2f}", f"{wait / (now - t_log):.2f}"])
            log_f.flush()
            if step % (args.log_every * 10) == 0:
                print(f"step {step:6d}  {(now - t_start) / 60:5.1f} min  loss {np.mean(losses):.4f}  "
                      f"{ips:.1f} it/s  data wait {wait / (now - t_log):.0%}", flush=True)
            losses, t_log, wait = [], now, 0.0
        for m in [x for x in pending_snapshots if elapsed >= x * 60]:
            snap = CHECKPOINTS / f"{args.name}_{int(m)}min.pt"
            save_checkpoint(snap, model, ema, opt, step, elapsed, args)
            pending_snapshots.remove(m)
            print(f"snapshot saved: {snap.name} (step {step})", flush=True)
        if step in step_snapshots:
            snap = CHECKPOINTS / f"{args.name}_{step_snapshots.pop(step)}.pt"
            save_checkpoint(snap, model, ema, opt, step, elapsed, args)
            print(f"snapshot saved: {snap.name} (step {step})", flush=True)
        if time.perf_counter() - t_ckpt > args.ckpt_every * 60:
            save_checkpoint(ckpt_path, model, ema, opt, step, time.perf_counter() - t_start, args)
            t_ckpt = time.perf_counter()
            print(f"checkpoint saved at step {step}", flush=True)
        t_fetch = time.perf_counter()

    elapsed = time.perf_counter() - t_start
    save_checkpoint(ckpt_path, model, ema, opt, step, elapsed, args)
    log_f.close()
    print(f"Done: {step} steps in {elapsed / 60:.1f} min ({step / elapsed:.1f} it/s). "
          f"Peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB. Saved {ckpt_path}")
    quick_check(ema, device, args.simulator)


if __name__ == "__main__":
    main()
