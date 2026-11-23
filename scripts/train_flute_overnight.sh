#!/usr/bin/env bash
# PREPARED 2026-11-23, NOT STARTED: overnight training of DiffFLIM on the FLUTE configuration of
# simulator v3 (src/data/simulate_v3.py: flute_config(); 56 bins x 0.223 ns, 12.5 ns period with
# wrap-around, FLUTE IRF set, base lifetimes 0.2 to 5 ns). Same settings as DiffFLIM v2 except the
# simulator and the input binning (56 bins summed in pairs -> 28 channels).
#
# Training IDs: train_ids(write=True) -> all current training IDs (474 on 2026-11-23: the 386 of
# v3cells plus 88 samples downloaded since), written to data/splits/hpa_train.txt and stored in the
# checkpoint. Validation, test, second-test and calibration IDs are excluded automatically.
# Graceful stop: create a file named STOP in the project root (final checkpoint, status in
# logs/diffflim_flute_v1_status.txt); resume later with --resume checkpoints/diffflim_flute_v1.pt.
#
# Not yet available for this model: a FLUTE test/validation set and the post-training chain
# (scripts/post_training_chain.sh evaluates on the 256-bin v3 sets and cannot be used).
#
# Usage: bash scripts/train_flute_overnight.sh      (about 6 h 05 min)
cd "$(dirname "$0")/.."
python -W ignore scripts/train_diffflim.py --simulator v3flute --minutes 360 --name diffflim_flute_v1 \
    --snapshot-at 180 --batch 16 --lr 2e-4 --warmup 500 --ema 0.999 --base 64 --prediction v \
    --ckpt-every 15 --workers 16 --seed 0 > logs/diffflim_flute_v1_stdout.txt 2>&1
echo "exit $?: $(cat logs/diffflim_flute_v1_status.txt)"
