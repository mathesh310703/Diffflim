#!/usr/bin/env bash
# Overnight FLUTE chain (2026-11-23/24):
#   STEP 2  DiffFLIM-FLUTE training (200,000 steps, snapshot at 100,000; checkpoint every 15 min;
#           a file named STOP in the project root stops it cleanly)
#   STEP 3  post-training chain for the final model and the 100k snapshot
#           (scripts/post_training_chain_flute.sh: simulated FLUTE test set, calibration, real FLUTE data)
#   STEP 4  only if STEP 3 finished before 03:30: retrained FLIMngo on the same FLUTE simulator
#           (scripts/flimngo_flute_chain.sh); otherwise skipped and noted.
# If the training is stopped early, the remaining steps are skipped.
cd "$(dirname "$0")/.."
S=logs/night_flute_run.txt
echo "$(date +%H:%M) STEP 2 training start" > $S
python -W ignore scripts/train_diffflim.py --simulator v3flute_nadh --steps 200000 --snapshot-at-step 100000:100k \
    --name diffflim_flute --batch 16 --lr 2e-4 --warmup 500 --ema 0.999 --base 64 --prediction v \
    --ckpt-every 15 --workers 16 --seed 0 > logs/diffflim_flute_stdout.txt 2>&1
echo "$(date +%H:%M) STEP 2 training exit $?: $(cat logs/diffflim_flute_status.txt)" >> $S
if ! grep -q "^completed" logs/diffflim_flute_status.txt; then
    echo "$(date +%H:%M) training stopped early: STEPS 3 and 4 skipped" >> $S
    exit 0
fi
if [ -f STOP ]; then echo "$(date +%H:%M) STOP exists: STEPS 3 and 4 skipped" >> $S; exit 0; fi
bash scripts/post_training_chain_flute.sh diffflim_flute flute >> $S 2>&1
echo "$(date +%H:%M) STEP 3 chain flute exit $?" >> $S
if [ -f STOP ]; then echo "$(date +%H:%M) STOP exists: rest skipped" >> $S; exit 0; fi
bash scripts/post_training_chain_flute.sh diffflim_flute_100k flute_100k >> $S 2>&1
echo "$(date +%H:%M) STEP 3 chain flute_100k exit $?" >> $S
now=$(date +%H%M)
if [ "$now" -lt 0330 ] || [ "$now" -gt 1800 ]; then
    if [ -f STOP ]; then echo "$(date +%H:%M) STOP exists: STEP 4 skipped" >> $S; exit 0; fi
    echo "$(date +%H:%M) STEP 4 start (STEP 3 finished before 03:30)" >> $S
    bash scripts/flimngo_flute_chain.sh >> $S 2>&1
    echo "$(date +%H:%M) STEP 4 exit $?" >> $S
else
    echo "$(date +%H:%M) STEP 4 skipped: STEP 3 finished after 03:30" >> $S
fi
echo "$(date +%H:%M) all done" >> $S
