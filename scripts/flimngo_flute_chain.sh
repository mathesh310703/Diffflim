#!/usr/bin/env bash
# STEP 4 of the overnight FLUTE chain: retrain the FLIMngo architecture on the FLUTE simulator
# (150 min, selection on the FLUTE validation set), then evaluate it on the simulated FLUTE test set
# and on the real FLUTE data (fixed protocol of scripts/eval_real_flute.py). Run from the project root.
set -e
python -W ignore scripts/train_flimngo_flute.py --minutes 150 --name flimngo_flute > logs/flimngo_flute_stdout.txt 2>&1
cat logs/flimngo_flute_status.txt
if ! grep -q "^completed" logs/flimngo_flute_status.txt; then echo "FLIMngo-FLUTE stopped early: evaluation skipped"; exit 0; fi
python -W ignore scripts/eval_flimngo_flute.py > logs/eval_flimngo_flute.txt 2>&1
echo "FLIMngo-FLUTE simulated test done"
python -W ignore scripts/eval_real_flute.py --flimngo-ckpt checkpoints/flimngo_flute.pt --tag flimngo_flute \
    > logs/eval_real_flute_flimngo.txt 2>&1
echo "FLIMngo-FLUTE real data done"
