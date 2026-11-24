#!/usr/bin/env bash
# PREPARED 2026-11-24, NOT STARTED: overnight run of DiffFLIM-FLUTE with the FLUTE simulator v2 prior
# (src/data/simulate_v3.py: flute_v2prior_config(), lifetime model "flute_v2": 15 % uniform, 15 %
# near-uniform, 35 % structured mono-exponential scenes over 0.2 to 6 ns, 35 % NAD(P)H-like scenes).
# Everything else as DiffFLIM-FLUTE (checkpoints/diffflim_flute.pt): 200,000 steps, snapshot at 100,000,
# batch 16, lr 2e-4, EMA 0.999, seed 0, the same 568 training IDs (data/splits/hpa_train.txt, identical
# to the list stored in checkpoints/diffflim_flute.pt; validation, test, second-test and calibration
# cells excluded), bin factor 2 (56 bins summed in pairs -> 28 channels; kept on purpose:
# scripts/flute_bin_factor_check.py showed at most 0.7 % loss in the Cramer-Rao bound for tau >= 1 ns,
# and keeping it changes only the prior between the two runs).
#
#   STEP 1  training (about 3 h 15 min at 17.8 it/s)
#   STEP 2  for the final model and the 100k snapshot: scripts/post_training_chain_flute.sh with the
#           v2-prior test and calibration sets (data/raw/own/sim_flute_v2_{test,calib}); the real FLUTE
#           protocol runs inside it with the new factors (unguided)
#   STEP 3  for both: the OLD fixed FLUTE test set (eval_diffflim.py, comparability with diffflim_flute),
#           the uniform check on the test cells (flute_uniform_check.py), and select_guidance.py on the
#           v2-prior validation set (its "none" row is the checkpoint comparison; the rest re-chooses the
#           guidance setting for the new model; nothing is chosen on real data)
# Graceful stop: create a file named STOP in the project root; training saves and stops, later steps are
# skipped. The second test set is never used.
#
# Usage: bash scripts/night_flute_v2prior.sh      (about 4 h 30 min in total)
cd "$(dirname "$0")/.."
S=logs/night_flute_v2prior_run.txt
N=diffflim_flute_v2p
echo "$(date +%H:%M) STEP 1 training start" > $S
python -W ignore scripts/train_diffflim.py --simulator v3flute_v2prior --bin-factor 2 --steps 200000 \
    --snapshot-at-step 100000:100k --name $N --batch 16 --lr 2e-4 --warmup 500 --ema 0.999 --base 64 \
    --prediction v --ckpt-every 15 --workers 16 --seed 0 --ids-file data/splits/hpa_train.txt \
    > logs/${N}_stdout.txt 2>&1
echo "$(date +%H:%M) STEP 1 exit $?: $(cat logs/${N}_status.txt)" >> $S
if ! grep -q "^completed" logs/${N}_status.txt; then echo "$(date +%H:%M) stopped early: rest skipped" >> $S; exit 0; fi
for pair in "$N flutev2p" "${N}_100k flutev2p_100k"; do
    set -- $pair
    if [ -f STOP ]; then echo "$(date +%H:%M) STOP exists: rest skipped" >> $S; exit 0; fi
    TEST_DIR=data/raw/own/sim_flute_v2_test CALIB_DIR=data/raw/own/sim_flute_v2_calib \
        bash scripts/post_training_chain_flute.sh $1 $2 >> $S 2>&1
    echo "$(date +%H:%M) STEP 2 $2 exit $?" >> $S
done
for pair in "$N flutev2p" "${N}_100k flutev2p_100k"; do
    set -- $pair
    if [ -f STOP ]; then echo "$(date +%H:%M) STOP exists: rest skipped" >> $S; exit 0; fi
    python -W ignore scripts/eval_diffflim.py --ckpt checkpoints/$1.pt --tag ${2}_oldtest \
        --test-set data/raw/own/sim_flute_test --n-images 40 > logs/chain_${2}_oldtest.txt 2>&1
    python -W ignore scripts/flute_uniform_check.py --ckpts $1 --out results/tables/flute_uniform_check_$2.csv \
        > logs/chain_${2}_uniform.txt 2>&1
    python -W ignore scripts/select_guidance.py --ckpt checkpoints/$1.pt --sim flute --tag $2 \
        --val-dir data/raw/own/sim_flute_v2_val --kappas 10 100 1000 --windows 0.2 0.5 \
        > logs/select_guidance_$2.txt 2>&1
    echo "$(date +%H:%M) STEP 3 $2 exit $?" >> $S
done
echo "$(date +%H:%M) all done" >> $S
