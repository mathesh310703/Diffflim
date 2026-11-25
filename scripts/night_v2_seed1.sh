#!/usr/bin/env bash
# Overnight run of 24/25 November 2026: DiffFLIM v2 with a second random seed, to measure the
# run-to-run variation. Identical to DiffFLIM v2 (checkpoints/diffflim_v2.pt; settings from its stored
# args: simulator v3, v-prediction, base 64, batch 16, lr 2e-4, warmup 500, EMA 0.999, 16 data workers,
# the 246 training IDs of data/splits/hpa_train_diffflim_v2.txt, default photon and lifetime ranges,
# 186,116 steps, snapshot at step 92,168 = the step of the 3-hour snapshot diffflim_v2_180min.pt) except
# --seed 1 (torch seed and data-stream seed). DiffFLIM v2 was stopped by time (360 min); here the run is
# stopped at the same step count.
#   STEP 1  training (about 6 h at 8.6 it/s); a speed check after 10 minutes writes a warning to the
#           status log if the rate is below 7 it/s
#   STEP 2  scripts/post_training_chain.sh for the final model (tag v2_seed1) and the snapshot (v2_seed1_3h):
#           v3 test set (10 DDIM steps x 10 samples), calibration on the 74 calibration cells, the fixed
#           C. elegans protocol, versions table
#   STEP 3  scripts/compare_seeds.py (seed variation table and figure)
# Graceful stop: create a file named STOP in the project root (training saves and stops; later steps are
# skipped). The second test set is never used.
cd "$(dirname "$0")/.."
S=logs/night_v2_seed1_run.txt
N=diffflim_v2_seed1
echo "$(date +%H:%M) STEP 1 training start" > $S
( sleep 600
  L=$(tail -1 logs/${N}.csv); R=$(echo "$L" | cut -d, -f5); STEP=$(echo "$L" | cut -d, -f1)
  ETA=$(python -c "import datetime as d; r=float('$R'); s=int('$STEP'); print((d.datetime.now()+d.timedelta(seconds=(186116-s)/max(r,0.1))).strftime('%d.%m. %H:%M'))")
  if python -c "import sys; sys.exit(0 if float('$R') < 7 else 1)"; then
      echo "$(date +%H:%M) WARNING: speed after 10 min is $R it/s (below 7; likely a power-saving mode or battery). Step $STEP; expected end of training at this rate: $ETA. Training continues." >> $S
  else
      echo "$(date +%H:%M) speed check: $R it/s at step $STEP; expected end of training: $ETA" >> $S
  fi ) &
python -W ignore scripts/train_diffflim.py --simulator v3 --steps 186116 --snapshot-at-step 92168:3h \
    --name $N --batch 16 --lr 2e-4 --warmup 500 --ema 0.999 --base 64 --prediction v --ckpt-every 15 \
    --workers 16 --seed 1 --ids-file data/splits/hpa_train_diffflim_v2.txt > logs/${N}_stdout.txt 2>&1
echo "$(date +%H:%M) STEP 1 exit $?: $(cat logs/${N}_status.txt)" >> $S
if ! grep -q "^completed" logs/${N}_status.txt; then echo "$(date +%H:%M) stopped early: rest skipped" >> $S; exit 0; fi
for pair in "$N v2_seed1" "${N}_3h v2_seed1_3h"; do
    set -- $pair
    if [ -f STOP ]; then echo "$(date +%H:%M) STOP exists: rest skipped" >> $S; exit 0; fi
    bash scripts/post_training_chain.sh $1 $2 >> $S 2>&1
    echo "$(date +%H:%M) STEP 2 $2 exit $?" >> $S
done
if [ -f STOP ]; then echo "$(date +%H:%M) STOP exists: rest skipped" >> $S; exit 0; fi
python -W ignore scripts/compare_seeds.py > logs/compare_seeds.txt 2>&1
echo "$(date +%H:%M) STEP 3 exit $?" >> $S
echo "$(date +%H:%M) all done" >> $S
