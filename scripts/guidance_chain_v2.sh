#!/usr/bin/env bash
# Guidance follow-up of 24 November 2026, part 2: DiffFLIM v2 with the guidance setting chosen on the v3
# validation set by scripts/select_guidance.py (rule A; results/tables/guidance_selection_v2.csv).
#   1. uniform-lifetime check on simulator v3 (256 bins, FLIMngo IRFs; 30 test-cell scenes), guided and
#      unguided DiffFLIM v2, MLE, binned MLE 5 x 5
#   2. recalibration on the v3 calibration sets (sim_v3_calib + sim_v3_calib_extra, as for
#      calibration_factors_v2_extra_s10.csv) and coverage on the v3 test set
#   3. fixed worm protocol (scripts/eval_real_celegans.py) with the new factors
# Second test set never used. Graceful stop: touch STOP.
# Usage: bash scripts/guidance_chain_v2.sh <kappa> <stop fraction>
set -e
K=$1; W=$2
step() { if [ -f STOP ]; then echo "STOP found before: $1"; exit 0; fi; echo "== $1 $(date +%H:%M)"; }
step "v2 uniform check (v3)"
python -W ignore scripts/flute_uniform_check.py --sim v3 --ckpts diffflim_v2 --guide-kappa $K --guide-window 0 $W \
    --out results/tables/v3_uniform_check_guided_v2.csv > logs/guid_v2_uniform.txt 2>&1
step "v2 calibration"
python -W ignore scripts/calibrate_diffflim.py --ckpt checkpoints/diffflim_v2.pt --tag v2g_extra_s10 --preds-tag v2g \
    --smooth --photon-estimator otsu_adaptive --test-set data/raw/own/sim_v3_test \
    --calib-dir data/raw/own/sim_v3_calib,data/raw/own/sim_v3_calib_extra --guide-kappa $K --guide-stop $W \
    > logs/guid_v2_calibrate.txt 2>&1
step "v2 worm protocol"
python -W ignore scripts/eval_real_celegans.py --ckpt checkpoints/diffflim_v2.pt --tag v2g \
    --factors results/tables/calibration_factors_v2g_extra_s10.csv --guide-kappa $K --guide-stop $W \
    > logs/guid_v2_worm.txt 2>&1
echo "v2 chain done $(date +%H:%M)"
