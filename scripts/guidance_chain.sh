#!/usr/bin/env bash
# Guidance follow-up of 24 November 2026 (src/models/guidance.py). Run from the project root (Git Bash).
# Settings were chosen on simulated validation data only (scripts/select_guidance.py):
#   DiffFLIM-FLUTE  A = kappa 1000, first 20 % of the DDIM steps (rule A: all validation scenes)
#                   B = kappa 10,   first 50 % of the DDIM steps (rule B: mono-exponential validation scenes)
# For each: uniform check on the test cells, recalibration (FLUTE calibration set) and test coverage,
# per-scene split of the test set, fixed real FLUTE protocol. Then DiffFLIM v2: selection on the v3
# validation set (rule A), uniform check (v3, test cells), recalibration on the v3 calibration sets, worm
# protocol. Second test set never used. Graceful stop: touch STOP (checked between steps).
set -e
step() { if [ -f STOP ]; then echo "STOP found before: $1"; exit 0; fi; echo "== $1 $(date +%H:%M)"; }
run_flute() {   # $1 label, $2 kappa, $3 stop
    step "FLUTE $1 uniform check"
    python -W ignore scripts/flute_uniform_check.py --ckpts diffflim_flute --no-classical --guide-kappa $2 \
        --guide-window 0 $3 --out results/tables/flute_uniform_check_guided_$1.csv > logs/guid_flute_$1_uniform.txt 2>&1
    step "FLUTE $1 calibration"
    python -W ignore scripts/calibrate_diffflim.py --ckpt checkpoints/diffflim_flute.pt --tag flute_g$1_s10 \
        --preds-tag flute_g$1 --smooth --photon-estimator otsu_adaptive --test-set data/raw/own/sim_flute_test \
        --calib-dir data/raw/own/sim_flute_calib --guide-kappa $2 --guide-stop $3 > logs/guid_flute_$1_calibrate.txt 2>&1
    step "FLUTE $1 real protocol"
    python -W ignore scripts/eval_real_flute.py --ckpt checkpoints/diffflim_flute.pt --tag flute_g$1 \
        --factors results/tables/calibration_factors_flute_g$1_s10.csv --guide-kappa $2 --guide-stop $3 \
        > logs/guid_flute_$1_real.txt 2>&1
}
run_flute A 1000 0.2
run_flute B 10 0.5
step "FLUTE scene split"
python -W ignore scripts/flute_sim_by_scene.py --tags flute flute_gA flute_gB --out results/tables/flute_sim_by_scene_guided.csv \
    > logs/guid_flute_by_scene.txt 2>&1
step "v2 selection"
python -W ignore scripts/select_guidance.py --ckpt checkpoints/diffflim_v2.pt --sim v3 --tag v2 \
    --kappas 10 100 1000 --windows 0.2 0.5 1.0 > logs/select_guidance_v2.txt 2>&1
echo "all FLUTE steps and v2 selection done $(date +%H:%M)"
