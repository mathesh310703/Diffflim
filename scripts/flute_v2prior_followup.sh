#!/usr/bin/env bash
# Follow-up of the FLUTE v2-prior night run (24 November 2026). Run from the project root (Git Bash).
# The night chain (post_training_chain_flute.sh) ignored TEST_DIR/CALIB_DIR, so its "flutev2p" results
# are on the OLD fixed FLUTE test and calibration sets. This script adds the planned runs on the v2-prior
# sets and the guided runs with the setting selected on the v2 validation set (both rules: kappa 10,
# first 50 % of the DDIM steps = yesterday's setting B). Second test set never used.
# Graceful stop: touch STOP (checked between steps).
set -e
step() { if [ -f STOP ]; then echo "STOP found before: $1"; exit 0; fi; echo "== $1 $(date +%H:%M)"; }
V2T=data/raw/own/sim_flute_v2_test; V2C=data/raw/own/sim_flute_v2_calib
OLDT=data/raw/own/sim_flute_test; OLDC=data/raw/own/sim_flute_calib
step "v2 sets: diffflim_flute_v2p"
TEST_DIR=$V2T CALIB_DIR=$V2C bash scripts/post_training_chain_flute.sh diffflim_flute_v2p flutev2p_v2s
step "v2 sets: diffflim_flute_v2p_100k"
TEST_DIR=$V2T CALIB_DIR=$V2C bash scripts/post_training_chain_flute.sh diffflim_flute_v2p_100k flutev2p_100k_v2s
step "v2 sets: old model diffflim_flute (test and calibration only)"
python -W ignore scripts/calibrate_diffflim.py --ckpt checkpoints/diffflim_flute.pt --tag flute_v2s_s10 \
    --preds-tag flute_v2s --smooth --photon-estimator otsu_adaptive --test-set $V2T --calib-dir $V2C \
    > logs/chain_flute_v2s_calibrate.txt 2>&1
step "guided B uniform check"
python -W ignore scripts/flute_uniform_check.py --ckpts diffflim_flute_v2p --no-classical --guide-kappa 10 \
    --guide-window 0 0.5 --out results/tables/flute_uniform_check_flutev2p_gB.csv > logs/guid_flutev2p_B_uniform.txt 2>&1
step "guided B calibration on v2 sets"
python -W ignore scripts/calibrate_diffflim.py --ckpt checkpoints/diffflim_flute_v2p.pt --tag flutev2p_gB_v2s_s10 \
    --preds-tag flutev2p_gB_v2s --smooth --photon-estimator otsu_adaptive --test-set $V2T --calib-dir $V2C \
    --guide-kappa 10 --guide-stop 0.5 > logs/guid_flutev2p_B_v2s_calibrate.txt 2>&1
step "guided B calibration on old sets"
python -W ignore scripts/calibrate_diffflim.py --ckpt checkpoints/diffflim_flute_v2p.pt --tag flutev2p_gB_old_s10 \
    --preds-tag flutev2p_gB_old --smooth --photon-estimator otsu_adaptive --test-set $OLDT --calib-dir $OLDC \
    --guide-kappa 10 --guide-stop 0.5 > logs/guid_flutev2p_B_old_calibrate.txt 2>&1
step "guided B real protocol (v2-set factors)"
python -W ignore scripts/eval_real_flute.py --ckpt checkpoints/diffflim_flute_v2p.pt --tag flutev2p_gB \
    --factors results/tables/calibration_factors_flutev2p_gB_v2s_s10.csv --guide-kappa 10 --guide-stop 0.5 \
    > logs/guid_flutev2p_B_real.txt 2>&1
step "scene split, old test set"
python -W ignore scripts/flute_sim_by_scene.py --tags flute flute_gB flutev2p flutev2p_gB_old \
    --out results/tables/flute_sim_by_scene_v2p_oldtest.csv > logs/flute_by_scene_v2p_oldtest.txt 2>&1
step "scene split, v2 test set"
python -W ignore scripts/flute_sim_by_scene.py --test-dir $V2T --tags flute_v2s flutev2p_v2s flutev2p_100k_v2s flutev2p_gB_v2s \
    --out results/tables/flute_sim_by_scene_v2p_v2test.csv > logs/flute_by_scene_v2p_v2test.txt 2>&1
echo "all done $(date +%H:%M)"
