#!/usr/bin/env bash
# Post-training chain for a DiffFLIM checkpoint trained on the FLUTE configuration of simulator v3
# (--simulator v3flute_nadh; 56 bins of 0.223 ns). Run from the project root (Git Bash).
#   1. evaluation on the simulated FLUTE test set (data/raw/own/sim_flute_test, 40 images per level,
#      levels 10 to 500) with the classical baselines (periodic FLUTE forward model, each image's IRF);
#      10 DDIM steps x 10 samples
#   2. calibration on the FLUTE calibration set (data/raw/own/sim_flute_calib: 20 calibration + 54
#      extra calibration cells, 200 images per level), per-level factors as a quadratic in
#      log(estimated photons), photon estimate otsu_adaptive
#      -> results/tables/calibration_factors_<tag>_s10.csv, coverage_calibrated_<tag>_s10.csv
#   3. FLUTE real data with the fixed protocol and the new factors (scripts/eval_real_flute.py)
# The second test set is not used.
#
# Optional environment variables (24 November 2026, for the FLUTE v2 prior): TEST_DIR and CALIB_DIR
# replace the test and calibration folders (defaults: the sets above).
#
# Usage: bash scripts/post_training_chain_flute.sh <checkpoint name> <tag> [quick]
# Example: bash scripts/post_training_chain_flute.sh diffflim_flute flute
set -e
NAME=$1
TAG=$2
CKPT=checkpoints/$NAME.pt
NIMG=40; NCAL=0; NTEST=0
TEST_DIR=${TEST_DIR:-data/raw/own/sim_flute_test}
CALIB_DIR=${CALIB_DIR:-data/raw/own/sim_flute_calib}
if [ "$3" = "quick" ]; then NIMG=2; NCAL=3; NTEST=2; fi
test -f "$CKPT" || { echo "missing $CKPT"; exit 1; }
python -W ignore scripts/eval_diffflim.py --ckpt "$CKPT" --tag "$TAG" --test-set data/raw/own/sim_flute_test \
    --n-images $NIMG > "logs/chain_${TAG}_eval.txt" 2>&1
echo "eval done"
python -W ignore scripts/calibrate_diffflim.py --ckpt "$CKPT" --tag "${TAG}_s10" --preds-tag "$TAG" --smooth \
    --photon-estimator otsu_adaptive --test-set data/raw/own/sim_flute_test \
    --calib-dir data/raw/own/sim_flute_calib --n-calib $NCAL --n-test $NTEST > "logs/chain_${TAG}_calibrate.txt" 2>&1
echo "calibration done"
if [ "$3" != "quick" ]; then
    python -W ignore scripts/eval_real_flute.py --ckpt "$CKPT" --tag "$TAG" \
        --factors "results/tables/calibration_factors_${TAG}_s10.csv" > "logs/chain_${TAG}_real.txt" 2>&1
    echo "real data done"
fi
echo "chain done"
