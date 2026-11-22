#!/usr/bin/env bash
# Post-training chain for a DiffFLIM checkpoint trained on simulator v3 (run from the project root,
# Git Bash). Steps, each logged to logs/chain_<tag>_*.txt:
#   1. evaluation on the fixed v3 test set with all baselines (eval_diffflim.py, 10 DDIM steps x 10 samples)
#   2. calibration on both v3 calibration sets (74 cells), per-level factors, photon estimate otsu_adaptive
#      -> results/tables/calibration_factors_<tag>_extra_s10.csv, coverage_calibrated_<tag>_extra_s10.csv
#   3. real data (FLIMngo C. elegans) with the fixed protocol and the new factors (eval_real_celegans.py)
#   4. versions table (model_versions.py) -> results/tables/model_versions.csv
# The second test set (hpa_test2 / sim_v3_test2) is NOT used.
#
# Usage:
#   bash scripts/post_training_chain.sh <checkpoint name> <tag>          # full chain (about 30 min)
#   bash scripts/post_training_chain.sh <checkpoint name> <tag> quick    # dry run: tiny image counts
# Example: bash scripts/post_training_chain.sh diffflim_v3cells v3cells
set -e
NAME=$1
TAG=$2
CKPT=checkpoints/$NAME.pt
if [ "$3" = "quick" ]; then NIMG=2; NCAL=3; NTEST=2; else NIMG=40; NCAL=0; NTEST=0; fi
test -f "$CKPT" || { echo "missing $CKPT"; exit 1; }

python -W ignore scripts/eval_diffflim.py --ckpt "$CKPT" --tag "$TAG" --test-set data/raw/own/sim_v3_test \
    --n-images $NIMG > "logs/chain_${TAG}_eval.txt" 2>&1
echo "eval done"
python -W ignore scripts/calibrate_diffflim.py --ckpt "$CKPT" --tag "${TAG}_extra_s10" --preds-tag "$TAG" --smooth \
    --photon-estimator otsu_adaptive --test-set data/raw/own/sim_v3_test \
    --calib-dir data/raw/own/sim_v3_calib,data/raw/own/sim_v3_calib_extra \
    --n-calib $NCAL --n-test $NTEST > "logs/chain_${TAG}_calibrate.txt" 2>&1
echo "calibration done"
python -W ignore scripts/eval_real_celegans.py --ckpt "$CKPT" --tag "$TAG" \
    --factors "results/tables/calibration_factors_${TAG}_extra_s10.csv" > "logs/chain_${TAG}_real.txt" 2>&1
echo "real data done"
python -W ignore scripts/model_versions.py > "logs/chain_${TAG}_versions.txt" 2>&1
cat "logs/chain_${TAG}_versions.txt"
echo "chain done"
