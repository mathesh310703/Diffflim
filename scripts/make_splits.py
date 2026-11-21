"""Make a fixed, reproducible train/val/test split of the FLIMngo .h5 files.

Run once, after downloading. Saves data/splits/{train,val,test}.txt.
Refuses to overwrite existing splits unless --force, so the test set never changes by accident.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import RAW_FLIMNGO, ROOT, SPLITS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    files = sorted((RAW_FLIMNGO / "h5").glob("*.h5"))
    if not files:
        sys.exit("No .h5 files found. Run scripts/download_flimngo.py first.")

    SPLITS.mkdir(parents=True, exist_ok=True)
    if (SPLITS / "test.txt").exists() and not args.force:
        sys.exit("Splits already exist. Use --force only if you really want a new test set.")

    random.Random(args.seed).shuffle(files)
    n = len(files)
    n_test, n_val = int(n * args.test), int(n * args.val)
    parts = {
        "test": files[:n_test],
        "val": files[n_test:n_test + n_val],
        "train": files[n_test + n_val:],
    }
    for name, fs in parts.items():
        rel = [str(f.relative_to(ROOT)) for f in fs]
        (SPLITS / f"{name}.txt").write_text("\n".join(rel) + "\n")
        print(f"{name:5s}: {len(fs)} files")
    print(f"Saved to {SPLITS} (seed={args.seed})")


if __name__ == "__main__":
    main()
