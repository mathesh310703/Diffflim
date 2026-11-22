"""Download a small random subset of HPA training images (4 channels each) from Kaggle.

Usage:  python scripts/download_hpa_subset.py --n 300
Needs:  kaggle CLI, logged in (kaggle auth login), competition rules accepted.
Data rules: academic use only, do NOT share or commit these images (data/raw/ is git-ignored).
"""
import argparse, csv, io, random, subprocess, zipfile
from pathlib import Path
import time

COMP = "human-protein-atlas-image-classification"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "hpa"
CHANNELS = ["red", "green", "blue", "yellow"]


def read_ids():
    csv_path = OUT / "train.csv"
    zip_path = OUT / "train.csv.zip"
    if csv_path.exists():
        text = csv_path.read_text()
    elif zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            text = z.read("train.csv").decode()
    else:
        raise SystemExit("train.csv not found - download it first.")
    return [row["Id"] for row in csv.DictReader(io.StringIO(text))]


def fetch(name, pause=1.5, max_tries=6):
    """Download one file politely; wait and retry when Kaggle rate-limits (429)."""
    target = OUT / name
    if target.exists():
        return True
    wait = 60
    for attempt in range(max_tries):
        r = subprocess.run(["kaggle", "competitions", "download", "-c", COMP, "-f", name, "-p", str(OUT), "-q"],
                           capture_output=True, text=True)
        for z in [target.with_name(target.name + ".zip"), OUT / (Path(name).name + ".zip")]:
            if z.exists():
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(target.parent)
                z.unlink()
        if target.exists():
            time.sleep(pause)
            return True
        msg = (r.stdout + r.stderr)
        if "429" in msg or "rate limit" in msg.lower():
            print(f"  rate limited, waiting {wait} s ...")
            time.sleep(wait)
            wait = min(wait * 2, 900)
            continue
        print("  failed:", name, msg.strip()[:200])
        return False
    print("  gave up:", name)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of samples (each = 4 PNGs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = read_ids()
    random.Random(args.seed).shuffle(ids)
    chosen = ids[: args.n]
    (OUT / "subset_ids.txt").write_text("\n".join(chosen))
    ok = 0
    for i, sid in enumerate(chosen, 1):
        if all(fetch(f"train/{sid}_{c}.png") for c in CHANNELS):
            ok += 1
        if i % 10 == 0:
            print(f"{i}/{len(chosen)} samples done ({ok} complete)")
    print(f"Finished: {ok}/{len(chosen)} complete samples in {OUT}")


if __name__ == "__main__":
    main()