"""Download and extract FLIMngo tar files into data/raw/flimngo/.

Usage:
    python scripts/download_flimngo.py --n-files 1      # one tar (~4.5 GB)
    python scripts/download_flimngo.py --list           # just list the files
"""
import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import RAW_FLIMNGO  # noqa: E402

from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

REPO = "SofiaKaps/FLIMngo_data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=1, help="how many tar files to download")
    ap.add_argument("--list", action="store_true", help="only list available tar files")
    ap.add_argument("--keep-tar", action="store_true", help="keep the .tar after extracting")
    args = ap.parse_args()

    tars = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".tar"))
    print(f"{len(tars)} tar files available")
    if args.list:
        print("\n".join(tars))
        return

    tar_dir = RAW_FLIMNGO / "_tar"
    h5_dir = RAW_FLIMNGO / "h5"
    tar_dir.mkdir(parents=True, exist_ok=True)
    h5_dir.mkdir(parents=True, exist_ok=True)

    for name in tars[: args.n_files]:
        print(f"Downloading {name} ...")
        path = Path(hf_hub_download(REPO, name, repo_type="dataset", local_dir=tar_dir))
        print(f"Extracting into {h5_dir} ...")
        with tarfile.open(path) as tf:
            # only extract regular .h5 files, flattened, to avoid unsafe paths
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(".h5"):
                    m.name = Path(m.name).name
                    tf.extract(m, h5_dir)
        if not args.keep_tar:
            path.unlink()

    n = len(list(h5_dir.glob("*.h5")))
    print(f"Done. {n} .h5 files in {h5_dir}")


if __name__ == "__main__":
    main()
