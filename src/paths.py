"""Central place for all project paths, so no script hard-codes a folder."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_FLIMNGO = RAW / "flimngo"
RAW_FLUTE = RAW / "flute"
RAW_CONVALLARIA = RAW / "convallaria"
RAW_OWN = RAW / "own"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"
CONFIGS = ROOT / "configs"
