# DiffFLIM – Status Report (2026-11-21)

**Summary:** The project is a clean, well-organised **skeleton**. The folder layout, path handling, FLIMngo download/split/inspect scripts and an h5 loader exist. There is **no data, no model, no baseline, no simulator and no results** yet. Total size: **~32 KB, 26 files** (22 of them empty `.gitkeep` / `__init__.py`).

## 1. Folder structure
Project root: `d:\diffflim\diffflim\` (nested inside `d:\diffflim\`, which holds nothing else).
```
diffflim/                      ~32 KB total
├── README.md (3.5 KB)  requirements.txt (71 B)  .gitignore (155 B)
├── configs/unet_baseline.yaml (425 B)
├── scripts/  download_flimngo.py (1.9 KB)  inspect_data.py (1.8 KB)  make_splits.py (1.6 KB)
├── src/      paths.py (0.5 KB)  __init__.py (0 B)
│   ├── data/        flimngo.py (1.4 KB)  __init__.py (0 B)
│   ├── baselines/   __init__.py (0 B)    ← empty package
│   ├── models/      __init__.py (0 B)    ← empty package
│   ├── evaluation/  __init__.py (0 B)    ← empty package
│   └── utils/       __init__.py (0 B)    ← empty package
├── data/
│   ├── raw/ flimngo/ flute/ convallaria/ own/   (each: only .gitkeep)
│   ├── processed/ (.gitkeep)   splits/ (.gitkeep)
├── notebooks/ (.gitkeep)   checkpoints/ (.gitkeep)   logs/ (.gitkeep)
└── results/ figures/ (.gitkeep)   tables/ (.gitkeep)
```
Every top-level folder except `scripts/`, `src/`, `configs/` is **0 bytes**.

## 2. Data
| Dataset | Status |
|---|---|
| Simulated (own) | **None.** No simulator code either. |
| FLIMngo | **Empty** `data/raw/flimngo/`. A download script exists (HF `SofiaKaps/FLIMngo_data`, ~4.5 GB per tar → .h5). |
| FLUTE (Zenodo 8046636) | **Empty** folder. README says to download it by hand. |
| Convallaria (Zenodo 15007900) | **Empty** folder. README says to download it by hand. |
| Own lab data / .sdt | **Empty** folder. |
| Splits / processed | **Empty.** |

The README notes that FLIMngo decays are **normalised to 0–1**, so photon counts are lost. That makes FLIMngo unusable for photon thinning or for a Poisson likelihood, and it is another reason to write your own simulator.

## 3. Code
| File | What it does | State |
|---|---|---|
| `src/paths.py` | Central path constants (no hard-coded absolute paths) | Finished |
| `src/data/flimngo.py` | `load_h5` (keys `tpsf`, `tau_maps`), `read_split`, `FLIMngoDataset` → (T,H,W) decay, (1,H,W) τ | Finished but untested (no data). No intensity target. |
| `scripts/download_flimngo.py` | Lists/downloads FLIMngo tars from Hugging Face and extracts .h5 | Finished, **entry point #1** |
| `scripts/inspect_data.py` | Loads first .h5, prints shapes, saves `results/figures/inspect_sample.png` | Finished, **entry point #2** |
| `scripts/make_splits.py` | Seeded 70/15/15 split → `data/splits/*.txt`, won't overwrite the test split unless you pass `--force` | Finished, **entry point #3** |
| `configs/unet_baseline.yaml` | U-Net config (256 in-channels → 1 lifetime out, L1, 50 epochs) | Placeholder; nothing reads it |
| `src/{baselines,models,evaluation,utils}/` | — | **Empty** |
| `notebooks/` | — | **Empty** |

No training or evaluation script exists yet.

## 4. Models and results
None. There are no weights, checkpoints, logs, figures or tables, only `.gitkeep` files.

## 5. Environment
- `requirements.txt` (no versions pinned): numpy, scipy, h5py, torch, matplotlib, pyyaml, tqdm, tifffile, huggingface_hub
- `python --version` → **Python 3.11.0** (system install, no venv/conda found)
- Installed: torch 2.5.1+cu121, torchvision 0.20.1, numpy 2.4.5, scipy 1.17.1, matplotlib 3.10.9, PyYAML 6.0.3, tqdm 4.67.3
- **Not installed:** `h5py`, `tifffile`, `huggingface_hub`, `sdtfile`, `diffusers`
- GPU: **NVIDIA RTX 4060 Laptop, 8 GB**, driver 621.24; `torch.cuda.is_available()` → True (CUDA 12.1)

## 6. Status against the plan
| Item | Status | Evidence |
|---|---|---|
| Data loading: simulated | Not started | no simulator or loader |
| Data loading: FLIMngo | Partly done | loader and download script written, no data, never run |
| Data loading: real .sdt | Not started | no `sdtfile` code or dependency |
| Decay simulation (Poisson, IRF) | Not started | `src/data/` has only flimngo.py |
| Photon thinning | Not started | only mentioned in README |
| Baseline: MLE | Not started | `src/baselines/` empty |
| Baseline: phasor | Not started | `src/baselines/` empty |
| Baseline: FLIMngo pretrained | Not started | only the dataset is handled, not the model or repo |
| Baseline: U-Net | Partly done (config only) | yaml exists, no model or training code |
| Conditional diffusion (τ + intensity) | Not started | `src/models/` empty |
| N samples → mean + uncertainty | Not started | — |
| Calibration / coverage | Not started | `src/evaluation/` empty |
| Evaluation plots | Not started | `results/` empty |

## 7. Problems
- **`h5py` is missing**, so `src/data/flimngo.py`, `inspect_data.py` and the dataset class fail on import. `huggingface_hub` is missing, so `download_flimngo.py` fails. Fix: `pip install -r requirements.txt`.
- **`sdtfile` and `diffusers`** are not in requirements and not installed. They will be needed for real `.sdt` data and possibly for the diffusion model.
- There is **no data**, so every script except `--list` exits with "No .h5 files found".
- The loader only returns τ. The plan also needs an **intensity** target, and the FLIMngo normalisation removes photon counts.
- `make_splits.py` writes paths relative to the project root, but `read_split` does not prepend `ROOT`. It only works when you run from the project root.
- The project is nested at `d:\diffflim\diffflim`. Open/run from the inner folder. Not a git repo yet (.gitignore is present).
- No unpinned-version problems yet, but numpy 2.x is new and you should pin versions once things work.
- No TODO/FIXME comments. No hard-coded absolute paths (good).

## 8. Suggested next 3 steps
1. **Write your own simulator** (`src/data/simulate.py`): random τ/intensity maps → mono/bi-exponential decays convolved with a Gaussian IRF → Poisson counts at a chosen photon level. You get exact ground truth for **both τ and intensity**, raw counts for thinning (`np.random.binomial`), and the data for the calibration check. Save a small fixed test set.
2. **Implement the MLE and phasor baselines** (`src/baselines/`) on the simulated data, plus `src/evaluation/metrics.py` (τ MAE/bias vs photon count). This gives you the "error vs photons" plot early and a reference to beat.
3. **Build the conditional diffusion model** (a small U-Net denoiser conditioned on the noisy decay histogram, outputting τ + intensity). Sample it N≈20 times → mean/std maps, then compute coverage of the 68%/95% intervals on the simulated set. Install `h5py` and fetch one FLIMngo tar in parallel to set up the FLIMngo comparison later.
