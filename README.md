# DiffFLIM

Uncertainty-aware fluorescence lifetime imaging from low-photon data using conditional diffusion models.

## Folder layout

```
diffflim/
├── data/
│   ├── raw/                 # downloaded data, never edited
│   │   ├── flimngo/         # FLIMngo simulated data (.tar → .h5), has ground truth
│   │   ├── flute/           # FLUTE real data (Zenodo 8046636)
│   │   ├── convallaria/     # FLIM LABS Convallaria + fluorescein (Zenodo 15007900)
│   │   └── own/             # your own lab acquisitions (high-photon, dyes)
│   ├── processed/           # cleaned / converted data (e.g. cropped, cached arrays)
│   └── splits/              # train.txt / val.txt / test.txt (lists of file paths)
├── src/
│   ├── data/                # dataset loaders, simulator, photon thinning/splitting
│   ├── models/              # U-Net, later the diffusion model
│   ├── baselines/           # MLE fitting, phasor analysis
│   ├── evaluation/          # metrics (lifetime MAE, calibration)
│   └── utils/               # shared helpers
├── scripts/                 # runnable entry points (download, split, train, evaluate)
├── notebooks/               # exploration and plots
├── configs/                 # experiment settings (YAML)
├── results/figures, tables  # outputs for the paper
├── checkpoints/             # saved model weights
└── logs/                    # training logs
```

## Rules for saving data

1. **`data/raw/` is read-only.** Put downloads here and never modify them. Anything derived goes to `data/processed/`.
2. **One folder per source**, so you always know where a file came from.
3. **Split once, before training**, with `scripts/make_splits.py`. The split files are saved in `data/splits/` and every experiment reads the same ones. Never train on test files.
4. **Results are tied to a config.** Each run saves its config next to its outputs, so any figure can be reproduced.

## Getting started

```bash
pip install -r requirements.txt

# 1. Download 1 FLIMngo tar file (~4.5 GB) and extract it
python scripts/download_flimngo.py --n-files 1

# 2. Check the data loads and look at one sample
python scripts/inspect_data.py

# 3. Make a fixed train/val/test split (70/15/15)
python scripts/make_splits.py
```

Real test data (download manually via the "Download" buttons on each page):
- FLUTE: https://zenodo.org/records/8046636 → save into `data/raw/flute/`
- Convallaria: https://zenodo.org/records/15007900 → save into `data/raw/convallaria/`

## Notes on the FLIMngo data

- 256 time bins, 0.0977 ns per bin, 40 MHz, 10–2500 photons/pixel, up to 4 lifetime components.
- Each decay curve is **normalized to 0–1**, so raw photon counts are lost. Fine for the supervised U-Net baseline; photon thinning and Noise2Noise need un-normalized data (use the FLIMngo simulation notebooks or our own simulator later).
- License CC-BY-4.0. Cite: Kapsiani et al., "Deep learning for fluorescence lifetime predictions enables high-throughput in vivo imaging", JACS 2025.

## Roadmap

- [ ] Stage 0: classical baselines (MLE, phasor)
- [ ] Stage 1: supervised U-Net baseline
- [ ] Stage 2: improve baseline (intensity channel, time compression, multi-patch)
- [ ] Stage 2b: photon-split Noise2Noise baseline
- [ ] Stage 3: conditional diffusion model (DiffFLIM)
- [ ] Stage 4: physics-informed sampling (Poisson likelihood + IRF)
- [ ] Stage 5: real-data validation (thinning, dye standards)
