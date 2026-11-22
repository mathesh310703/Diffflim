# Guide to the result files

This file explains every figure in `results/figures/` and every table in `results/tables/`: what it shows, how to read it, which data, model and version it belongs to, which script made it, and its status. It is kept up to date whenever a result file is added or replaced (rule in `CLAUDE.md`). State: 22 November 2026, 35 figures and 36 tables.

Status labels:

- **main result**: the current number or figure to quote for this question.
- **supporting check**: a valid result that supports or qualifies a main result (sensitivity analysis, older setting, diagnostic).
- **superseded**: replaced by a later result (other data, wrong information given to a method, older calibration); kept as a record and referenced in the project log.

Figures derived from HPA images (`simulate_v3_preview.png`, the `*_examples.png` of v3 models) show crops of Human Protein Atlas images and are for academic use only.

## Naming scheme and tags

File names are `<kind>_<tag>.<ext>`. The kind says what is shown (`error_vs_photons`, `calibration`, `coverage`, `diffflim`, ...), the tag says which model, data set or setting.

| Tag | Meaning |
|---|---|
| (no tag) | first versions on simulator v2 data (`baselines.csv`, `error_vs_photons.png`) |
| `v1` | DiffFLIM v1 (v-prediction, trained 60 min on simulator v2 data), evaluated on the v2 test set |
| `v1_on_v3` | DiffFLIM v1 (trained on v2) evaluated on the paired v3 test set without retraining |
| `v2` | DiffFLIM v2 (trained 360 min on simulator v3 data, `checkpoints/diffflim_v2.pt`), 50 DDIM steps; for calibration files: 20 calibration cells, photon estimate `otsu_log` |
| `v2_3h` | the 3-hour snapshot of the same run (`checkpoints/diffflim_v2_180min.pt`), 50 DDIM steps |
| `v3` | v3 test set (`baselines_v3.csv`, `flimngo_v3.csv`) or the retrained FLIMngo (`training_flimngo_v3.png`) |
| `baselines_v3` | classical baselines on the paired v3 test set |
| `v2_flimngo` | DiffFLIM v2 compared with the pretrained FLIMngo given the ground-truth mask |
| `compare_v2` | the main comparison: all methods, same information, DiffFLIM with 10 steps |
| `v2_newest` | calibration of DiffFLIM v2 with the 20 original calibration cells and the new photon estimate `otsu_adaptive` |
| `v2_extra_oldest` | calibration with all 74 cells (20 + 54 extra) and the old photon estimate `otsu_log` |
| `v2_extraonly` | calibration with the 54 extra cells only, `otsu_adaptive` |
| `v2_extra` | calibration with all 74 cells, `otsu_adaptive`, 50 DDIM steps |
| `v2_extra_s10` | the same with 10 DDIM steps and 10 samples (the main calibration) |
| `s10`, `s10_n10` | sampler setting: 10 DDIM steps (and 10 samples); also used in the cache names `data/processed/preds_v2_s{steps}_n{samples}_*.npz` |
| `cellboot` | coverage intervals from a bootstrap over HPA cells (`coverage_cellboot_*`) |
| `p100`, `p1000` | photon level of a preview figure (100 or 1000 photons per foreground pixel) |
| `real_celegans_..._<tag>` | real-data evaluation on FLIMngo's raw *C. elegans* recordings; the tag names the DiffFLIM model (`v2` = `checkpoints/diffflim_v2.pt`, 10 DDIM steps × 10 samples, calibration `v2_extra_s10`); the MLE, binned MLE and FLIMngo rows do not depend on the tag |
| `simref` | split-half reference computed on simulated data with the same protocol as a real-data test |
| `smoke` | two-minute smoke tests; their outputs (`diffflim_smoke.csv`, `coverage_smoke.csv`, `error_vs_photons_smoke.png`, `diffflim_smoke_examples.png`) were deleted and are no longer in `results/` |

Photon levels are always 10, 20, 50, 100, 500 and 1000 mean signal photons per foreground pixel.

No result file uses the second test set (`data/splits/hpa_test2.txt`, `data/raw/own/sim_v3_test2/`); it is reserved for the final evaluation. Result files made from it must carry the tag `test2`.

## Columns used in many tables

- `method`: `MLE`, `Phasor`, `BinMLE3`, `BinMLE5` (binned MLE with 3 × 3 or 5 × 5 neighbourhoods), `DiffFLIM`, `FLIMngo ...`.
- `photons`: photon level, or `all` for all levels pooled.
- `subset`: pixel set. `fg` or `all_fg` = all true foreground pixels; `mle_valid` = foreground pixels where the MLE is finite (at least one photon); `bright` = foreground pixels with true intensity at least 20 % of the crop's intensity range above its dimmest foreground pixel (in `diffflim_*.csv` additionally MLE-valid).
- `n_images`: images per photon level; `n_pixels`: number of pixels in the subset.
- `tau_mare_pct`, `int_mare_pct`: median absolute relative error of lifetime or intensity, in %.
- `tau_bias_pct`, `int_bias_pct`: median signed relative error, in % (positive = overestimate).
- `tau_invalid_frac`: fraction of subset pixels with a NaN or infinite lifetime estimate (excluded from the medians).
- `gt_jitter_median_ns`, `gt_jitter_median_pct`: median absolute difference between the true lifetime of a pixel and the median of its 3 × 3 foreground neighbourhood, in ns and in % of τ (roughness of the ground truth).
- `tau_mare_vs_tau_int_pct`: lifetime error measured against the intensity-weighted instead of the amplitude-weighted true lifetime (v3 only).
- Coverage values are fractions between 0 and 1; `nominal_coverage` is 0.9 or 0.68.

## Figures (`results/figures/`)

### Simulator previews

- **`simulate_preview_p1000.png`**: one simulator v1 image at 1000 photons per pixel (photon level then defined over the whole image): true lifetime, true intensity, summed counts, and three example pixel histograms (grey steps) with their expected curves (coloured). Made by `scripts/simulate_preview.py --photons 1000` with the v1 code, which no longer exists. Status: superseded (simulator v1).
- **`simulate_preview_p100.png`**: the same layout for simulator v2 at 100 photons per foreground pixel, with the foreground mask as a white outline. Made by `scripts/simulate_preview.py --photons 100 --seed 0`. Status: supporting check.
- **`simulate_v3_preview.png`**: one simulator v3 sample at 100 photons. Top row: the four HPA channels of the crop. Middle row: true (amplitude-weighted) lifetime, true intensity, summed counts, number of lifetime components; circles mark three example pixels. Bottom row: their histograms with expected curves, and the 12 IRFs of the library (thin) with the IRF used (bold, with its shift). Made by `scripts/simulate_v3_preview.py --photons 100 --seed 3`. Status: supporting check. HPA-derived (academic use only).

### Error against photon level

All error figures have the photon level on a logarithmic x-axis and the median absolute relative error (%) on a logarithmic y-axis. Where present, a grey dotted horizontal line in lifetime panels marks the median ground-truth jitter, and a grey dotted curve in intensity panels the per-pixel Poisson limit 0.674/√N (the median absolute relative error of a Poisson count with mean N).

- **`error_vs_photons.png`**: MLE and phasor lifetime and intensity error on the v2 test set (40 images per level). Made by the first version of `scripts/eval_baselines.py` (no arguments). Status: superseded (v2 data).
- **`error_vs_photons_v1.png`**: DiffFLIM v1 against MLE, phasor and binned MLE on the v2 test set (8 images per level, 50 DDIM steps). 2 × 2 panels: rows all foreground (baselines on MLE-valid pixels; DiffFLIM solid on all, dotted on MLE-valid pixels) and bright pixels; columns lifetime and intensity. Made by `scripts/eval_diffflim.py --ckpt checkpoints/diffflim_v1.pt --tag v1`. Status: superseded (v2 data).
- **`error_vs_photons_baselines_v3.png`**: MLE, phasor, binned MLE 3 × 3 and 5 × 5 on the paired v3 test set (40 images per level). 2 × 2 panels: rows all foreground and bright pixels, columns lifetime and intensity. Made by `scripts/eval_baselines.py --test-set data/raw/own/sim_v3_test --tag v3`. Status: supporting check (the same numbers appear in `compare_v2`).
- **`error_vs_photons_v1_on_v3.png`**: DiffFLIM v1 (trained on v2) on the paired v3 test set with the baselines, layout as `error_vs_photons_v1.png`, 8 images per level. Made by `scripts/eval_diffflim.py --tag v1_on_v3 --test-set data/raw/own/sim_v3_test`. Status: supporting check (shows that v1 does not transfer).
- **`error_vs_photons_v2.png`**: DiffFLIM v2 (6 h, 50 steps) with the baselines on the paired v3 test set, 40 images per level, layout as `error_vs_photons_v1.png`. Made by `scripts/eval_diffflim.py --ckpt checkpoints/diffflim_v2.pt --tag v2 --test-set data/raw/own/sim_v3_test --n-images 40 --steps 50`. Status: supporting check (50 steps; superseded for the main comparison by `error_vs_photons_compare_v2.png`).
- **`error_vs_photons_v2_3h.png`**: the same for the 3-hour snapshot (tag `v2_3h`). Status: supporting check.
- **`error_vs_photons_v2_flimngo.png`**: lifetime error only; left all foreground, right bright pixels; MLE, phasor, binned MLE 3 × 3 and 5 × 5, DiffFLIM v2 (6 h, 50 steps) and the pretrained FLIMngo with the ground-truth mask (crosses). Made by `scripts/eval_flimngo.py`. Status: superseded for the comparison (FLIMngo had the ground-truth mask, which DiffFLIM does not get); still valid as an upper bound for FLIMngo.
- **`error_vs_photons_compare_v2.png`**: the main comparison on the paired v3 test set (40 images per level). 2 × 2 panels: rows all foreground and bright pixels, columns lifetime and intensity. Lines: MLE (blue circles), binned MLE 5 × 5 (red triangles), FLIMngo pretrained (brown crosses, dashed) and FLIMngo retrained on v3 (orange crosses), both with the recall-oriented mask `flimngo_recall`, DiffFLIM v2 3 h (cyan diamonds, dashed) and 6 h (purple diamonds), both with 10 DDIM steps and 10 samples. FLIMngo has no intensity output and is absent from the intensity panels. Made by `scripts/eval_compare_v2.py`. Status: **main result**.

### DiffFLIM examples

Each example figure has one row per photon level (10 and 100 photons, first test image) and five panels: input summed counts, true lifetime, DiffFLIM mean lifetime (with its error in the title), DiffFLIM standard deviation of the lifetime, and the MLE lifetime (with its error). The lifetime colour scale is the 1st to 99th percentile of the truth; the white outline is the foreground mask. All with 50 DDIM steps.

- **`diffflim_v1_examples.png`**: DiffFLIM v1 on v2 data. Made by `eval_diffflim.py --tag v1`. Status: superseded (v2 data).
- **`diffflim_v1_on_v3_examples.png`**: DiffFLIM v1 on v3 data (almost flat lifetime map at 100 photons). Tag `v1_on_v3`. Status: supporting check. HPA-derived.
- **`diffflim_v2_examples.png`**: DiffFLIM v2 (6 h) on v3 data. Tag `v2`. Status: supporting check (the main qualitative example). HPA-derived.
- **`diffflim_v2_3h_examples.png`**: the 3-hour snapshot. Tag `v2_3h`. Status: supporting check. HPA-derived.

### Calibration and coverage

Two figure types are made by `scripts/calibrate_diffflim.py` for every calibration tag:

- `calibration_<tag>.png`: two panels (lifetime, intensity). x-axis: promised coverage of the interval (10 to 95 %); y-axis: observed coverage on the test set, all photon levels pooled. Circles: uncalibrated intervals mean ± z · std (Gaussian z); squares: global conformal factor; diamonds: per-level factors. The dashed diagonal is perfect calibration.
- `calibration_<tag>_vs_photons.png`: 2 × 2 panels; rows nominal 90 % and 68 %, columns lifetime and intensity. x-axis: photon level (log); y-axis: observed coverage (%) with 95 % image-bootstrap error bars; the three variants are slightly offset horizontally for readability; the dashed line is the target.

| Files | Model | Calibration set | Photon estimate | DDIM steps | Status |
|---|---|---|---|---|---|
| `calibration_v1.png`, `calibration_v1_vs_photons.png` | DiffFLIM v1 | v2 simulator, 20 images per level | `otsu_log`, linear interpolation | 50 | superseded (v2 data; made by an earlier version of the script whose table has no confidence intervals) |
| `calibration_v2.png`, `calibration_v2_vs_photons.png` | DiffFLIM v2 (6 h) | 20 calibration cells (600 images) | `otsu_log` | 50 | superseded (fewer cells, old estimate) |
| `calibration_v2_newest.png`, `calibration_v2_newest_vs_photons.png` | DiffFLIM v2 | 20 cells | `otsu_adaptive` | 50 | supporting check (effect of the photon estimate) |
| `calibration_v2_extra_oldest.png`, `calibration_v2_extra_oldest_vs_photons.png` | DiffFLIM v2 | 74 cells (1200 images) | `otsu_log` | 50 | supporting check |
| `calibration_v2_extraonly.png`, `calibration_v2_extraonly_vs_photons.png` | DiffFLIM v2 | 54 extra cells only | `otsu_adaptive` | 50 | supporting check (cells never seen by v2) |
| `calibration_v2_extra.png`, `calibration_v2_extra_vs_photons.png` | DiffFLIM v2 | 74 cells | `otsu_adaptive` | 50 | supporting check (main calibration before the 10-step default) |
| `calibration_v2_extra_s10.png`, `calibration_v2_extra_s10_vs_photons.png` | DiffFLIM v2 | 74 cells | `otsu_adaptive` | 10 | **main result** |

The image-bootstrap error bars in these figures are too narrow (images of the same cell are not independent); cell-level intervals are in `coverage_cellboot_*.csv`.

### Training curves

- **`loss_v2.png`**: training of DiffFLIM v2 (360 min). Left: training loss (MSE of the v-prediction) on a logarithmic axis against training time; light blue the values logged every 50 steps, dark blue the centred running median over 100 log entries, dotted vertical line the 3-hour snapshot. Right: percentage of each logging interval in which the GPU waited for data, with the 50 % limit as a dashed red line. Made by `scripts/plot_training_curves.py diffflim` from `logs/diffflim_v2.csv` (first made with an inline command; regenerated with the same data). Status: supporting check.
- **`training_flimngo_v3.png`**: training of the FLIMngo architecture on v3 (180 min). Left: training MSE (light blue: logged values; dark blue: moving mean over 20 log entries) and validation MSE (orange) in ns², logarithmic y-axis, against training time. Right: median relative lifetime error on the validation set. Made by `scripts/plot_training_curves.py flimngo` from `logs/flimngo_v3.csv` and `logs/flimngo_v3_val.csv` (identical to the first, inline-made version). Status: supporting check.

### Speed, masks and bias

- **`speed_tradeoff_v2.png`**: DiffFLIM v2 (6 h) with 50, 20 and 10 DDIM steps (labels at the points) and 10 (circles) or 5 (squares) samples, on 60 test images from 10 cells. x-axis: time per image (ms, log). Panels: lifetime error, intensity error, and the coverage of the 90 % interval after calibration with the global factor (lifetime and intensity; dashed line at 90 %). Made by `scripts/speed_tradeoff_v2.py`. Status: **main result** for the speed question.
- **`flimngo_bias_breakdown.png`**: bar charts of the lifetime bias (median relative error, %) on the v3 test set, split by photon level, true lifetime (ns), number of decay components and IRF. Bars: retrained FLIMngo best checkpoint with the training mask and with the recall mask, last checkpoint with the training mask, pretrained FLIMngo with the recall mask, DiffFLIM v2 (6 h) with 50 and with 10 DDIM steps. Made by `scripts/analyse_flimngo_bias.py`. Status: supporting check.

### Real data (FLIMngo *C. elegans* recordings)

Both figures are made by `scripts/eval_real_celegans.py` (protocol in its docstring) from the 12 raw `.sdt` recordings in `external/FLIMngo/sample_data/celegans/raw_data/` (6 Abeta-mScarlet, 6 mScarlet; 1 s, unparalysed worms; 256 x 256 pixels, 256 bins of 0.0977 ns). There is no ground truth.

- **`real_celegans_groups_v2.png`**: Test A. One panel per method (MLE, binned MLE 5 x 5, FLIMngo pretrained, DiffFLIM v2, and FLIMngo with its own manual masks and the mean statistic of its notebook, as a reproduction). Each dot is one recording: the median lifetime (ns) inside the recall-oriented mask; red Abeta-mScarlet, blue mScarlet; boxes show median and quartiles. The panel title gives the two-sided exact Mann-Whitney p value (6 versus 6 recordings; the smallest possible value is 0.0022). Status: **main result** for the biological comparison on real data.
- **`real_celegans_examples_v2.png`**: one Abeta-mScarlet (`abeta_1s_w1`) and one mScarlet (`mscarlet_1s_w1`) recording, cropped to the mask. Columns: log10(1 + summed counts), lifetime maps of MLE, binned MLE 5 x 5, FLIMngo pretrained and DiffFLIM v2 (colour range 1 to 5 ns, white outside the mask), and the calibrated 90 % half-width of DiffFLIM v2 (0 to 1.5 ns). Status: supporting check (qualitative).

## Tables (`results/tables/`)

### Baselines

- **`baselines.csv`**: MLE and phasor on the v2 test set (40 images per level). Columns: `method`, `photons`, `n_images`, `tau_mare_pct`, `tau_bias_pct`, `int_mare_pct`, `int_bias_pct`, `tau_invalid_frac`, `n_fg_pixels` (foreground pixels), `sec_per_image` (s per 64 × 64 image). Made by the first version of `scripts/eval_baselines.py`. Status: superseded (v2 data).
- **`baselines_v3.csv`**: MLE, phasor, binned MLE 3 × 3 and 5 × 5 on the paired v3 test set (40 images per level), subsets `fg` and `bright`. Columns: the common columns plus `tau_mare_vs_tau_int_pct`. Made by `scripts/eval_baselines.py --test-set data/raw/own/sim_v3_test --tag v3`. Status: **main result** for the classical baselines (read by `eval_compare_v2.py`).

### DiffFLIM evaluations (`eval_diffflim.py`)

- **`diffflim_v1.csv`**, **`diffflim_v1_on_v3.csv`**, **`diffflim_v2.csv`**, **`diffflim_v2_3h.csv`**: DiffFLIM (mean of 10 samples, 50 DDIM steps) and the four baselines on the same images, subsets `all_fg` (DiffFLIM only), `mle_valid` and `bright`. Columns: the common columns. Data: v1 on the v2 test set (8 images per level); v1 on v3 (8 per level); v2 6 h and 3 h on v3 (40 per level). Status: v1 superseded (v2 data); v1_on_v3 supporting check; v2 and v2_3h supporting checks (50-step numbers; the main 10-step numbers are in `compare_v2.csv`).
- **`coverage_v1.csv`**, **`coverage_v1_on_v3.csv`**, **`coverage_v2.csv`**, **`coverage_v2_3h.csv`**: coverage of the **uncalibrated** intervals on the same images. Columns: `photons`, `subset` (`all_fg`, `bright`), `n_images`, `n_pixels`, `cov90_tau` and `cov90_int` (fraction of pixels whose truth lies within mean ± 1.645 std), `cov68_tau` and `cov68_int` (within mean ± 1.0 std), `median_rel_std_tau` and `median_rel_std_int` (median of std / mean). Status: superseded by the calibrated coverage tables (v1 and v1_on_v3 also by later data); kept as the uncalibrated reference.

### Calibration (`calibrate_diffflim.py`)

- **`calibration_factors_<tag>.csv`** for the tags `v1`, `v2`, `v2_newest`, `v2_extra_oldest`, `v2_extraonly`, `v2_extra`, `v2_extra_s10` (calibration set, photon estimate and steps as in the calibration figure table above). Columns: `quantity` (`tau` or `int`), `nominal_coverage`, `variant` (`global` or `per_level`), `photons` (`all` for global, else the nominal level), `photons_est_median` (median estimated photon level of the calibration images at that level, where the factor is placed on the photon axis), `factor` (the conformal quantile q of the scores |truth − mean| / std, dimensionless), `factor_curve` (value of the quadratic curve at that level, which is what test images use; not in `v1`, which used linear interpolation), `gaussian_z` (1.645 at 90 %, 0.994 at 68 %), `n_scores` (number of calibration pixels). Status: `v2_extra_s10` **main result**; `v2_extra` supporting check (50 steps); `v2_newest`, `v2_extra_oldest`, `v2_extraonly` supporting checks; `v2` and `v1` superseded. Factors of one sampler setting must not be used for another.
- **`coverage_calibrated_<tag>.csv`** for the same tags: coverage on the v3 test set (v1: v2 test set). Columns: `photons` (level or `all`), `quantity`, `nominal_coverage`, `n_images`, `cov_none`, `cov_global`, `cov_per_level` (coverage of the uncalibrated interval and of the two calibrated intervals), `cov_<variant>_ci_lo` and `_ci_hi` (95 % image-bootstrap interval, 1000 resamples; not in `v1`), `relwidth_<variant>` (mean over foreground pixels of 2 · q · std / |truth|, the relative width of the interval). Status as for the factor files. The image-level intervals are too narrow; use `coverage_cellboot_*` for intervals.
- **`coverage_cellboot_v2_s10_n10.csv`**: coverage of DiffFLIM v2 (6 h, 10 steps, 10 samples, 74 calibration cells, `otsu_adaptive`) with intervals that resample HPA cells. Columns: `quantity`, `nominal_coverage`, `photons`, `variant` (`none`, `global`, `per_level`), `coverage`, `testcell_ci_lo` and `testcell_ci_hi` (95 % interval from 5000 bootstrap resamples of the 30 test cells, factors fixed), `joint_ci_lo` and `joint_ci_hi` (95 % interval from 500 resamples of the 74 calibration and the 30 test cells with refitted factors; empty for `none`), `n_test_cells`, `n_calib_cells`. Made by `scripts/coverage_cell_bootstrap.py`. Status: **main result** for coverage.
- **`coverage_cellboot_v2.csv`**: the same for 50 DDIM steps (`--steps 50`). Status: supporting check.

### Comparison, FLIMngo and speed

- **`compare_v2.csv`**: the main comparison on the paired v3 test set (40 images per level). Columns: `method` (`FLIMngo pretrained`, `FLIMngo retrained v3`, `DiffFLIM v2 (3 h)`, `DiffFLIM v2 (6 h)`, `MLE`, `Binned MLE 5x5`), `mask` (FLIMngo only: `flimngo_recall`, the main recall-oriented mask, or `otsu_adaptive`, the precision-oriented mask the retrained FLIMngo was trained with), `ddim_steps` (DiffFLIM only: 10, the default, or 50), `photons`, `subset` (`fg`, `bright`), `n_images`, `tau_mare_pct`, `tau_bias_pct`, `int_mare_pct` and `int_bias_pct` (empty for FLIMngo), `tau_mare_inside_mask_pct` (FLIMngo only: lifetime error on true foreground pixels inside the computed mask), `mask_recall` (FLIMngo only: fraction of true foreground pixels inside the computed mask at that level). Made by `scripts/eval_compare_v2.py`. Status: **main result** (rows with `flimngo_recall` and `ddim_steps` 10); the other rows are supporting checks.
- **`flimngo_v3.csv`**: pretrained FLIMngo on the v3 test set with five preprocessing variants. Columns: `method`, `variant` (`main`: ground-truth mask, normalised, 64 × 64; `pad256`: zero-padded to 256 × 256; `otsu_mask`: Otsu mask on log counts; `no_mask`; `no_norm`: raw counts), `photons`, `subset`, `n_images`, `tau_mare_pct`, `tau_bias_pct`, `tau_mare_vs_tau_int_pct`, `tau_zero_frac` (fraction of pixels predicted as 0 ns or less), `tau_invalid_frac`, `n_pixels`, `sec_per_image_batched` (network time per image at batch 8, s), `sec_per_image_batch1_64` and `sec_per_image_batch1_256` (batch 1 at 64 × 64 and padded to 256 × 256, s). Made by `scripts/eval_flimngo.py`. Status: supporting check (sensitivity to preprocessing; the ground-truth mask is an upper bound).
- **`flimngo_mask_selection.csv`**: the 90 candidate masks for FLIMngo's input, scored on the two v3 calibration sets (1200 images). Columns: `candidate` (rule and parameters), `recall_<level>` (fraction of true foreground pixels kept), `bg_excluded_<level>` (fraction of true background pixels excluded), `min_recall` (smallest recall over the six levels), `mean_bg_excluded` (mean over levels), `p5_image_recall` (5th percentile of the per-image recall). Sorted: candidates meeting the 0.95 recall target first, then by `mean_bg_excluded`. Made by `scripts/select_flimngo_mask.py`. Status: supporting check (documents the choice of `flimngo_recall`).
- **`flimngo_bias_breakdown.csv`**: lifetime bias and error on the v3 test set, true foreground pixels, split by a factor. Columns: `method` (checkpoint and mask, pretrained FLIMngo, DiffFLIM v2 with 50 or 10 steps), `factor` (`all`, `photons`, `tau_gt_ns` true-lifetime bin, `n_components`, `irf`, `inside_otsu_adaptive_mask`), `group` (the value of the factor), `n_pixels`, `bias_pct` (median relative error, %), `mare_pct` (median absolute relative error, %), `bias_ns` (median of estimate − truth, ns), `mean_rel_pct` (mean relative error, %). Made by `scripts/analyse_flimngo_bias.py`. Status: supporting check.
- **`flimngo_bias_checks.csv`**: the mismatch checks of the bias analysis. Columns: `check` (name of the check), `training` (value on the training side or on training cells), `test` (value on the test set). Rows: the largest difference between the training and evaluation input paths (ns); percentiles of the true lifetime, shares of 1 to 4 components, IRF shares, IRF shift mean and standard deviation, and background per signal photon for the training stream and the test set; the bias on training-cell images per level and on the validation set for both checkpoints (%); and the 95 % scene-bootstrap interval of the test bias for four methods (%). Status: supporting check.
- **`speed_tradeoff_v2.csv`**: DiffFLIM v2 (6 h) with 50, 20 and 10 DDIM steps and 10 or 5 samples, on the first 10 test images per level. Columns: `steps`, `n_samples`, `photons` (level or `all`), `n_images`, `sec_per_image` (median time per image in s, GPU synchronised), `tau_mare_pct` and `int_mare_pct` (true foreground), `cov90_tau_none`, `cov90_tau_global`, `cov90_tau_per_level` (coverage of the 90 % interval before and after recalibrating this setting on 1200 calibration images), `factor90_tau_global` (the global 90 % factor of this setting), and the same four columns for intensity. Made by `scripts/speed_tradeoff_v2.py`. Status: **main result** for the speed question (only differences between settings are meaningful; the 60 images come from 10 cells).

### Real data (`eval_real_celegans.py`)

- **`real_celegans_groups_v2.csv`**: Test A. Rows with `row_type` = `image` (one per recording and method): `method`, `file`, `group` (`abeta`, `mscarlet`), `mask` (`flimngo_recall`, or `FLIMngo manual` for the reproduction rows), `n_mask_px`, `photons_median`, `photons_p10`, `photons_p90` (summed counts per pixel inside the recall mask), `frac_mask_px_below_10` (fraction of mask pixels with fewer than 10 photons), `photons_est` (photon level estimated with `otsu_adaptive`, as used for the per-level factor), `photons_est_outside_calib` (1 if outside the calibrated range 10.3 to 1020, then clamped), `bg_per_bin` (background counts per bin and pixel, estimated outside the mask in bins 7 to 18), `irf_shift_bins` (shift of the PMT IRF used by the MLE fits), `median_tau_ns` and `mean_tau_ns` (lifetime inside the mask), `tau_q90_factor` (DiffFLIM only: calibrated 90 % factor). Rows with `row_type` = `test` (one per method): `statistic`, `n_abeta`, `n_mscarlet`, `median_abeta_ns`, `median_mscarlet_ns`, `difference_ns`, `U`, `p_value` (two-sided exact Mann-Whitney). Status: **main result**.
- **`real_celegans_splithalf_v2.csv`**: Test B. Each recording split into two halves by binomial thinning (p = 0.5). Rows `image` (per recording and method) and `median over recordings` (groups `all`, `abeta`, `mscarlet`). Columns: `n_px` (mask pixels valid in both halves and the full data), `n_px_invalid` (MLE: pixels without photons in a half), `photons_est_A`, `photons_est_B`, `median_tau_A_ns`, `median_tau_B_ns`, `median_tau_full_ns`, `mad_AB_ns` (median absolute difference between the two halves' lifetime maps), `mad_AB_rel_pct` (the same relative to their mean, %), `mad_A_full_ns`, `mad_B_full_ns` (each half against the full data); DiffFLIM only: `q90_A`, `q90_B` (factors), `photons_est_outside_calib`, `median_halfwidth90_A_ns`, `overlap90_frac` (fraction of pixels whose 90 % intervals of A and B overlap), `B_in_A90_frac`, `A_in_B90_frac` (one half's mean inside the other half's interval), `full_in_A90_frac`, `full_in_B90_frac`. These fractions are not coverages (the halves share the same truth); compare them with `real_celegans_splithalf_simref_v2.csv`. Status: **main result** for consistency.
- **`real_celegans_splithalf_simref_v2.csv`**: the same split-half analysis for DiffFLIM v2 on simulated v3 test images with known truth (first 10 images at 20, 50 and 100 photons, i.e. halves of 10, 25 and 50). Columns: `photons_full`, `photons_half`, `n_images`, `mad_AB_rel_pct`, `overlap90_frac`, `B_in_A90_frac`, `coverage90_A_vs_truth` (calibrated coverage of half A against the truth), `median_halfwidth90_rel_pct` (relative 90 % half-width, %), `mare_A_vs_truth_pct` (lifetime error of half A, %). Status: supporting check (reference values for Test B).

### Model versions

- **`model_versions.csv`**: one row per DiffFLIM model whose post-training chain has run (currently only `v2`). Columns: `model` (tag), `checkpoint`, `train_minutes`, `train_steps`, `n_train_ids` (number of HPA training IDs; from the checkpoint, or the frozen list for v2), `n_test_images` (images in the cached test predictions), `tau_mare_pct_10`, `_100`, `_1000`, `_all` and `int_mare_pct_*` (median absolute relative error on the true foreground of the v3 test set, 10 DDIM steps × 10 samples, %), `tau_bias_pct_all`, `cov90_tau_per_level` and `cov90_int_per_level` (pooled coverage of the calibrated 90 % interval, per-level factors, 74 calibration cells), `real_tau_abeta_ns`, `real_tau_mscarlet_ns`, `real_mannwhitney_p` (Test A on the *C. elegans* recordings), `real_splithalf_diff_pct` (Test B, median split-half difference). Made by `scripts/model_versions.py` from the files written by `scripts/post_training_chain.sh`. Status: **main result** for comparing model versions.
