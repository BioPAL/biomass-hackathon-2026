# biomass-tom-gedi

Relating BIOMASS P-band tomographic profiles (TOM L2A) to GEDI canopy-structure metrics (RH98, cover, PAI, PAVD, AGBD)
over Lope, Gabon. The code started as an exploration notebook and is split here into small steps that can be run one at a time.

Pipeline (step numbers in brackets):

    GEDI shots (01) -> TOM profile under each shot (02) -> slope mask (03) -> 500 m cells (04) -> correlations (05)
    ridge with spatial-block CV (06) -> filter experiment (07) -> plots (08)

| Step | What it does | Writes (under `data/interim` or `results`) |
|---|---|---|
| 01 | download GEDI L2A/L2B/L4A shots from gediDB with the MAAP quality filters (optional, needs network) | `gedi_download.parquet` |
| 02 | read the TOM cube block by block, mean 3x3-pixel profile under every shot | `shots.parquet`, `tom_profiles.npz` |
| 03 | slope from the BIOMASS DTM, keep slope < 5 deg | `slope_deg`, `slope_ok` columns |
| 04 | average the kept shots and their profiles into ~500 m cells | `cells.parquet`, `cell_profiles.npz` |
| 05 | Pearson/Spearman correlations, GEDI metrics vs radar features | `corr_*.csv`, heatmaps |
| 06 | ridge from the whole profile to 18 GEDI metrics, spatial CV, cell and shot level | `ridge_cv_*.csv`, `predictions_*.parquet` |
| 07 | drop the brightest TomoSAR profiles, repeat the CV, compare with random removal | `filter_experiment/` |
| 08 | observed vs predicted density plot | `figures/obs_vs_pred_*.png` |

`scripts/curtain3d.py` is an optional 3-D curtain of the TOM profile along one GEDI track (needs a GEDI L1B granule).

## Setup

    pip install -e .              # core dependencies
    pip install -e ".[gedidb]"    # only for step 01
    pip install -e ".[dev]"       # pytest

Data is not included. You need the BIOMASS TOM L2A and DTM L2A product folders for the Lope scene, and a GEDI shot table
(from step 01, or your own parquet/csv with the usual L2A/L2B/L4A columns: `latitude`, `longitude`, `rh_*`, `cover`, `agbd`,
`pavd_z_*`, `pai_z_*`, `cover_z_*`, `digital_elevation_model`, ...).

## Running

    cp configs/lope.example.yaml configs/local.yaml    # local.yaml is git-ignored, put your paths in it
    tomgedi run all --config configs/local.yaml        # steps 02-08
    tomgedi run 06 --config configs/local.yaml         # a single step
    pytest                                             # tests use synthetic data

Run from the repository root. Slope limit, cell size, minimum shots per cell, CV block size, feature window and the
filter grid are all in the config.

## Method notes

- The coordinate variables in the cube are empty, so lat/lon/height axes are rebuilt from the annotation XML (row 0 = south).
- Profile under a shot: mean linear power in a 3x3 window (~140 m, matching the 150 m averaging window of the tomography).
- Slope: DTM smoothed with a Gaussian (sigma 3 px), then the gradient. The DTM GeoTIFF has a malformed geotransform, so the
  axes are rebuilt by hand and the N-S orientation is checked against GEDI's own DEM (r 0.99 vs 0.48 flipped). Only slope is
  masked, not true layover/shadow. About 61 % of the shots are kept.
- Cells: `cell = (row // 11) * 100000 + col // 11` (11 pixels of 46 m, ~500 m); cells with fewer than 5 shots are dropped.
- Model: dB profile from -20 to 90 m (2 m step, HH+HV+VV, 168 features), standardised, `RidgeCV` per target, 5-fold
  `GroupKFold` over 0.1 deg blocks (at cell level the block of the cell centroid). r, R2 and RMSE are pooled over held-out predictions.
- Targets (18): rh_25/50/75/90/98/100, cover, agbd, fhd_normal, pai, PAI above 10 and 30 m, cover above 30 m, PAVD layers.
- Filters: per polarization `floor_db` (10th percentile of the profile), `top_db` (mean dB above 60 m) or `mean_db`; the
  upper 1, 2.5, 5 or 10 % of the slope-kept shots are removed (HV only or any polarization) and the cells are rebuilt.

## Checked against the original notebook results

Run on the Lope scene (steps 02-06): 49,787 shots with a valid profile (identical to the original profile file), 30,279 kept
by the slope mask (same mask), 2,525 cells (same ids and shot counts), identical Pearson correlation table. Ridge CV for
rh_98 at cell level gives r 0.449, R2 0.201, RMSE 4.32 m, as in the original run. AGBD: r 0.491, R2 0.241, RMSE 65 Mg/ha per
cell; r 0.300, R2 0.090, RMSE 135 Mg/ha per shot. The notebook's first-pass table (`TOM_vs_GEDI_metrics_cv.csv`) agrees to
within about 0.01 in r; the remaining difference was not traced. The shot-level filter results are identical to the original
run; the cell-level ones depend on how cells are assigned to folds (see below).

## Caveats

- The vertical axis of the TOM cube is not confirmed to be height above ground. `Z_GROUND` (~4 m, HH peak of the mean
  profile) is only a scene-wide proxy and is used just by the `ground_relative` features and the correlation features.
- GEDI `agbd` (L4A) is a model estimate, not field biomass.
- The best filter setting is picked on the same data it is scored on (24 candidates), so its gain is optimistic. At cell
  level the gain moved by up to 0.011 in R2 when only the fold assignment changed, and random removal of the same number of
  shots gave +0.004 to +0.014, so cell-level gains below ~0.01 are within noise.
- Only a slope mask is implemented, not a layover/shadow mask (that needs incidence angle and look direction).

## Layout

    configs/lope.example.yaml      settings
    src/biomass_tom_gedi/          tom, gedi, terrain, cells, features, modeling, filters, experiment, dataset, plotting
    src/biomass_tom_gedi/steps/    s01 ... s08, one module per step
    scripts/curtain3d.py           optional 3-D curtain
    tests/                         pytest, synthetic data
    docs/pipeline.md               data flow and notebook-to-module map

## Author

Abhilash Dutta Roy, University of Maryland (duttaroy@umd.edu)

Data and tools: ESA BIOMASS (TOM L2A and DTM L2A products, BIOMASS Coding workshop data); NASA GEDI (L2A, L2B, L4A),
accessed through gediDB (Besnard, Dombrowski & Holcomb, https://doi.org/10.5281/zenodo.13885229).

No licence chosen yet; add a `LICENSE` file before publishing.
