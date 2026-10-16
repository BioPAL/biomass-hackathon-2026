# Pipeline notes

## Data flow

    GEDI table (step 01 or your own) + TOM product -> 02 -> data/interim/shots.parquet, tom_profiles.npz
    DTM product                                    -> 03 -> adds slope_deg, slope_ok to shots.parquet
                                                      04 -> cells.parquet, cell_profiles.npz, results/cells_500m.csv
                                                      05 -> results/corr_GEDI_vs_TOM_{pearson,spearman}.csv
                                                      06 -> results/ridge_cv_{cell,shot}.csv, predictions_{cell,shot}.parquet
                                                      07 -> results/filter_experiment/
                                                      08 -> results/figures/obs_vs_pred_<target>_<level>.png

`shots.parquet` and `tom_profiles.npz` stay row-aligned: step 02 writes both, later steps only add columns.

## Comparing numbers

- Cell-level CV block: taken from the cell centroid. Using the rounded mean of the shots' block ids moves a few edge cells
  to other folds and changes r by about 0.002 (0.006 for the best filter config).
- Slope mask on or off: without it (49,787 shots, 4,401 cells) rh_98 at cell level is r 0.38 instead of 0.45.
- `ridge_cv_*.csv` has one row per target; a mean over the 18 targets is not comparable with a single-target number like rh_98.

## Notebook (`BIOMASS_TOM_Lope_exploration.ipynb`) to module

| Notebook | Module |
|---|---|
| 1-4: setup, XML axes, lazy cube, `read`, `db` | `tom.py` |
| 12: GEDI table, bbox crop, MAX_DAYS | `gedi.load_gedi_table` |
| 12: block-wise profile extraction, `Z_GROUND` | `tom.extract_profiles`, `tom.ground_proxy` (step 02) |
| 12b: slope mask | `terrain.DTMSlope` (step 03) |
| 12c: cells | `cells.py`, `Dataset.cell_table` (step 04) |
| 12d: targets, ridge with spatial CV | `features.build_targets`, `modeling.py` (step 06) |
| correlation tables | `features.radar_cell_features` (step 05) |
| `GEDI_data_access_gedidb_filtered.ipynb` | `gedi.download` (step 01) |
| `curtain3d.py` | `scripts/curtain3d.py` |

Not ported: the cube plots (5-11), the along-track curtain and band-correlation explorations (12e onwards) and the
DTM-vs-GEDI-ground checks. They stay in the notebook.
