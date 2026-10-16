# sm_pipeline: soil-moisture information in BIOMASS L1C stacks

Team 11a, 1st ESA BIOMASS MAAP Hackathon (October 2026). For one BIOMASS L1C STA stack (7 acquisitions, 3-day repeat) the
pipeline compares backscatter, polarimetric decompositions, InSAR coherence and closure phase with soil moisture at the
ISMN stations inside the footprint or at a chosen point. A dashboard shows all 24 904 stacks (3 903 tiles).

## Quick start (MAAP)
1. `cp -r ~/biomass-hackathon-2026/topics/11-A-SOIL-MOISTURE/sm_pipeline ~/sm_pipeline && cd ~/sm_pipeline`
2. Save your own MAAP offline token as one line in `~/token.txt` (portal.maap.eo.esa.int -> services -> auth -> token).
   Without a token everything except reading the SAR data runs.
3. Open `select_tile.ipynb` and run the cell: click a tile, then a stack, then **compute full report**.

Missing packages are installed on first use (or up front: `pip install --user -r requirements.txt`; PyPolSARpro needs
Python >= 3.12). Outputs go to `~/data/runs/<stack>/` (figures, `metrics.csv`, `report.html`).

    python run_stack.py --stack S2_T009_F118_G01_M04                        # all ISMN stations in the footprint
    python run_stack.py --stack <stack> --lat <lat> --lon <lon> --only-point
    python run_batch.py my_stacks.csv                                       # several stacks, then summary tables

## What one run does
1. Products of the stack (public MAAP STAC) and points: ISMN stations in the footprint, a chosen point, else the stack centre.
2. Reference per point: ISMN station (flag G), ERA5-Land / ERA5 / ECMWF IFS, rain, frost, SoilGrids, WorldCover, slope,
   Sentinel-2 NDVI / NDMI.
3. SAR: analysis window (default 600 m) of the SLCs, LUT geolocation, flattening, denoising, sigma0, 100 m looks,
   coherence, closure phases, PyPolSARpro decompositions.
4. Metrics: sigma0 sensitivity (dB per m3/m3), correlation of changes, coherence vs |dSM|, closure-phase model.
5. Figures and one report per stack.

## Layout
| Path | Content |
|---|---|
| `select_tile.ipynb` | dashboard (entry point) |
| `run_stack.py`, `run_batch.py`, `stack_report.py`, `build_dashboard.py` | one stack, many stacks, report, static dashboard |
| `sm_pipeline/` | package: catalogue and token, L1C reading, observables, estimators, references, plots, dashboard app |
| `tools/` | `overview_figure.py` (one point over one or more stacks), `find_overlap_point.py` (a point inside two stacks) |
| `build_tile_index.py`, `tile_landcover.py` | rebuild the shipped stack index and the tile land cover |
| `l1c_items_index.parquet`, `l1c_stack_index.csv`, `tile_landcover.csv` | shipped index of all L1C stacks (13 Oct 2026), tile land cover |
| `tests/` | synthetic tests (no data needed); `replot_from_npz.py` redraws a finished run |
| `docs/` | `USAGE.md` (all options, dashboard, details), `INTERPRETATION_GUIDE.md` (how to read the outputs), `data_sources_overview.xlsx` |
| `vendor/` | Leaflet 1.9.4 (BSD-2), Natural Earth land outline (public domain) |

## Data
- BIOMASS L1C from the MAAP catalogue. The SAR part needs a personal MAAP token; never commit it (`token.txt` is git-ignored).
- ISMN station data are not in the repository (ISMN terms: no onward distribution). Default source: the team copy in the
  MAAP shared bucket. Your own download: convert it with `../legacy/ismn_parsing.ipynb`, then set `SM_PIPELINE_ISMN_DIR`
  (folder with `<network>+<station>.csv`) and, if needed, `SM_PIPELINE_ISMN_GEOJSON`. Without ISMN data the runs use the
  model references.
- Models via Open-Meteo (free, 10 000 calls per day and IP); Sentinel-2 via Earth Search (AWS).

## Main limitations
- A point usually lies in one stack only (7 acquisitions in 18 days); consecutive stacks come from neighbouring swaths.
- Decomposition shares depend on the scattering model; only their change over the 7 dates is interpreted.
- The closure-phase inversion is experimental and not used for results.
- Denoising is skipped where the LUT noise maps exceed the measured cross-pol power (several stacks; open question to the BPS team).

Full list and all options: `docs/USAGE.md`.
