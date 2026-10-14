# Soil moisture from BIOMASS L1C stacks — team 11a, 1st BIOMASS MAAP Hackathon (Oct 2026)

Question: under which conditions do BIOMASS P-band observations carry soil-moisture information, how well, and with which
observable (backscatter, PolSAR decomposition, InSAR coherence / closure phase)?

What the code does for ONE L1C stack (7 acquisitions, 3-day repeat), fully automatic:
1. catalogue: products, primary, asset URLs (public STAC);
2. points: all ISMN stations inside the footprint, plus the user point (or the tile centre);
3. reference per point: ISMN station file (flag G), ECMWF IFS / ERA5 / ERA5-Land soil moisture at the overpass hour, rain between
   acquisitions, frost/snow, DEM slope, land cover, SMOS and Sentinel-1 listings -> reference tier 1 (station), 2 (satellite), 3 (models);
4. SAR per point: 600 m window of the complex SLC per polarisation, LUT geolocation, flattening (LUT phase screen), denoising
   (LUT noise maps), sigma0, 100 m looks, coherence matrix, closure phases, Freeman-Durden, H/alpha;
5. estimators: sigma0 sensitivity (dB per m3/m3), coherence vs |dSM| controlled for time, De Zan closure-phase forward model
   (Mironov dielectric model) and an experimental closure inversion;
6. outputs: figures, `metrics.csv`, `observables_long.csv` (shared-table format), `closure_phases.csv`, `sar_info.json`.

## Layout
    run_stack.py                 driver, one stack (all stations inside or one point); --no-sar = reference part only (no token)
    run_batch.py                 runs a list of stacks (batch_stacks.csv) and aggregates all runs -> all_metrics.csv, all_tiles_summary.csv
    select_tile.ipynb            interactive: map of all L1C tiles -> point or stack key -> reference preview -> SAR run -> results
    screen_all_tiles.py          IFS screening of all tile footprints (rain events, SM range, frost, snow) -> tile_screening.csv + map
    build_tile_index.py          rebuilds l1c_stack_index.csv / l1c_items_index.parquet from the MAAP catalogue (~15 min)
    sm_pipeline/catalog.py       stack -> products + asset URLs; MAAP token exchange (offline -> access token)
    sm_pipeline/io_l1c.py        windowed COG reads -> complex SLC; LUT access (geometry, radiometry, denoising, flattening, coherence)
    sm_pipeline/observables.py   multilook, coherence matrix, closure phases, C3, H/alpha, Freeman-Durden (non-negative powers)
    sm_pipeline/estimators.py    Mironov / Hallikainen dielectric models, De Zan closure model, experimental inversion
    sm_pipeline/reference.py     Open-Meteo (IFS, ERA5, ERA5-Land) at the overpass hour, ISMN export reader
    sm_pipeline/auxiliary.py     stations in footprint, SMOS / Sentinel-1 listings, WorldCover + Copernicus DEM
    sm_pipeline/evaluate.py      slopes with t-CI, Pearson/Spearman, correlation of first differences, pair regression, closure check
    sm_pipeline/plots.py         all figures (Okabe-Ito palette, safe for red-green colour-vision deficiency)
    sm_pipeline/interactive.py   TilePicker widget (folium map; ipyleaflet optional)
    tests/test_closure_synthetic.py   De Zan forward model -> noise -> inversion (no data needed)
    tests/test_end_to_end_mock.py     full pipeline on a simulated stack with known sensitivities (no data needed)
    tests/replot_from_npz.py          rebuild all stage-6 outputs of an existing run from slc_window.npz (no token needed)
    criteria.json                thresholds for the candidate-window search (editable)
    INTERPRETATION_GUIDE.md      how to read every figure, table column and log line
    l1c_stack_index.csv, l1c_items_index.parquet   catalogue index (13 Oct 2026: 24 904 stacks with >= 6 acquisitions, 3 903 tiles)

## Setup on MAAP
    pip install --user rasterio netCDF4 folium            # --user survives a server restart, /opt/conda does not
    cp sm_pipeline_draft ~/sm_pipeline_draft -r            # run from the home folder: it is a local disk, my-private-bucket is S3 (slow)
    # offline token (90 days) from portal.maap.eo.esa.int -> services -> auth -> token -> one line in ~/token.txt
    # ISMN exports: ~/shared-buckets/<id>/ISMN/<network>+<station>.csv (+ ismn_locations.geojson); set the paths in the first notebook cell
Outputs go to `~/data/runs/<stack>/` (option `--out`). The home folder is not shared and may not survive a server rebuild:
at the end of the day `rsync -a ~/data/runs/ ~/my-private-bucket/data/runs/`.

## Run
    # one stack, all stations inside the footprint (+ tile centre if no station)
    python run_stack.py --stack S2_T009_F118_G01_M04 --ismn <ISMN>/ismn_locations.geojson --ismn-dir <ISMN>
    # one stack, one point only
    python run_stack.py --stack S1_T009_F118_G01_M04 --lat 42.25 --lon 70.30 --name Interior --only-point
    # no station: rate all 600 m blocks of the footprint by land cover and slope, run the 3 best (>= 5 km apart)
    python run_stack.py --stack S3_T018_F137_G01_M03 --auto-points 3          # --candidates = only rate and map, add nothing
    # many stacks without interaction (skips stacks that already have metrics.csv), then the summary tables
    nohup python run_batch.py batch_stacks.csv > batch.log 2>&1 &
    python run_batch.py --aggregate-only
    # re-draw the figures of a finished run after a code update (no token)
    python tests/replot_from_npz.py ~/data/runs/S2_T009_F118_G01_M04
Main options: `--half-m 300` (window half width), `--look-m 100`, `--flatten auto|on|off`, `--no-denoise`, `--min-valid 0.8`,
`--min-sm-range 0.05`, `--dielectric mironov|hallikainen --clay 0.2 --sand 0.4`, `--no-quicklooks`, `--lut-check`.

## Interactive use
`select_tile.ipynb`: one panel. Map with ALL 3 903 L1C tiles as one vector layer (colour = screening score once `tile_screening.csv`
exists, black outline = ISMN station inside, blue = stacks found at your point; tooltip with stacks, period, numbers); *Map options*
can thin it to station/screening tiles; WorldCover layer switchable. Click → paste the coordinate (or a stack
key) → Find → choose the stack → Run (*reference only* = no token). Extras folded: candidate windows in the tile, reference preview.
Results as tabs: main figure, coherence, quicklooks, key numbers (with the window facts from `sar_info.json`), check figures.
`picker.show_results('~/data/runs/<stack>')` re-displays an earlier run.

## Choosing the window when there is no station (transparent, nothing is picked silently)
`auxiliary.candidate_windows` rates every 600 m block of the footprint: WorldCover 2021 class fractions (10 m, sampled every 50 m)
and Copernicus DEM slope (30 m, sampled every 90 m). A block is `suitable` if grassland + cropland >= 90 %, tree <= 5 %, built <= 1 %,
water <= 1 %, mean slope <= 5 deg and <= 20 % of the block steeper than 5 deg. All thresholds, the ranking columns, the number of
stars and their minimum separation are in **`criteria.json`** (plain text, edit the numbers; `--criteria other.json` for a
different file). Suitable blocks are ranked by the `rank_by` columns; `pick_spread` takes the best `n_best` at least
`min_separation_km` apart. Hawaii F138: 559 of 21 514 blocks pass with slope <= 5 deg, 2 070 with slope <= 15 deg. Everything is written out: `candidate_windows.csv`
(all blocks with all numbers), `candidate_windows.png` (land-cover map, slope map, suitable blocks, chosen stars), the criteria and the
chosen table in the log. In the notebook: step 1b "Find candidate windows" shows the same on the map (WorldCover layer switchable) and
lets you adopt one of the stars as the point, or ignore them. Whether a window lies in the valid swath is only known at SAR time
(skipped with a message if not).

## Reference tiers (decided per point, automatically)
1. ISMN station inside the footprint: ground truth; sensitivities and RMSE are computed against the shallowest depth.
2. Satellite soil moisture (SMOS L2 listed; value extraction TODO): consistency check at 25-40 km scale.
3. Models only (IFS 9 km, ERA5 31 km, ERA5-Land 9 km): plausibility check of timing and sign, never a calibration target.

## Known limitations (14 Oct 2026)
- Denoising: in several stacks the LUT noise maps of the secondaries are 8-10 dB above those of the primary and above the measured
  cross-pol power (NSR > 0 dB); denoising is then skipped for the whole stack and the NSR table is written to `sar_info.json`.
  To be clarified with the BPS team (units or scaling of `denoisingXX` in the secondary LUTs).
- Only L1C STA stacks are supported; L1B DGM products (no phase) stop with a message.
- SMOS L2 values, SMAP and IMERG are not read (listing only / Earthdata login).
- Pixel selection inside the window is by distance; a selection by slope / land cover is prepared (LUT geometry in `window_geometry.png`).
- Open-Meteo free quota: 10 000 location-requests per day and IP; a pipeline run needs 3-4 per point, the tile screening ~40.
