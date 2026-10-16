# Soil moisture from BIOMASS L1C stacks — team 11a, 1st BIOMASS MAAP Hackathon (Oct 2026)

Question: under which conditions do BIOMASS P-band observations carry soil-moisture information, how well, and with which
observable (backscatter, PolSAR decomposition, InSAR coherence / closure phase)?

What the code does for ONE L1C stack (7 acquisitions, 3-day repeat), fully automatic:
1. catalogue: products, primary, asset URLs (public STAC);
2. points: all ISMN stations inside the footprint, plus the point chosen in the dashboard (`--lat/--lon`; else the stack centre);
3. reference per point: ISMN station file (flag G), hourly ERA5-Land / ERA5 / ECMWF IFS soil moisture (ERA5-Land is the default
   model reference; values at the overpass hour for the metrics), rain between acquisitions, frost/snow, DEM slope, land cover, soil texture / organic carbon / bulk density (ISRIC SoilGrids 2.0, 0-5 and
   5-15 cm; also the clay/sand input of the dielectric model), Sentinel-2 L2A NDVI / NDMI over the analysis window (30 days
   before to 30 days after the stack; method of the team notebook 'Biomass L1C Station Ancillary'), SMOS listing -> reference
   tier 1 (station) or 3 (models);
4. SAR per point: square analysis window (default 600 m, 200-2 000 m selectable in the dashboard, `--half-m`) of the complex SLC
   per polarisation, LUT geolocation, flattening (LUT phase screen), denoising
   (LUT noise maps), sigma0, 100 m looks, coherence matrix, closure phases and the polarimetric decompositions of
   PyPolSARpro (Freeman-Durden, Yamaguchi-4 y4r, van Zyl 1992, H/A/alpha; without PyPolSARpro only an own H/alpha);
5. estimators: sigma0 sensitivity (dB per m3/m3), coherence vs |dSM| controlled for time, De Zan closure-phase forward model
   (Mironov dielectric model) and an experimental closure inversion;
6. outputs: figures, `metrics.csv`, `observables_long.csv` (shared-table format), `closure_phases.csv`, `sar_info.json`,
   `s2_<point>_<window>m.csv` (Sentinel-2 scenes).

Global approach (this package): `select_tile.ipynb` shows ONE dashboard of all 3 903 L1C tiles (24 904 stacks), coloured and
filtered by land cover and slope as information. Click a tile, then a stack: public BPS quicklooks; **compute full report** runs
`run_stack.py` for that stack and opens `~/data/runs/<stack>/report.html` with all figures and numbers. Station case studies are
done separately by the team.

## Layout
    run_stack.py                 driver, one stack (all stations inside or one point); --no-sar = reference part only (no token)
    run_batch.py                 runs a list of stacks (CSV: stack, lat, lon, name, extra) and aggregates all runs -> all_metrics.csv,
                                 all_tiles_summary.csv (used by the dashboard for computed stacks)
    select_tile.ipynb            the dashboard app: all tiles -> click tile -> stack -> compute / open full report
    build_dashboard.py           index + land cover + computed runs -> one static dashboard.html (also usable without the notebook)
    stack_report.py              full report of one computed stack -> <run>/report.html (all figures embedded)
    tile_landcover.py            WorldCover 2021 fractions and Copernicus DEM slope per tile -> tile_landcover.csv (shipped, ~1 h)
    build_tile_index.py          rebuilds l1c_stack_index.csv / l1c_items_index.parquet from the MAAP catalogue (~15 min)
    sm_pipeline/catalog.py       stack -> products + asset URLs; MAAP token exchange (offline -> access token)
    sm_pipeline/io_l1c.py        windowed COG reads -> complex SLC; LUT access (geometry, radiometry, denoising, flattening, coherence)
    sm_pipeline/observables.py   multilook, coherence matrix, closure phases, PyPolSARpro wrapper, own H/alpha (fallback)
    sm_pipeline/estimators.py    Mironov / Hallikainen dielectric models, De Zan closure model, experimental inversion
    sm_pipeline/reference.py     Open-Meteo (IFS, ERA5, ERA5-Land) at the overpass hour, ISMN export reader
    sm_pipeline/deps.py          checks the required packages and pip-installs missing ones (called by every entry point)
    sm_pipeline/auxiliary.py     stations in footprint, SMOS listing, SoilGrids, WorldCover + Copernicus DEM, Sentinel-2 NDVI / NDMI
                                 (Earth Search on AWS; Sentinel-1 RTC reader kept, not used)
    sm_pipeline/evaluate.py      slopes with t-CI, Pearson/Spearman, correlation of first differences, pair regression, closure check
    sm_pipeline/plots.py         all figures (Okabe-Ito palette, safe for red-green colour-vision deficiency)
    sm_pipeline/app.py           notebook app: dashboard in an iframe, runs run_stack.py + stack_report.py on request
    vendor/                      Leaflet 1.9.4 (BSD-2) and Natural Earth 110 m land (public domain, via world-atlas) for offline use
    tests/test_closure_synthetic.py   De Zan forward model -> noise -> inversion (no data needed)
    tests/test_end_to_end_mock.py     full pipeline on a simulated stack with known sensitivities (no data needed)
    tests/replot_from_npz.py          rebuild all stage-6 outputs of an existing run from slc_window.npz (no token needed)
    INTERPRETATION_GUIDE.md      how to read every figure, table column and log line
    l1c_stack_index.csv, l1c_items_index.parquet   catalogue index (13 Oct 2026: 24 904 stacks with >= 6 acquisitions, 3 903 tiles)

## Setup on MAAP
    # run from the home folder (local disk; my-private-bucket is S3 and slow)
    unzip -q ~/sm_pipeline_draft.zip -d ~                 # or from the repository: cp -r topics/11-A-SOIL-MOISTURE/sm_pipeline ~/sm_pipeline_draft
    cd ~/sm_pipeline_draft
    pip install --user -r requirements.txt                # --user survives a server restart, /opt/conda does not
    # (optional: run_stack.py, run_batch.py, build_dashboard.py, stack_report.py and select_tile.ipynb check the packages
    #  themselves and pip-install missing ones, incl. PyPolSARpro on Python >= 3.12; SM_PIPELINE_NO_INSTALL=1 switches this off)
    # PyPolSARpro (Apache-2.0, satim-co/PolSARpro): all polarimetric decompositions (without it: own H/alpha only); needs Python >= 3.12,
    # dask and xarray; check: python -c "import dask, xarray, polsarpro.decompositions"
    pip install --user --no-deps "git+https://github.com/satim-co/PolSARpro@bea8352fec10496f1a78d9b9992154006d8e0514"
    # offline token (90 days) from portal.maap.eo.esa.int -> services -> auth -> token -> one line in ~/token.txt
    # ISMN exports: ~/shared-buckets/<id>/ISMN/<network>+<station>.csv (+ ismn_locations.geojson); set the paths in the first notebook cell
Outputs go to `~/data/runs/<stack>/` (option `--out`). The home folder is not shared and may not survive a server rebuild:
at the end of the day `mkdir -p ~/my-private-bucket/data/runs && cp -ru ~/data/runs/. ~/my-private-bucket/data/runs/`
(MAAP has no rsync).

## Run
    # one stack, all stations inside the footprint (the stack centre if there is no station)
    python run_stack.py --stack S2_T009_F118_G01_M04 --ismn <ISMN>/ismn_locations.geojson --ismn-dir <ISMN>
    # one stack, one point only
    python run_stack.py --stack S1_T009_F118_G01_M04 --lat 42.25 --lon 70.30 --name Interior --only-point
    # many stacks without interaction (skips stacks that already have metrics.csv), then the summary tables
    nohup python run_batch.py my_stacks.csv > batch.log 2>&1 &
    python run_batch.py --aggregate-only
    # re-draw the figures of a finished run after a code update (no token)
    python tests/replot_from_npz.py ~/data/runs/S2_T009_F118_G01_M04
Main options: `--half-m 300` (window half width), `--look-m 100`, `--flatten auto|on|off`, `--no-denoise`, `--min-valid 0.8`,
`--min-sm-range 0.05`, `--dielectric mironov|hallikainen` (`--clay/--sand` override the SoilGrids texture; without SoilGrids
data 0.2 / 0.4), `--no-quicklooks`, `--lut-check`, `--no-s2` (no Sentinel-2), `--image-look-m 50` (display cells of
`window_images.png`; the analysis uses `--look-m`).

## Interactive use (the dashboard)
Open `select_tile.ipynb`, run the cell. The first start builds `dashboard.html` (about 30 s). Left: colour (grassland + cropland,
tree cover, mean slope, number of stacks), filters (land cover, slope, ISMN, computed points, months of the stack window, warm
season by hemisphere as a proxy against frozen soil and snow), preset *open land, gentle terrain*, export of the filtered stacks
as CSV for `run_batch.py`. Map: tiles; computed points as dots: yellow = with in-situ station data (ISMN), orange = without
in-situ data (model reference only), grey = SAR skipped (tooltip with the key numbers, click opens their stack). Right, after a
click: land-cover bar and slope of the tile, its stacks; for a stack the seven public quicklooks, its footprint as a dashed box and **compute full report** / **open full report**. A click
inside the dashed box sets your own point (switch to *Esri satellite* to aim); without a point the run uses the ISMN stations inside the stack, or the stack centre if
there is none. The
**analysis window** (200 m to 2 000 m square, default 600 m = `--half-m 300`) is chosen in the stack panel; inside it 100 m cells
are averaged. ISMN stations inside the stack are always added. The report shows the points of the latest run of that
stack (a new point replaces the previous user point in the report). With a point set, **compute all N stacks at this point**
runs every stack whose footprint contains the point and that passes the filters (all tracks, swaths and major cycles = all dates
there), one after the other, each with its own report. The run log and the report appear below the map. The report starts with
the main figure of every point, each with a small table of the site facts above it (elevation, incidence, slopes, land cover,
soil, window, processing), then key numbers, stack facts and the other figures. Land cover and slope are tile averages (bounding box of the tile's stacks) and serve as information, not as a selection rule.
`dashboard.html` alone (downloaded, opened in a browser) shows everything except that the button prints the command to run.

## Reference tiers (decided per point, automatically)
1. ISMN station inside the footprint: ground truth; sensitivities and RMSE are computed against the shallowest depth.
2. Satellite soil moisture: not assigned in this version (SMOS L2 files are listed, their values are not read yet).
3. Models only (ERA5-Land 9 km by default, else ERA5 31 km or IFS 9 km): plausibility check of timing and sign, never a
   calibration target.

## Known limitations (15 Oct 2026)
- The closure-phase inversion is experimental: on the synthetic dry-down it recovers dSM with 0.012 m3/m3 RMSE without noise but
  0.083 m3/m3 with 0.05 rad closure noise (`tests/test_closure_synthetic.py` asserts the noise-free case and reports the noisy
  one); its output is reported but not used.
- Model references come from the free Open-Meteo API (10 000 calls/day per IP). A run needs about 3 calls per point and re-uses
  the model series already in the run folder; after HTTP 429 the run continues without models (station runs are unaffected).
- Denoising: in several stacks the LUT noise maps of the secondaries are 8-10 dB above those of the primary and above the measured
  cross-pol power (NSR > 0 dB); denoising is then skipped for the whole stack and the NSR table is written to `sar_info.json`.
  To be clarified with the BPS team (units or scaling of `denoisingXX` in the secondary LUTs).
- Only L1C STA stacks are supported; L1B DGM products (no phase) stop with a message.
- SMOS L2 values, SMAP and IMERG are not read (listing only / Earthdata login). Sentinel-1 was dropped on 15 Oct 2026 (C-band, orbit
  and incidence effects as large as the moisture signal; too hard to interpret quickly).
- Decomposition shares depend strongly on the scattering model (SilverSword, first date: surface share 37 % Freeman-Durden,
  37 % Yamaguchi-4 y4r, 65 % van Zyl 1992, all PyPolSARpro); only their change over the 7 dates is interpreted. The Dubois soil-moisture inversion in
  PyPolSARpro is not used: it was derived for 1.5-11 GHz and bare soil (BIOMASS: 0.435 GHz).
- Soil properties are SoilGrids 2.0 values at 250 m (the ISMN bucket on MAAP has no static station metadata). The ISRIC service is
  sometimes slow (5-60 s per point): one retry, then the run continues without soil data; `replot_from_npz.py` tries again later.
- Sentinel-2 needs internet access to earth-search.aws.element84.com and its S3 bucket (public). Cloudy scenes are masked with the
  scene classification; in cloudy regions few clear dates remain (Hawaii, SilverSword: 10 clear of 14 dates in 78 days).
- Pixel selection inside the window is by distance; a selection by slope / land cover is prepared (LUT geometry in `window_geometry.png`).
