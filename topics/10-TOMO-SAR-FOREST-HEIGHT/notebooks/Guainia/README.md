# Guainía: BIOMASS P-band tomography, forest height and polarimetry

Five notebooks that take a BIOMASS Level-1c tomographic stack over Guainía (Colombia) from the ESA MAAP catalogue through to forest-height maps and polarimetric analysis. Each notebook is driven by one configuration cell at the top. They pass results to each other through `Guainia_outputs/run_manifest.json`.

| | |
|---|---|
| Stack | `S1_M03_T029_F152_v02`: swath S1, major cycle 3, track 29, frame 152, product version 02 |
| Acquisitions | 7 repeat cycles (C01–C07), 2026-04-24 → 2026-05-12, 3-day revisit, descending; primary C04 (2026-05-03) |
| Product | `BIO_S1_STA__1S`: quad-pol (HH HV VH VV) single-look complex, β⁰, coregistered on the primary |
| Frame | 20 765 azimuth × 1 315 range samples (6.7 m × 19.8 m slant ≈ 47 m ground); ≈ 139 km × 60 km; 2.78–4.14° N, 70.34–69.52° W |
| Baselines | $k_z$ from −0.155 to +0.165 rad/m → vertical resolution ≈ 19.6 m, height of ambiguity ≈ 129 m |
| Reference | `Forest_Height_Guainia.tif`: 30 m canopy-height map (EPSG:4326, uint8 m; 0 = no trees, 101 = water, 102 = snow/ice, 103 = no data); not included in the GitHub repository, place it in this folder |

## Workflow

```
00_Data_Access ──► stack_download/            MAAP search, stack maps/timeline/quick-looks, download
01_Stack_Build ──► Guainia_outputs/<site>_<stack>_stack.nc   all acquisitions + LUT layers on one grid, baseline check
02_Tomography  ──► Guainia_outputs/tomography/         LUT phase calibration, coherence, Capon/beamformer tomograms
03_Forest_Height ► Guainia_outputs/forest_height/      6 height estimators, assessment vs reference, bias correction, GeoTIFF
04_Polarimetric_Analysis ► Guainia_outputs/polarimetry/  σ⁰ + noise removal, Pauli/H-A-α/Freeman/Wishart, PolTomo, Pauli GeoTIFFs
```

| notebook | what it does | run time* |
|---|---|---|
| `00_Data_Access` | Searches the MAAP STAC catalogue around an AOI and groups products into stacks (swath/major cycle/track/frame/version). Shows footprints on OpenStreetMap, an acquisition timeline and catalogue quick-looks. Completes the chosen stack and downloads it, either as whole zipped products (`zip`) or as only the files needed (`assets`), skipping anything already on disk. Finally checks that notebook 1 can read the download. | ~1 min (search) |
| `01_Stack_Build` | Lists the products and identifies the primary. Inspects one product (annotation grid, LUT layers and axes). Writes every acquisition and LUT layer to one netCDF on the SLC grid. Then verifies it: layout, no-data, LUT plots, $k_z$ and baseline requirement, quick-looks. Writes the manifest. | ~2 min |
| `02_Tomography` | Shows what the flattening and SKP calibration screens do to an interferogram. Finds the LUT phase signs and the $k_z$ sign from the data. Computes multibaseline coherence for HH, VV and X = (HV+VH)/2, then beamformer and Capon tomograms, with profiles, transects and horizontal slices. | ~4.5 min, 14 GB RAM |
| `03_Forest_Height` | Puts the reference on the radar grid. Applies a quality mask. Computes six estimators (energy envelope, percentile envelope, min–max edge, canopy top, phase centre, RVoG) and assesses them against the reference. Bias-corrects the best estimator with a spatial train/test split. Writes geographic maps and a multi-band GeoTIFF. | ~2.5 min |
| `04_Polarimetric_Analysis` | **Part A** (reference date): σ⁰ calibration and noise removal with a check of the noise LUT, backscatter, Pauli RGB and **Pauli GeoTIFF**, H/A/α and the H–α plane, Freeman–Durden, Wishart H/A/α classes, backscatter over the 7 dates. **Part B**: polarimetric tomography (coherent focusing of S(z), then Pauli/H-A-α/Freeman per height) with profiles, height slices, a vertical section with forest height, and Pauli GeoTIFFs of selected heights. | ~3 min |

\*Full frame, 32-core workstation, data on the external drive.

Notebooks 1-4 run in order. Notebook 0 is only needed to find or download a stack. Notebook 4 Part B needs notebook 2 (phase conventions), and its section/height plots use notebook 3.

## Running

Environment: `~/miniconda3/envs/tomosar` (numpy, xarray, netCDF4, rasterio, pystac-client, shapely, folium, polsarpro, nbconvert). Open the notebooks **from this folder**. The first cell finds `TomoSAR_Scripts/` (the `biomass_tomo` package) by itself.


### Downloading (notebook 0)

The search is anonymous. Downloading needs two files in `TomoSAR_Scripts/`, which the code reads but never prints:
- `token.txt`: your MAAP offline token;
- `maap_credentials.txt`: `CLIENT_ID=offline-token` and `CLIENT_SECRET=...`, the public value from the ESA course notebook. Copy `maap_credentials.example.txt` to create it.

Then set `DOWNLOAD = True`. Access tokens are refreshed automatically during long downloads, and the catalogue is retried if it answers with an error page.

## Key settings

| notebook | setting | default | meaning |
|---|---|---|---|
| 0 | `AOI_BBOX` | central Guainía, 0.25° box | search area; keep it smaller than a frame, since frames are tilted |
| 0 | `STACK_KEY` | `S1_M03_T029_F152_v02` | `None` = stack with most acquisitions fully covering the AOI |
| 0 | `DOWNLOAD`, `DOWNLOAD_MODE` | `False`, `"zip"` | `"assets"` downloads only the needed files |
| 1 | `CROP` | `None` (full frame) | e.g. `{"azimuth": (8000, 12000), "range": (0, 1315)}` |
| 1 | `Z_RANGE` | (−30, 80) m | must fit inside the height of ambiguity |
| 2 | `WINDOW_M` | 200 m | multilook window **on the ground** (5 × 29 samples, 145 looks) |
| 2 | `Z_VECTOR` | −30…80 m, 2 m | tomographic height axis |
| 3 | `HEIGHT_CHANNEL`, `METHOD` | `"X"`, `"Capon"` | channel and tomogram used for heights |
| 3 | `REFERENCE_HEIGHT` | `Forest_Height_Guainia.tif` | `None` disables the assessment |
| 3 | `REFERENCE_POOL` | (3, 3) cells ≈ 280 m | pooling of the reference onto the coherence grid |
| 3 | `CORRECTION` | `"bias"` | or `"linear"` (regression; shrinks towards the reference mean when $r$ is low) |
| 4 | `POL_LINES`, `BOX` | 4000 lines (27 km), (21, 3) | polarimetry block, multilook ≈ 140 m × 140 m |
| 4 | `NOISE_REMOVAL`, `NOISE_TOL_DB` | `True`, 1.5 dB | noise subtraction and the noise-LUT outlier check |
| 4 | `HEIGHTS`, `FOCUSING` | −10…50 m, 2.5 m; beamforming | Part B focusing heights and filter |
| 4 | `STRETCH`, `GEOTIFF_HEIGHTS` | (2, 98) %, [0, 20] m | colour stretch; height slices exported as Pauli GeoTIFFs |

## Outputs (`Guainia_outputs/`)

```
Guainia_outputs/
├── run_manifest.json                                   shared state of the pipeline (see below)
├── guainia_S1_M03_T029_F152_v02_stack.nc               9.6 GB stack (notebook 1)
├── tomography/
│   ├── tomograms_<tag>.nc                              Capon/beamformer per channel, intensity, kz, geolocation
│   └── coherence_{HH,VV,X}_<tag>.npz                   multibaseline coherence matrices (range, azimuth, 7, 7)
├── forest_height/
│   ├── forest_height_Capon_X_<tag>.nc                  all estimators, corrected height, pooled reference, masks
│   └── forest_height_Capon_X_<tag>.tif                 GeoTIFF, EPSG:4326, ≈ 95 m, 8 bands (see band descriptions)
└── polarimetry/
    ├── polsar_<block>.nc                               Part A: σ⁰, T3 diagonal, H/A/α, Freeman, Wishart classes
    ├── poltomo_beamforming_<block>.nc                  Part B cube (height, y, x)
    ├── pauli_rgb_<block>.tif                           Pauli RGB, 8-bit, EPSG:4326, ≈ 47 m (0 = no-data)
    ├── pauli_db_<block>.tif                            Pauli T22/T33/T11 in dB (float)
    └── pauli_rgb_z{+00,+20}m_beamforming_<block>.tif    Pauli RGB of tomographic height slices
```

`<tag>` = `az0-20765_rg0-1315_w200_s+1+1` (crop, window, LUT signs); `<block>` = `az8382-12382_rg44-1259`.

### `run_manifest.json`

Each notebook reads it at the start and adds its results at the end, so no paths or settings are copied by hand and every step uses the same conventions:

| key | written by | content |
|---|---|---|
| `stack_path`, `stack_key`, `primary`, `product_ids`, `crop`, `z_range`, `baseline`, `reference` | 1 | the stack and the analysis area |
| `phase` | 2 | LUT sign conventions (`sign_dem`, `sign_cal`, `kz_sign`), reference acquisition |
| `tomography` | 2 | tomogram/coherence paths, window, resolution |
| `forest_height` | 3 | output paths, median heights, assessment table, bias correction |
| `polarimetry` | 4 | block, output paths, calibration/noise settings, GeoTIFFs |

Re-running notebook 1 rewrites the manifest from scratch. Notebooks 2–4 then have to be re-run.

## Code

All shared code lives in the package `../biomass_tomo/`:

| module | content |
|---|---|
| `biomass_tomo/l1c.py` | L1c products: discovery, annotation grid, SLC, LUT interpolation onto the SLC grid, `inspect_product`, and `build_stack` (the stack netCDF writer). Also a command line: `python -m biomass_tomo.l1c STACK_DIR OUT.nc [--crop AZ0 AZ1 RG0 RG1]`, or `--inspect` |
| `biomass_tomo/maap.py` | MAAP authentication, STAC search (with retries), stack grouping and completion, zip/asset download |
| `biomass_tomo/stack.py` | working with the stack: AOI crop, open/crop into the course layout, baseline report, manifest, `save_netcdf` |
| `biomass_tomo/tomo.py` | coherence (`coherence_matrix` for full frames), phase signs, tomograms, height estimators, RVoG |
| `biomass_tomo/reference.py` | reference rasters: `map_to_grid` (pooling a gridded map onto the radar grid), `compare`, `to_geotiff` (float or RGB) |
| `biomass_tomo/polsar.py` | calibrated, noise-corrected T3, noise-LUT check, H–α plane |
| `biomass_tomo/poltomo.py` | polarimetric tomography: coherent focusing and per-height decompositions |
| `biomass_tomo/display.py` | maps at true ground scale, percentile/dB stretches, RGB helpers |
