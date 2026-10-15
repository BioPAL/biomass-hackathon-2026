# Flooded forest

**Group `11-B-FLOODED-FOREST`** — BIOMASS / P-band applications for flooded forest (e.g. inundation under canopy).

One entry point: [`run_pipeline.py`](run_pipeline.py). For a ROI, it does the following:

1. Searches the MAAP catalogue.
2. Extracts the STA Level-1C stacks over the common area of the selected stacks, applying the SKP calibration and flattening phases.
3. Computes the MBMP coherence and lets you remove scenes per stack in a manual review.
4. Computes the PolSARpro decompositions (T3, Pauli, HSI, H/A/alpha_averaged) of the kept scenes with the same averaging window.
5. Saves the figures.

Everything generated goes into `outputs/` (git-ignored).

## Setup

Create the conda environment with BPS (`biomass-bps` channel) and PolSARpro (`conda-forge`):

```bash
conda env create -f environment.yml
conda activate flooded-forest
```

BPS is optional at runtime. When it is importable, the MBMP coherence is cross-checked against `bps main_correlation_estimation_SR`. PolSARpro is required for the PolSAR step; skip that step with `--skip-polsar` if PolSARpro is not installed.

## Configure (once)

1. **`config.yaml`** — user-specific settings:
   - the credentials path
   - the ROI list
   - `outputs_dir`
   - extraction defaults (`max_files`, `min_coverage`, `workers`)
   - the phase corrections (`apply_skp`, `apply_flat`)
   - the averaging window (`averaging_window`, in metres)
   - the PolSAR incidence angle of each swath (`swath_incidence_deg`)

   Paths are relative to this folder.
2. **`credentials.txt`** (git-ignored, next to `config.yaml` by default) — either a raw MAAP offline token, or:
   ```
   OFFLINE_TOKEN=eyJ...
   CLIENT_ID=offline-token
   CLIENT_SECRET=...
   ```
   Get the offline token from your ESA MAAP account (see the MAAP documentation). The catalogue search runs without the token, because the MAAP STAC API rejects authenticated search requests. The token is only used to download assets (quicklooks, LUTs, COG tiles).

## Run

```bash
# a preset ROI (from rois.json, substring match works)
python run_pipeline.py --roi "Lac Télé Community Reserve (LTCR)" --start 2026-05-01 --end 2026-10-13

# a custom ROI given as 4 coordinates (west south east north)
python run_pipeline.py --bbox 16.98 0.94 17.48 1.44 --name "Lac Tele" --start 2026-05-01 --end 2026-10-13

# keep specific stacks without the interactive prompt, also save the catalogue Pauli quicklooks
python run_pipeline.py --roi bolobo --start 2026-05-01 --stacks "0,2" --quicklook

# reuse the extracted stacks: coherence + PolSAR with another averaging window
python run_pipeline.py --roi bolobo --start 2026-05-01 --skip-search --skip-extract --window 300

# redo only the manual scene review and the PolSAR of the kept scenes
python run_pipeline.py --roi bolobo --start 2026-05-01 --skip-search --skip-extract

# no questions at all: recommended stacks, saved scene selection (all scenes the first time)
python run_pipeline.py --roi bolobo --start 2026-05-01 --yes

python run_pipeline.py --list-rois    # preset ROIs
python run_pipeline.py --help         # all flags
```

## Pipeline steps

| Step | What happens | Code | Output (`outputs/…`) |
| --- | --- | --- | --- |
| 1. ROI | Preset name, 4 coordinates, or interactive prompt. | `core/roi.py` | — |
| 2. Search | MAAP STAC search of the STA L1c products overlapping the ROI. | `catalogue/search.py` | `catalogue/sta_level1c_by_roi.json` |
| 3. Quicklooks (`--quicklook`) | Pauli quicklook PNG of every catalogue product. | `catalogue/quicklook.py` | `<roi>/quicklooks/catalogue/` |
| 4. Extraction | Stack table and folium map, then interactive selection. The common area is the ROI ∩ every selected stack. Each product is range-read, corrected with `s·exp(j·skp)·exp(j·flat)` and saved. | `extraction/` | `<roi>/extraction/` |
| 5. Stack quicklooks | Pauli quicklook of every stack product. | `catalogue/quicklook.py` | `<roi>/quicklooks/<stack>/` |
| 6. Coherence | MBMP (HH, XP, VV) and single-pol coherence of all scenes on the averaging grid. | `processing/coherence.py` | `<roi>/coherence/<stack>/` |
| 7. Scene review | Per stack, the mean \|γ\| MBMP matrix and the HH, HV, VV pair grids open in windows. You type the scenes to keep in the terminal. | `plotting/coherence.py`, `core/scenes.py` | `<roi>/extraction/<stack>/scene_selection.json` |
| 8. Stack figures | HH intensity of every kept scene, HH/HV time series. | `plotting/stacks.py` | `<roi>/extraction/…/figures/` |
| 9. PolSAR | For every **kept** scene: PolSARpro `S_to_T3`, with T3 averaged on the **same window and grid** as the coherence, then `pauli_rgb` and `h_a_alpha` (eigenvalues, alpha_1..3, entropy, anisotropy, alpha_averaged). Figures per scene, a chronological Pauli/HSI comparison per stack and the dominant-mechanism time series. | `processing/polsar.py`, `plotting/polsar.py` | `<roi>/polsar/<stack>/` |

How the steps behave:

- **Manual scene review (step 7):** for each stack, four windows open: the mean |γ| of the MBMP matrix and the HH, HV and VV single-pol pair grids, with every scene labelled `[index] date`. The terminal also prints each scene's mean |γ| with its ±3-day partners as a hint, flagging values below mean − 1 std. Type the scenes to keep (`0,2,4-6`, `all`, Enter = the current selection) and confirm.
  - The choice is saved in `scene_selection.json`. Steps 8 and 9 (stack figures, PolSAR, temporal RGB, comparisons, mechanism plot) use only the kept scenes. The coherence is not recomputed, because each pair's |γ| does not depend on the other scenes.
  - `--skip-review`, `--yes` or a non-interactive terminal reuse the saved selection (all scenes the first time).
- **Stack selection:** the selection at step 4 is interactive. A table of stacks is printed and `stack_selection_map.html` opens in the browser; Enter keeps the recommended stacks. `--stacks "0,2,5-7"` or `--yes` skip the prompt, and `--dry-run` previews the windows without downloading.
- **Geometry once per stack:** the members of a stack are coregistered on the reference grid. So one LUT per stack (the reference product's) gives the geometry for every product: footprint, window, lat/lon, incidence and pixel spacings. Each other product's LUT only contributes its own phase screens (flattening, SKP, kz). On a re-run, products whose window and phase corrections are unchanged are skipped, and their screens are read back from `phase_lut.nc`.
- **Phase corrections:** this follows Thesis_v2 `load_or_extract_data_lat_lon`. The LUT is read with netCDF4; BPS's reader would need the whole product folder. It is interpolated onto the pixels as BPS `read_lut_annotation(fit_to_l1c_raster=True)` does: bilinear, with the SKP phase interpolated through its sine and cosine. No-data nodes become 0 rad. Set `apply_skp` / `apply_flat` in `config.yaml`.
- **One averaging window:** `averaging_window` (m on ground) becomes a pixel window per stack with the BPS rule `NW = int(L / spacing / 2) * 2 + 1`, subsampling `(NW − 1) / 2`. Coherence and PolSAR use that same sparse boxcar and decimation (`processing/averaging.py`), so their maps share cells and resolution.
- **PolSAR incidence:** S is scaled by `sqrt(sin θ)`, with one θ per swath (`swath_incidence_deg`, the BIOMASS mean swath incidence).
- **PolSAR quantities**, with p_i = λ_i / (λ1 + λ2 + λ3) and λ1 ≥ λ2 ≥ λ3 the eigenvalues of T3:
  - entropy H = −Σ p_i log₃ p_i
  - anisotropy A = (p2 − p3) / (p2 + p3)
  - alpha_i = scattering angle of the eigenvector of λ_i
  - alpha_averaged = Σ p_i alpha_i, the average of alpha_1, alpha_2 and alpha_3 weighted by p_i
  - span = T11 + T22 + T33
- **HSI composite:** hue = 4 · alpha_averaged (0° red, 120° green, 240° blue), saturation = 1 − H, intensity = span in dB stretched between its stack-wide 2nd and 98th percentile.
- **Pauli RGB figures:** sqrt(T22), sqrt(T33), sqrt(T11), clipped at their stack-wide 98 % quantile, so the scenes of a stack are comparable.
- **Dominant-mechanism time series:** the mechanism of each cell is the one of the largest eigenvalue λ1, from its alpha_1: surface < 40° ≤ volume ≤ 50° < double bounce (`processing/polsar.py` `dominant_mechanism`). The cells of each mechanism are counted per kept scene, on the cells valid in every scene, against the temporal baseline (days since the first kept scene).
- **Colour maps:** every scalar map and matrix uses `jet`.
- **Re-runs:** coherence and PolSAR are recomputed only when missing or out of date (products, corrections, window, θ). `--force` recomputes them.

## Outputs (all in `outputs/`, never committed)

```
outputs/
├── catalogue/sta_level1c_by_roi.json
└── <roi>/
    ├── extraction/
    │   ├── stack_selection_map.html, extraction_summary.json, common_area.geojson, extraction.log
    │   ├── figures/timeseries.png                 mean HH/HV in the common area per date
    │   └── <stack>/
    │       ├── BIO_S1_STA__...tif                 complex64 HH HV VH VV, SKP + flattening applied (GCPs)
    │       ├── overlap_mask.tif, latlon.tif       common-area mask, lat/lon of every pixel
    │       ├── geometry_lut.nc                    reference LUT geometry (lat, lon, height, incidence)
    │       ├── phase_lut.nc                       flattening, SKP (+ quality), kz of every product
    │       ├── stack_info.json                    window, LUT steps, pixel spacings, products, corrections
    │       ├── scene_selection.json               manual review: kept / removed scenes
    │       └── figures/hh_dates.png               HH intensity of every kept scene
    ├── quicklooks/{catalogue,<stack>}/            Pauli quicklook PNGs
    ├── coherence/<stack>/
    │   ├── coherence_w<L>m.npz, .json             MBMP + single-pol coherence, valid fraction, lat/lon
    │   ├── coherence_scenes.md                    window, scenes kept / removed, 3-day |γ| hint table
    │   └── figures/                               mean |γ| MBMP matrix, |γ| pair grids (all scenes), temporal RGB (kept)
    └── polsar/<stack>/
        ├── polsar_w<L>m.npz, .json                T3, Pauli RGB, lambdas, alphas, H, A, alpha_averaged, span (kept scenes)
        ├── mechanism_counts.json                  cells per dominant mechanism per kept scene
        └── figures/
            ├── <date>_{pauli,hsi,alphas,h_a_alpha,h_alpha}.png
            ├── stack_pauli_hsi.png                kept scenes in chronological order: Pauli RGB (top), HSI (bottom)
            └── mechanism_timeseries.png           surface / volume / double bounce cells vs temporal baseline
```

All figures are in radar geometry (rows = azimuth, columns = slant range, drawn with the true ground aspect); nothing is ground-projected.

## Layout

```
├── run_pipeline.py        entry point: runs everything
├── config.yaml            user-specific settings (edit this)
├── environment.yml        conda env: BPS + PolSARpro + pipeline dependencies
├── rois.json              preset ROI list
├── flooded_forest/        all pipeline functions, by category
│   ├── core/              config + output layout, MAAP auth, ROI, shared I/O
│   ├── catalogue/         STAC search, Pauli quicklooks
│   ├── extraction/        stack search/selection, LUT reading + phase corrections, geometry, extraction
│   ├── processing/        shared averaging, MBMP coherence, PolSAR (PolSARpro)
│   └── plotting/          stack, coherence and PolSAR figures
├── tools/                 one-off scripts (refine_roi, kml globe, DAHITI water levels)
├── archive/               code the pipeline no longer uses (marcos_dev, sams_dev, notebooks, ...), see archive/README.md
├── outputs/               generated/downloaded files (git-ignored)
└── resources.md           papers, links, ROI notes
```

## Tools (`tools/`, one-off scripts)

```bash
python tools/refine_roi.py                  # refine ROIs to the best asc/desc C01 frame intersection
python tools/kml_files_visualisation.py     # major-cycle KML footprints on a globe (pass your KML paths)
python tools/water_level_dahiti.py --dahiti-id 15307   # DAHITI water levels (needs DAHITI_API_KEY env var)
```

Their figures and downloads go to `outputs/tools/` and `outputs/validation/`.
