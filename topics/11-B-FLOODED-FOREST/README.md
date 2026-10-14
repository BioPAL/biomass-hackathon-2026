# Flooded forest

**Group `11-B-FLOODED-FOREST`** — BIOMASS / P-band applications for flooded forest (e.g. inundation under canopy).

One entry point: [`run_pipeline.py`](run_pipeline.py). From a ROI it searches the MAAP catalogue, shows Pauli quicklooks, selects and extracts STA Level-1C stacks, and plots them. Everything downloaded or generated lands in `assets/` (git-ignored).

## Setup

```bash
conda activate biomap
pip install -r requirements.txt
```

## Configure (once)

1. **`config.yaml`** — user-specific settings. Edit `credentials_path`, the ROI list path, and the extraction defaults (`max_files`, `min_coverage`, `workers`). Paths are relative to this folder.
2. **`credentials.txt`** (git-ignored, next to `config.yaml` by default) — either a raw MAAP offline token, or:
   ```
   OFFLINE_TOKEN=eyJ...
   CLIENT_ID=offline-token
   CLIENT_SECRET=...
   ```
   Get the offline token from your ESA MAAP account (see the MAAP documentation).

   Note: the catalogue **search runs without the token** — the MAAP STAC API
   rejects authenticated search requests, and its token service is occasionally
   flaky ("Invalid token" 400s), so the search is anonymous with automatic
   retries. The token is only used to download assets (quicklooks, COG tiles),
   same as `marcos_dev`.

## Run

```bash
# a preset ROI (from rois.json, substring match works)
python run_pipeline.py --roi "Lac Télé Community Reserve (LTCR)" --start 2026-05-01 --end 2026-10-13

# a custom ROI given as 4 coordinates (west south east north)
python run_pipeline.py --bbox 16.98 0.94 17.48 1.44 --name "Lac Tele" --start 2026-05-01 --end 2026-10-13

# also save Pauli quicklooks, and keep specific stacks without the interactive prompt
python run_pipeline.py --roi bolobo --start 2026-05-01 --quicklook --stacks "0,2"

# Pauli maps straight from one catalogue product (no stack extraction needed)
python run_pipeline.py --roi bolobo --start 2026-05-01 --pauli --skip-extract --skip-plots

# plots only, from already-extracted stacks
python run_pipeline.py --roi bolobo --start 2026-05-01 --skip-search --skip-extract --show

python run_pipeline.py --list-rois    # preset ROIs
python run_pipeline.py --help         # all flags (extraction flags pass through to the extractor)
```

## Pipeline steps

| Step | What happens | Output |
| --- | --- | --- |
| 1. ROI | preset name, 4 coordinates, or interactive prompt | — |
| 2. Search | MAAP STAC search of STA L1c products overlapping the ROI (`flooded_forest/search.py`) | `assets/catalogue/sta_level1c_by_roi.json` |
| 3. Quicklooks (optional, `--quicklook` / `--pauli`) | Pauli RGB PNG of the scene; `--pauli` also renders Pauli/component/mechanism maps from one product's COGs | `assets/quicklooks/`, `assets/figures/` |
| 4. Stack selection + extraction | runs `marcos_dev/1_data_extraction/extract_l1c_roi.py`: stack table + interactive folium map in the browser, then downloads the selected stacks | `assets/stacks/<roi>/` |
| 5. Post-processing + plots | QGIS dB rasters + quicklooks per stack; per-stack figures and ROI overview/time series (`flooded_forest/plots.py`, `flooded_forest/qgis.py`) | `assets/stacks/<roi>/` (see below) |

The stack selection at step 4 is interactive: a table of stacks is printed and a map (`stack_selection_map.html`) opens in the browser; Enter keeps the recommended stacks. `--stacks "0,2,5-7"` or `--yes` skip the prompt. Use `--dry-run` to preview the windows without downloading.

## Outputs (all in `assets/`, never committed)

```
assets/
├── catalogue/sta_level1c_by_roi.json       catalogue search results, one entry per ROI
└── stacks/<roi>/                           one folder per ROI/stack
    ├── figures/                            overview map, HH/HV time series (dpi 300)
    ├── quicklooks/                         Pauli quicklooks of the catalogue products
    ├── stack_selection_map.html, extraction_summary.json, common_area.geojson, extraction.log
    └── STA1_ascending_.../                 one folder per extracted stack
        ├── BIO_S1_STA__...tif              complex64 products (4 bands HH HV VH VV)
        ├── BIO_S1_STA__..._dB.tif          QGIS-friendly float32 dB version of each product
        ├── figures/                        Pauli RGB, HH backscatter per date (dpi 300)
        ├── quicklooks/                     Pauli quicklook of each product
        ├── overlap_mask.tif, latlon.tif, geometry_lut.nc, stack_info.json
```

## QGIS

The product tifs are **complex (CFloat32)** — QGIS cannot render complex rasters. For each product the pipeline also writes a `<product_id>_dB.tif` (float32 intensity in dB, same GCP georeferencing, NaN nodata, built-in overviews): open **those** in QGIS. The complex originals stay untouched for processing.

## Layout

```
├── run_pipeline.py        entry point: runs everything
├── config.yaml            user-specific settings (edit this)
├── rois.json              preset ROI list
├── flooded_forest/        pipeline package (search, quicklook, extract, plots, qgis)
├── tools/                 one-off scripts (refine_roi, kml globe, DAHITI water levels)
├── marcos_dev/            collaborator code (extraction + visualization) — do not edit here
├── sams_dev/              development tests — do not edit here
├── assets/                generated/downloaded files (git-ignored)
└── resources.md           papers, links, ROI notes
```

`marcos_dev/` and `sams_dev/` are development areas kept as-is; the pipeline calls `marcos_dev`'s extractor as a subprocess. More plots: `marcos_dev/1_data_extraction/visualize_stacks.ipynb` (run it with an ROI folder inside `assets/stacks/`).

## Tools (`tools/`, one-off scripts)

```bash
python tools/refine_roi.py                  # refine ROIs to the best asc/desc C01 frame intersection
python tools/kml_files_visualisation.py     # major-cycle KML footprints on a globe (pass your KML paths)
python tools/water_level_dahiti.py --dahiti-id 15307   # DAHITI water levels (needs DAHITI_API_KEY env var)
```

