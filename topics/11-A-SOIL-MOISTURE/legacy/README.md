# Legacy: exploratory notebooks of the first hackathon days

Kept for reference; not maintained. The maintained code is `../sm_pipeline/`.

| File | Content |
|---|---|
| `ismn_parsing.ipynb` | Converts an official ISMN download (Python package `ismn`) into one CSV per station (`ISMN_data/temporal/<network>+<station>.csv`) and `ISMN_data/ismn_locations.geojson`, the input format of `sm_pipeline`. |
| `Biomass L1C Point Availability.ipynb` | Which ISMN stations are covered by BIOMASS L1C products (MAAP STAC). |
| `Biomass L1C Point Extraction.ipynb` | SLC windows, multilooked intensity, closure phase and baselines at the stations. |
| `Biomass L1C Station Ancillary.ipynb` | SoilGrids soil properties and Sentinel-2 NDVI / NDMI around the stations. |
| `Biomass L1C vs Soil Moisture Plots.ipynb` | BIOMASS observables next to the ISMN soil-moisture series (figures in `plots/`). |
| `check_acquisition_plan.ipynb` | BIOMASS acquisition-plan KML files: area acquired up to a date. |
| `04-YamaguchiDecomposition/yamaguchi_stack.ipynb` | Yamaguchi decomposition of one L1C stack with PyPolSARpro. |
| `plots/` | Figures and CSV tables written by the notebooks above. |

The raw ISMN data are not in the repository (ISMN terms: no onward distribution). Create them from your own ISMN
download with `ismn_parsing.ipynb`, which writes `ISMN_data/` one level above the notebook (`11-A-SOIL-MOISTURE/ISMN_data/`).
The other notebooks read `ISMN_data/` relative to their own folder (now `legacy/ISMN_data/`): adjust the path in their first
cells or copy the folder there. Both locations are listed in `../.gitignore`.
