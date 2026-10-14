# Archive

Code that the pipeline (`run_pipeline.py` + `flooded_forest/`) no longer uses, kept untouched for reference. Nothing in the pipeline imports from here, and the scripts and notebooks are not maintained. Some of them expect the old layout (`assets/`, `find_stack_files.py` next to `config.yaml`), so they may not run as they are.

| Path | What it was | Where it lives now |
| --- | --- | --- |
| `marcos_dev/1_data_extraction/extract_l1c_roi.py` | stack extractor, run as a subprocess | `flooded_forest/extraction/` (`stacks`, `lut`, `geometry`, `extract`), now with SKP + flattening applied when saving |
| `marcos_dev/1_data_extraction/stack_selection.py` | interactive stack table + folium map | `flooded_forest/extraction/selection.py` |
| `marcos_dev/1_data_extraction/compute_coherence.py` | MBMP coherence, run as a subprocess | `flooded_forest/processing/{averaging,coherence}.py` |
| `marcos_dev/1_data_extraction/visualize_*.ipynb`, `marcos_dev/*.ipynb` | exploration notebooks | — |
| `polsar_stack.ipynb` | PolSARpro Pauli / T3 / H-A-alpha with its own downloads | `flooded_forest/processing/polsar.py`, `flooded_forest/plotting/polsar.py`, running on the pipeline's extracted stacks |
| `data_view.ipynb`, `find_stack_files.py`, `georef_quicklook.py` | catalogue search notebook, KML-georeferenced quicklooks, Pauli GeoTIFFs resampled to EPSG:4326 | search: `flooded_forest/catalogue/`; ground projection removed |
| `flooded_forest/plots.py` | lat/lon (geocoded) stack figures, catalogue Pauli tiles (`--pauli`) | radar-geometry figures in `flooded_forest/plotting/` |
| `flooded_forest/qgis.py` | QGIS `*_dB.tif` rasters with GCPs | removed (ground projection) |
| `flooded_forest/extract.py` | subprocess wrapper of the extractor | removed |
| `NISAR_extractor.py` | NISAR GCOV Pauli reader | — |
| `sams_dev/read_water_level.py` | water-level development script | — |
