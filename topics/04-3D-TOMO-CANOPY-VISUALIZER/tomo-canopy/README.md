# tomo-canopy

Multi-scale LiDAR and BIOMASS tomogram processing and visualisation tools.

## Package layout

- `providers`: readers and adapters for TLS, ALS, GEDI, and tomogram data
- `processors`: processing pipelines and data preparation
- `cubegen`: generation and assembly of three-dimensional data cubes
- `methods`: scientific algorithms and analysis methods
- `transforms`: spatial, coordinate, and value transformations
- `visualisation`: interactive and static visualisation backends

Like `alsdb`, this project uses a flat package layout. The distribution is
named `tomo-canopy`; import it in Python as `tomo_canopy`.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

## TLS point clouds

Open an ASCII or binary PLY point cloud as an Xarray dataset:

```python
from tomo_canopy.providers import TLSProvider

tls = TLSProvider().open("data/TLS/Ankasa/point_cloud.ply", crs="EPSG:32630")
print(tls)
```

Point clouds use a `point` dimension. Meshes use `vertex`, `face`, and `corner`
dimensions and expose triangle connectivity as `face_vertices`. In both cases,
`x`, `y`, and `z` are coordinates and other PLY vertex properties are data
variables. Binary files are memory-mapped unless `load=True` is supplied.

The companion tree-structure text format is supported by the same provider:

```python
trees = TLSProvider().open("data/TLS/Ankasa/trees.txt", crs="EPSG:32630")
```

It produces a flattened `segment` dimension with tree and segment identifiers,
radius, parent connectivity, section identifiers, and spatial coordinates.

See [`examples/tls_provider.ipynb`](examples/tls_provider.ipynb) for a complete
walkthrough. Install its optional dependencies with:

```bash
python -m pip install -e '.[examples]'
```

