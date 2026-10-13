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

## TLS processing

The TLS processor validates provider outputs and offers common preparation
steps without coupling them to file parsing:

```python
from tomo_canopy.processors import TLSProcessor

processor = TLSProcessor()
mesh = processor.open("data/TLS/Ankasa/mesh.ply", crs="EPSG:32630")

bounds = processor.bounds(mesh)
local_mesh = processor.to_local_coordinates(mesh)
preview_points = processor.sample_vertices(mesh, count=100_000, seed=42)

trees = processor.open("data/TLS/Ankasa/trees.txt", crs="EPSG:32630")
metrics = processor.tree_metrics(trees)
```

Voxelization belongs to the shared `cubegen` layer so the same grid-generation
machinery can later consume TLS, ALS, GEDI, and tomogram products.

The complete processing workflow, including plots, is available in
[`examples/tls_processor.ipynb`](examples/tls_processor.ipynb).

## Harmonized data interface

The shared interface represents only the modalities that are available. Missing
sources do not require placeholder arrays:

```python
import numpy as np

from tomo_canopy.types import (
    DataCube,
    DataLayer,
    GeometryType,
    GridSpec,
    Modality,
)

grid = GridSpec(
    x=np.arange(0.5, 100.0, 1.0),
    y=np.arange(0.5, 100.0, 1.0),
    z=np.arange(0.25, 50.0, 0.5),
    crs="EPSG:32630",
)
cube = DataCube(grid=grid)
cube.add_layer(
    DataLayer(
        name="tls_occupancy",
        data=tls_occupancy,
        modality=Modality.TLS,
        geometry=GeometryType.VOLUME,
    )
)

assert cube.has_modality(Modality.TLS)
assert not cube.has_modality(Modality.GEDI)
```

Every layer that is present must match the shared grid and CRS. Consumers can
use `missing_modalities()` for optional behavior or `require_modalities()` when
a particular operation genuinely needs multiple sources.

See [`examples/data_interface.ipynb`](examples/data_interface.ipynb) for a
self-contained walkthrough with optional TLS and ALS layers.

See [`examples/tls_provider.ipynb`](examples/tls_provider.ipynb) for a complete
walkthrough. Install its optional dependencies with:

```bash
python -m pip install -e '.[examples]'
```

