<h1 align="center">tomo-canopy</h1>

<h3 align="center">Multi-scale LiDAR and BIOMASS tomogram tools for canopy visualisation</h3>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="Development status: pre-alpha" src="https://img.shields.io/badge/status-pre--alpha-orange">
  <img alt="Code style: Ruff" src="https://img.shields.io/badge/code%20style-ruff-261230">
</p>

---

**tomo-canopy** is a Python package for bringing terrestrial laser scanning
(TLS), LVIS and GEDI observations, and BIOMASS tomographic products into a shared,
Xarray-based interface. It preserves each source's natural geometry while
providing a path towards aligned 2D and 3D canopy visualisation.

The package is under active development as part of the
[BIOMASS Hackathon 2026](https://github.com/BioPAL/biomass-hackathon-2026).

## Features

| Capability | Description |
|---|---|
| **TLS provider** | Read ASCII or binary PLY point clouds and tree-segment TXT files, with memory mapping for large binary PLY data. |
| **LVIS provider** | Read complete LVIS-C Level-2 TXT footprint records and normalize all 23 RH percentiles. |
| **GEDI provider** | Query `gedidb` and return GEDI footprints, metrics, and waveforms as Xarray datasets. |
| **Tomogram provider** | Inspect and lazily open BIOMASS TOM L1D and L2A product directories, including their STAC and annotation metadata. |
| **Processing** | Validate, filter, sample, summarize, slice, and derive analysis quantities for TLS, LVIS, GEDI, and tomograms. |
| **Transforms** | Reproject irregular observations, match vertical resolution, and resample regular volumes onto a target grid. |
| **Shared data interface** | Combine point clouds, meshes, tree segments, rasters, waveforms, and regular volumes without requiring every modality to be present. |
| **Xarray first** | Providers and processors return labelled `xarray.Dataset` objects with spatial metadata. |

> [!NOTE]
> Higher-level cube-generation helpers and the final visualisation API are
> currently scaffolds for upcoming development.

---

## Installation

The distribution name is `tomo-canopy`; the Python import name is
`tomo_canopy`. The package is not yet published on PyPI, so install it from a
local clone:

```bash
git clone https://github.com/BioPAL/biomass-hackathon-2026.git
cd biomass-hackathon-2026/topics/04-3D-TOMO-CANOPY-VISUALIZER/tomo-canopy
python -m pip install -e .
```

Install the optional dependencies needed for your workflow:

```bash
# Notebooks and plotting
python -m pip install -e ".[examples]"

# GEDI access
python -m pip install -e ".[gedi]"

# LVIS Level-2 TXT access
python -m pip install -e ".[lvis]"

# Lazy BIOMASS NetCDF access
python -m pip install -e ".[tomogram]"

# Full development environment
python -m pip install -e ".[dev,examples,gedi,lvis,tomogram]"
```

| Dependency group | Main packages | Purpose |
|---|---|---|
| Core | NumPy, Xarray | Labelled scientific arrays and datasets |
| `gedi` | gedidb, pyproj | GEDI queries and coordinate transforms |
| `lvis` | pandas | LVIS L2 TXT footprint access and spatial filtering |
| `tomogram` | Dask, netCDF4 | Lazy, chunked tomogram reading |
| `examples` | JupyterLab, Matplotlib | Example notebooks and plots |
| `dev` | pytest, Ruff | Testing and code quality |

---

## Quick start

### Open a TLS point cloud

```python
from tomo_canopy.providers import TLSProvider

tls = TLSProvider().open(
    "data/TLS/Lope/LPG-01/000_T0.ply",
    properties=["branch"],
    crs="LOCAL:Lope",
)

print(tls)
```

The Lope collection is organized into the `LPG-01`, `OKO-01`, `OKO-02`, and
`OKO-03` plot directories. Each tree-level PLY is a triangular mesh with a
`branch` vertex attribute. Binary PLY arrays remain memory-mapped unless
`load=True` is requested; the Lope files themselves are ASCII PLY.

### Query GEDI data

```python
import geopandas as gpd

from tomo_canopy.providers import GEDIProvider

roi = gpd.read_file("path/to/area.geojson")
gedi = GEDIProvider(
    storage_type="s3",
    s3_bucket="dog-ext.gedidb.gedi-l2-l4-v002.0",
    url="https://s3.gfz-potsdam.de",
).open(
    roi,
    variables=["agbd", "rh"],
    start_time="2018-01-01",
    end_time="2024-07-25",
)
```

### Open LVIS footprints

```python
from tomo_canopy.providers import LVISProvider

lvis = LVISProvider().open(
    "data/lvisc/raw/LVISC2_Gabon2023_0520_R2507_051090.TXT",
    variables=["rh", "ground_elevation", "sensitivity"],
    bounds=(11.6, -0.2, 11.8, 0.1),
)

rh98 = lvis.rh.sel(percentile=98)
```

LVIS Level-2 observations are represented as irregular footprints rather than
ALS point-cloud returns. TXT products expose all 23 RH percentiles and the
complete Level-2 record. Bounding-box selection scans the large ASCII file,
so it is slower than an indexed geospatial query.

### Open a BIOMASS tomogram

```python
from tomo_canopy.providers import TomogramProvider

provider = TomogramProvider()
product = provider.inspect("data/TOM/path/to/product")
tomogram = provider.open(
    product.root,
    polarization="HH",
    chunks="auto",
)

print(product.product_level, tomogram)
```

L1D products retain native SAR coordinates. L2A products are exposed as a
regular geographic `x/y/z` volume after their reconstructed axes are checked
against the product metadata.

Process the native values and explicitly match their physical vertical
resolution when needed:

```python
from tomo_canopy.processors import TomogramProcessor
from tomo_canopy.transforms import match_vertical_resolution

processor = TomogramProcessor()
power = processor.to_power(tomogram)
power_db = processor.to_db(power, floor_db=-40.0)
vertical_slice = processor.vertical_slice(
    power_db,
    axis="x",
    coordinate=float(power_db.x.mean()),
)
resolution_matched = match_vertical_resolution(
    power,
    target_resolution=20.0,
    variables=["tomographic_power"],
)
```

`vertical_sampling` records the coordinate spacing, while
`vertical_resolution` records the broader physical resolving capability; a
finely sampled tomogram should not be interpreted as having equally fine
independent vertical detail.

---

## Shared data interface

`DataCube` is a geometry-aware collection rather than a single dense array.
All layers share a CRS, but only raster and volume layers must align with the
cube grid. Native TLS point clouds, meshes, LVIS footprints, GEDI shots, and
waveforms keep their irregular coordinates.

```python
import numpy as np

from tomo_canopy.types import DataCube, DataLayer, GeometryType, GridSpec, Modality

grid = GridSpec(
    x=np.arange(-150.0, 151.0, 1.0),
    y=np.arange(-150.0, 151.0, 1.0),
    z=np.arange(-50.0, 101.0, 1.0),
    crs="LOCAL:Lope",
)
cube = DataCube(grid=grid)
cube.add_layer(
    DataLayer(
        name="tls_mesh",
        data=tls,
        modality=Modality.TLS,
        geometry=GeometryType.TRIANGULAR_MESH,
    )
)

print(cube.modalities)
```

Any subset of modalities is valid: a cube can contain only TLS, only a
tomogram, or several compatible sources.

---

## Examples

The notebooks use real datasets when paths or service access are available,
and document any setup required at the top of each notebook.

| Notebook | Demonstrates |
|---|---|
| [`tls_provider.ipynb`](examples/tls_provider.ipynb) | Reading PLY point clouds and tree-segment files |
| [`tls_processor.ipynb`](examples/tls_processor.ipynb) | TLS validation, sampling, bounds, tree metrics, and plotting |
| [`lvis_processor.ipynb`](examples/lvis_processor.ipynb) | LVIS filtering, RH metrics, canopy elevation, reprojection, and interface integration |
| [`gedi_provider.ipynb`](examples/gedi_provider.ipynb) | Querying GEDI observations with `gedidb` |
| [`gedi_processor.ipynb`](examples/gedi_processor.ipynb) | GEDI filtering, subsetting, and spatial preparation |
| [`tomogram_provider.ipynb`](examples/tomogram_provider.ipynb) | Inspecting and opening BIOMASS TOM L1D/L2A products |
| [`tomogram_processor.ipynb`](examples/tomogram_processor.ipynb) | Tomogram power, slices, profiles, and resolution-aware preparation |
| [`data_interface.ipynb`](examples/data_interface.ipynb) | Combining native and gridded layers and plotting them in 2D and 3D |

Start JupyterLab from the package directory:

```bash
jupyter lab examples/
```

---

## Architecture

```text
TLS files      LVIS L2 files      GEDI service      BIOMASS TOM products
    │               │                  │                      │
    └───────────────┴──────── providers ┴──────────────────────┘
                     │
              xarray.Dataset
                     │
          processors / transforms
                     │
                  DataLayer
                     │
                  DataCube
                     │
        2D and 3D visualisation
```

```text
tomo_canopy/
├── providers/       # Data access and format normalization
├── processors/      # Modality-specific processing
├── types/           # DataLayer, GridSpec, and DataCube contracts
├── cubegen/         # Cube-generation workflows (planned)
├── methods/         # Shared algorithms (planned)
├── transforms/      # Resolution matching and regular-grid resampling
└── visualisation/   # Rendering interfaces (planned)
```

---

## Development

```bash
python -m pip install -e ".[dev,examples,gedi,lvis,tomogram]"
python -m pytest
python -m ruff check tomo_canopy tests
```

Small TLS fixtures are kept under `tests/data/`; large source datasets under
`data/` remain outside version control.

