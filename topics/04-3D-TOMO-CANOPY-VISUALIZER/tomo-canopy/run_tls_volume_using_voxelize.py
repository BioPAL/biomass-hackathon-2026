from tomo_canopy.visualisation import TomographicPlotter
import numpy as np
import xarray as xr

from tomo_canopy.types import (
    DataLayer,
    GeometryType,
    GridSpec,
    Modality,
)
from pathlib import Path

from tomo_canopy.processors import TLSProcessor

TLS_CRS = "EPSG:32630"
TLS_VOXEL_SIZE = 5.0  # metres
tls_candidates = [
    Path("data/c01_r02_subsampled_1cm_trees_mesh.ply"),
    # Path("../data/c01_r02_subsampled_1cm_trees_mesh.ply"),
]
TLS_PATH = next((path for path in tls_candidates if path.is_file()), tls_candidates[0])

tls_processor = TLSProcessor()
tls_mesh = tls_processor.open(TLS_PATH, crs=TLS_CRS, load=False)
tls_points = tls_mesh.drop_vars("face_vertices").rename({"vertex": "point"})
tls_points.attrs = {**tls_mesh.attrs, "geometry_type": "point_cloud"}

tls_dataset = tls_processor.voxelize(
    tls_points,
    TLS_VOXEL_SIZE,
    crs=TLS_CRS,
    bounds=tls_processor.bounds(tls_mesh),
    source=str(TLS_PATH.resolve()),
)
tls_layer = DataLayer(
    name="tls_occupancy",
    data=tls_dataset,
    modality=Modality.TLS,
    geometry=GeometryType.VOLUME,
)

TomographicPlotter.from_data_layer(tls_layer, "bubbles").run()
