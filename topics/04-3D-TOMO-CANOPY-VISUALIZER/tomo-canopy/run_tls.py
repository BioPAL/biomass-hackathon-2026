import sys
from tomo_canopy.visualisation import TomographicPlotter
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from tomo_canopy.types import (
    DataCube,
    DataLayer,
    GeometryType,
    GridSpec,
    Modality,
)
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tomo_canopy.processors import TLSProcessor

TLS_CRS = "EPSG:32630"
TLS_VOXEL_SIZE = 1.0  # metres
tls_candidates = [
    Path("data/TLS/Ankasa//c01_r02_subsampled_1cm_trees_mesh.ply"),
    Path("../data/TLS/Ankasa//c01_r02_subsampled_1cm_trees_mesh.ply"),
]

TLS_PATH = next((path for path in tls_candidates if path.is_file()), tls_candidates[0])

tls_processor = TLSProcessor()
tls_mesh = tls_processor.open(TLS_PATH, crs=TLS_CRS, load=False)
tls_bounds = tls_processor.bounds(tls_mesh)

# Discard connectivity for this visualization while retaining every vertex.
tls_points = tls_mesh.drop_vars("face_vertices").rename({"vertex": "point"})
tls_points.attrs = {**tls_mesh.attrs, "geometry_type": "point_cloud"}
tls_point_layer = DataLayer(
    name="tls_points",
    data=tls_points,
    modality=Modality.TLS,
    geometry=GeometryType.POINT_CLOUD,
)

minimum = dict(zip(tls_bounds.axis.values, tls_bounds.minimum.values))
maximum = dict(zip(tls_bounds.axis.values, tls_bounds.maximum.values))
edges = {
    axis: np.arange(
        np.floor(minimum[axis] / TLS_VOXEL_SIZE) * TLS_VOXEL_SIZE,
        np.ceil(maximum[axis] / TLS_VOXEL_SIZE) * TLS_VOXEL_SIZE + TLS_VOXEL_SIZE,
        TLS_VOXEL_SIZE,
    )
    for axis in ("x", "y", "z")
}
centres = {
    axis: (axis_edges[:-1] + axis_edges[1:]) / 2 for axis, axis_edges in edges.items()
}
grid = GridSpec(x=centres["x"], y=centres["y"], z=centres["z"], crs=TLS_CRS)
print(f"Native TLS points: {tls_points.sizes['point']:,}")
print("Grid shape (z, y, x):", grid.shape)

print(tls_point_layer)
# cube = DataCube(grid=grid)
# cube.add_layer(tls_point_layer)
TomographicPlotter.from_data_layer(tls_point_layer).run()
