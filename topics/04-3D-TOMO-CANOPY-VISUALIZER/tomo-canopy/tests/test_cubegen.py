"""Tests for explicit multimodal cube assembly."""

import numpy as np
import xarray as xr

from tomo_canopy.cubegen import DataCubeBuilder, cube_from_layers
from tomo_canopy.types import DataLayer, GeometryType, GridSpec, Modality


def _volume_layer() -> DataLayer:
    dataset = xr.Dataset(
        {"power": (("z", "y", "x"), np.ones((2, 2, 2)))},
        coords={"x": [10.0, 11.0], "y": [1.0, 2.0], "z": [0.0, 20.0]},
        attrs={"crs": "EPSG:4979"},
    )
    return DataLayer(
        "tomogram", dataset, Modality.TOMOGRAM, GeometryType.VOLUME
    )


def test_builder_supports_chaining() -> None:
    layer = _volume_layer()
    grid = GridSpec(layer.data.x, layer.data.y, layer.data.z, layer.crs)

    cube = DataCubeBuilder(grid).add(layer).build()

    assert cube.layers["tomogram"] is layer


def test_cube_from_layers_infers_grid_from_volume() -> None:
    layer = _volume_layer()

    cube = cube_from_layers([layer])

    assert cube.grid.shape == (2, 2, 2)
    assert cube.grid.crs == "EPSG:4979"
