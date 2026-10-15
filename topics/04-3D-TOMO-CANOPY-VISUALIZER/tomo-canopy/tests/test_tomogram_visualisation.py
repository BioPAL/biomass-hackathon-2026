"""Contract tests for TOM visualisation entry points."""

import numpy as np
import pytest
import xarray as xr

from tomo_canopy.cubegen import cube_from_layers
from tomo_canopy.processors import TomogramProcessor
from tomo_canopy.types import DataLayer, GeometryType, Modality
from tomo_canopy.visualisation import (
    CanopyViewer,
    InteractiveTomogramConfig,
    InteractiveTomogramViewer,
)


def _cube(prepared: bool = True):
    values = np.ones((1, 3, 2, 2), dtype=complex)
    dataset = xr.Dataset(
        {"tomographic_cube": (("polarization", "z", "y", "x"), values)},
        coords={
            "polarization": ["HV"],
            "z": [-10.0, 10.0, 30.0],
            "y": [1.0, 2.0],
            "x": [10.0, 11.0],
        },
        attrs={
            "product_level": "L2A",
            "coordinate_space": "geographic",
            "crs": "EPSG:4979",
        },
    )
    if prepared:
        processor = TomogramProcessor()
        dataset = processor.prepare_vertical_structure(dataset)
        dataset = processor.to_db(dataset, floor_db=-35.0)
    layer = DataLayer(
        "tomogram", dataset, Modality.TOMOGRAM, GeometryType.VOLUME
    )
    return cube_from_layers([layer])


def test_viewer_requires_explicitly_prepared_variable() -> None:
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="prepare_vertical_structure"):
        CanopyViewer(_cube(prepared=False)).tomogram_bands(layer="tomogram")


def test_viewer_plots_prepared_bands() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    figure, axes = CanopyViewer(_cube()).tomogram_bands(layer="tomogram")

    assert axes.size >= 3
    plt.close(figure)


def test_viewer_plots_vertical_slice() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    figure, axis = CanopyViewer(_cube()).tomogram_vertical_slice(
        layer="tomogram",
        axis="x",
        coordinate=10.8,
        polarization="HV",
        height_range=(-10.0, 10.0),
        figsize=(12.0, 3.0),
    )

    assert "north offset" in axis.get_xlabel()
    assert "x=11" in axis.get_title()
    assert axis.get_ylim() == pytest.approx((-10.0, 10.0))
    assert figure.get_size_inches() == pytest.approx((12.0, 3.0))
    plt.close(figure)


def test_vertical_slice_requires_prepared_display_variable() -> None:
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="prepare_vertical_structure"):
        CanopyViewer(_cube(prepared=False)).tomogram_vertical_slice(
            layer="tomogram"
        )


def test_vertical_slice_rejects_empty_height_range() -> None:
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="does not intersect"):
        CanopyViewer(_cube()).tomogram_vertical_slice(
            layer="tomogram", height_range=(100.0, 120.0)
        )


def test_interactive_viewer_consumes_prepared_cube_without_pyvista() -> None:
    viewer = CanopyViewer(_cube()).interactive_tomogram(
        layer="tomogram",
        polarization="HV",
        config=InteractiveTomogramConfig(column_quantile=0.5),
    )

    assert isinstance(viewer, InteractiveTomogramViewer)
    assert viewer.state["column_quantile"] == 0.5
    energy = viewer._energy(0.5, normalize=True, filter_after=False)
    assert energy.shape == (3, 2, 2)
    assert np.isfinite(energy).all()


def test_interactive_viewer_rejects_complex_amplitude() -> None:
    with pytest.raises(ValueError, match="requires power"):
        InteractiveTomogramViewer(
            _cube(prepared=False), layer="tomogram", polarization="HV"
        )
