"""Tests for BIOMASS tomographic product access."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

netcdf4 = pytest.importorskip("netCDF4")

from tomo_canopy.providers import TomogramProvider, open_tomogram  # noqa: E402


def make_product(tmp_path: Path, level: str) -> Path:
    root = tmp_path / f"BIO_TP_TOM_{level}_TEST"
    measurement_dir = root / "measurement"
    annotation_dir = root / "annotation"
    measurement_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    measurement = measurement_dir / "cube.nc"
    annotation = annotation_dir / "annotation.xml"
    lut = annotation_dir / "lut.nc"

    with netcdf4.Dataset(measurement, "w") as dataset:
        for name, size in (
            ("relativeAzimuthTime", 4),
            ("slantRangeTime", 3),
            ("tomographicHeight", 3),
            ("polarizations", 3),
        ):
            dataset.createDimension(name, size)
        cube = dataset.createVariable(
            "tomographicCubes",
            "f4",
            (
                "relativeAzimuthTime",
                "slantRangeTime",
                "tomographicHeight",
                "polarizations",
            ),
            fill_value=-9999.0,
        )
        values = np.arange(108, dtype=np.float32).reshape(4, 3, 3, 3)
        values[0, 0, 0, 0] = -9999.0
        cube[:] = values

    with netcdf4.Dataset(lut, "w") as dataset:
        tomography = dataset.createGroup("tomography")
        tomography.createDimension("tomoRelativeAzimuthTime", 4)
        tomography.createDimension("tomoSlantRangeTime", 3)
        azimuth = tomography.createVariable(
            "tomoRelativeAzimuthTime", "f8", ("tomoRelativeAzimuthTime",)
        )
        slant_range = tomography.createVariable(
            "tomoSlantRangeTime", "f8", ("tomoSlantRangeTime",)
        )
        azimuth[:] = [0.0, 0.1, 0.2, 0.3]
        slant_range[:] = [0.01, 0.02, 0.03]
        quality = tomography.createVariable(
            "estimationQualityHV",
            "f4",
            ("tomoRelativeAzimuthTime", "tomoSlantRangeTime"),
        )
        quality[:] = np.ones((4, 3), dtype=np.float32)
        if level == "L1D":
            geometry = dataset.createGroup("geometry")
            geometry.createDimension("relativeAzimuthTime", 2)
            geometry.createDimension("slantRangeTime", 2)
            latitude = geometry.createVariable(
                "latitude", "f8", ("relativeAzimuthTime", "slantRangeTime")
            )
            latitude[:] = np.array([[5.0, 5.1], [5.2, 5.3]])

    geographic = "" if level == "L1D" else """
      <firstLatitude>-1.0</firstLatitude>
      <firstLongitude>5.0</firstLongitude>
      <rowStep>0.5</rowStep>
      <columnStep>0.5</columnStep>
    """
    annotation.write_text(
        f"""<annotation>
      <geodeticCoordinateReferenceSystem><code>EPSG:4979</code></geodeticCoordinateReferenceSystem>
      <tomographicCubesPolarizations>
        <polarization>HH</polarization><polarization>VV</polarization><polarization>HV</polarization>
      </tomographicCubesPolarizations>
      {geographic}
      <minVerticalAxis>-2.0</minVerticalAxis>
      <numberOfVerticalSamples>3</numberOfVerticalSamples>
      <verticalAxisStep>2.0</verticalAxisStep>
    </annotation>""",
        encoding="utf-8",
    )
    stac = {
        "id": root.name,
        "bbox": [-1.0, 5.0, 0.5, 6.0],
        "properties": {
            "processing:level": level,
            "proj:shape": [4, 3, 3, 3],
            "start_datetime": "2025-01-01T00:00:00Z",
            "end_datetime": "2025-01-02T00:00:00Z",
        },
        "assets": {
            "measurement": {
                "href": "./measurement/cube.nc",
                "roles": ["measurement", "data"],
            },
            "annotation": {
                "href": "./annotation/annotation.xml",
                "roles": ["annotation", "metadata"],
            },
            "lut": {
                "href": "./annotation/lut.nc",
                "roles": ["annotation", "lut"],
            },
        },
    }
    (root / "product_stac.json").write_text(json.dumps(stac), encoding="utf-8")
    return root


@pytest.mark.parametrize("level", ["L1D", "L2A"])
def test_inspect_detects_both_product_levels(tmp_path: Path, level: str) -> None:
    root = make_product(tmp_path, level)

    product = TomogramProvider().inspect(root)

    assert product.product_level == level
    assert product.shape == (4, 3, 3, 3)
    assert product.polarizations == ("HH", "VV", "HV")
    assert product.vertical_minimum == -2.0
    assert product.vertical_maximum == 2.0


def test_open_l2a_normalizes_and_subsets_geographic_volume(tmp_path: Path) -> None:
    root = make_product(tmp_path, "L2A")

    dataset = TomogramProvider().open(
        root,
        polarization="HV",
        height_range=(0.0, 2.0),
        bounds=(5.0, -0.5, 5.5, 0.5),
    )

    assert dataset.tomographic_cube.dims == ("polarization", "z", "y", "x")
    assert dataset.sizes == {"polarization": 1, "z": 2, "y": 3, "x": 2}
    np.testing.assert_array_equal(dataset.x, [5.0, 5.5])
    np.testing.assert_array_equal(dataset.y, [-0.5, 0.0, 0.5])
    assert dataset.attrs["coordinate_space"] == "geographic"
    assert "latitude-first" in dataset.attrs["coordinate_normalization"]
    dataset.close()


def test_open_l1d_retains_native_sar_coordinates(tmp_path: Path) -> None:
    root = make_product(tmp_path, "L1D")

    dataset = open_tomogram(root, polarization=["HH", "HV"])

    assert dataset.tomographic_cube.dims == (
        "polarization",
        "z",
        "azimuth_time",
        "slant_range_time",
    )
    assert dataset.sizes == {
        "polarization": 2,
        "z": 3,
        "azimuth_time": 4,
        "slant_range_time": 3,
    }
    np.testing.assert_allclose(dataset.azimuth_time, [0.0, 0.1, 0.2, 0.3])
    assert dataset.attrs["coordinate_space"] == "sar"
    dataset.close()


def test_l1d_rejects_geographic_bounds(tmp_path: Path) -> None:
    root = make_product(tmp_path, "L1D")

    with pytest.raises(ValueError, match="only for L2A"):
        TomogramProvider().open(root, bounds=(-1.0, 5.0, 0.0, 6.0))


def test_expected_level_and_polarization_are_validated(tmp_path: Path) -> None:
    root = make_product(tmp_path, "L2A")
    provider = TomogramProvider()

    with pytest.raises(ValueError, match="Expected L1D"):
        provider.open(root, expected_level="L1D")
    with pytest.raises(ValueError, match="Unknown polarization"):
        provider.open(root, polarization="VH")


def test_measurement_path_discovers_its_product(tmp_path: Path) -> None:
    root = make_product(tmp_path, "L2A")

    product = TomogramProvider().inspect(root / "measurement" / "cube.nc")

    assert product.root == root
