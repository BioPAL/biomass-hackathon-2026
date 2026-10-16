"""Tests for LVIS Level-2 footprint access."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pandas = pytest.importorskip("pandas")

from tomo_canopy.providers import LVISProvider, open_lvis  # noqa: E402

DATA = Path(__file__).parent / "data" / "lvis" / "tiny_lvis_l2.TXT"


def product_copy(tmp_path: Path) -> Path:
    target = tmp_path / "LVISC2_Gabon2023_0520_R2507_051090.TXT"
    shutil.copyfile(DATA, target)
    return target


def test_inspect_parses_filename_metadata(tmp_path: Path) -> None:
    product = LVISProvider().inspect(product_copy(tmp_path))

    assert product.source_format == "ASCII TXT"
    assert product.campaign == "Gabon2023"
    assert product.acquisition_date == "2023-05-20"
    assert product.release == "R2507"
    assert product.segment == "051090"
    assert product.crs == "EPSG:4326"


def test_open_txt_normalizes_complete_rh_profile(tmp_path: Path) -> None:
    dataset = open_lvis(
        product_copy(tmp_path),
        variables=["rh", "ground_elevation", "sensitivity", "channel_rh"],
    )

    assert dataset.sizes == {"shot": 2, "percentile": 23}
    assert dataset.rh.dims == ("shot", "percentile")
    assert float(dataset.rh.sel(shot=0, percentile=98)) == pytest.approx(48.09)
    np.testing.assert_allclose(dataset.x, dataset.longitude)
    np.testing.assert_allclose(dataset.y, dataset.latitude)
    np.testing.assert_array_equal(dataset.shot_number, [4200716, 4200717])
    assert dataset.attrs["modality"] == "lvis"
    assert dataset.attrs["geometry_type"] == "footprints"
    assert dataset.attrs["crs"] == "EPSG:4326"


def test_txt_bounds_variable_selection_and_limit(tmp_path: Path) -> None:
    dataset = LVISProvider().open(
        product_copy(tmp_path),
        variables=["ground_elevation"],
        bounds=(11.7193, 0.0509, 11.72, 0.051),
        max_shots=1,
    )

    assert dataset.sizes == {"shot": 1}
    assert set(dataset.data_vars) == {"ground_elevation"}
    assert int(dataset.shot_number[0]) == 4200717


def test_rejects_geopackage_source(tmp_path: Path) -> None:
    source = tmp_path / "LVISC2_Gabon2023_0520_R2507_051090.gpkg"
    source.touch()

    with pytest.raises(ValueError, match="Level-2 .TXT"):
        LVISProvider().open(source)


def test_rejects_invalid_bounds_and_limit(tmp_path: Path) -> None:
    source = product_copy(tmp_path)
    provider = LVISProvider()

    with pytest.raises(ValueError, match="minima"):
        provider.open(source, bounds=(2.0, 0.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="positive integer"):
        provider.open(source, max_shots=0)
