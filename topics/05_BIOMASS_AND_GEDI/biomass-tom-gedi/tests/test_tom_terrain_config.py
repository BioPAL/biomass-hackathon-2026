import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin

from biomass_tom_gedi.config import Config
from biomass_tom_gedi.terrain import DTMSlope, M_PER_DEG
from biomass_tom_gedi.tom import TOMGrid, extract_profiles

ANNOT = """<root>
<firstLatitude>-0.5</firstLatitude><firstLongitude>11.0</firstLongitude>
<rowStep>0.001</rowStep><columnStep>0.002</columnStep>
<numberOfRows>40</numberOfRows><numberOfColumns>30</numberOfColumns>
<minVerticalAxis>-60</minVerticalAxis><verticalAxisStep>2</verticalAxisStep><numberOfVerticalSamples>5</numberOfVerticalSamples>
<startTime>2025-10-07T05:03:00</startTime>
<tomographicCubesPolarizations><polarization>HH</polarization><polarization>VV</polarization></tomographicCubesPolarizations>
</root>"""


def _grid(tmp_path):
    p = tmp_path / "a_annot.xml"
    p.write_text(ANNOT)
    return TOMGrid.from_annotation(p)


def test_grid_axes_and_row_col(tmp_path):
    g = _grid(tmp_path)
    assert g.pols == ["HH", "VV"] and g.nz == 5
    assert np.isclose(g.lat[0], -0.5) and np.isclose(g.lat[-1], -0.5 + 39 * 0.001)   # row 0 = south
    r, c = g.row_col([-0.5004, 10.0, -9.0], [11.0041, 11.0, 12.0])
    assert r.tolist() == [0, 39, 0] and c.tolist() == [2, 0, 29]                     # nearest pixel, clipped to the scene
    assert g.height_index(-56) == 2


def test_extract_profiles_window_mean_and_fill(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.random((40, 30, 5, 2)).astype("float32")
    data[0:3, 0:3] = -9999                                                             # empty pixels -> NaN
    f = tmp_path / "cube.h5"
    with h5py.File(f, "w") as h:
        h.create_dataset("tomographicCubes", data=data, chunks=(16, 16, 5, 2))
    with h5py.File(f, "r") as h:
        cube = h["tomographicCubes"]
        prof = extract_profiles(cube, np.array([20, 1, 39]), np.array([10, 1, 29]), half=1)
    assert np.allclose(prof[0], data[19:22, 9:12].mean(axis=(0, 1)), atol=1e-6)
    assert np.isnan(prof[1]).all()
    assert np.isfinite(prof[2]).all()                                                  # window clipped at the scene edge


def test_slope_of_a_plane(tmp_path):
    """East-rising ramp of 1 m per 30 m -> slope = atan(1/30)."""
    dlon = dlat = 0.0003
    n = 60
    xs_m = np.arange(n) * dlon * M_PER_DEG
    dtm = np.tile(xs_m / 30.0, (n, 1)).astype("float32")
    tif = tmp_path / "x_height.tiff"
    with rasterio.open(tif, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(11.0, 0.0, dlon, dlat)) as dst:
        dst.write(dtm, 1)
    d = DTMSlope(tmp_path, smooth_px=2)
    assert np.isclose(np.nanmedian(d.slope[10:50, 10:50]), np.degrees(np.arctan(1 / 30)), rtol=0.02)


def test_config_expands_env_and_user(tmp_path, monkeypatch):
    monkeypatch.setenv("TOMGEDI_TEST_DIR", str(tmp_path))
    y = tmp_path / "c.yaml"
    y.write_text("paths:\n  data_dir: ${TOMGEDI_TEST_DIR}/d\n  results_dir: ~/res\n")
    cfg = Config.load(y)
    assert cfg.interim_dir == tmp_path / "d" / "interim" and cfg.interim_dir.is_dir()
    assert not str(cfg.get("paths", "results_dir")).startswith("~")
    assert cfg.get("nope", "x", default=7) == 7
