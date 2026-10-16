import numpy as np

from src.process.geometry import kz_height_ratio, local_incidence, look_azimuth_deg


def test_flat_terrain_keeps_incidence():
    assert np.isclose(local_incidence(30.0, 0.0, 123.0, 80.0), 30.0)


def test_slope_facing_radar_reduces_incidence():
    look = 80.0
    assert np.isclose(local_incidence(30.0, 10.0, look + 180.0, look), 20.0)
    assert np.isclose(local_incidence(30.0, 10.0, look, look), 40.0)


def test_kz_ratio_is_one_on_flat_terrain():
    assert np.isclose(kz_height_ratio(30.0, 30.0), 1.0)
    assert kz_height_ratio(30.0, 15.0) > 1.0


def test_left_and_right_look_are_opposite():
    d = (look_azimuth_deg(10.0, "left") - look_azimuth_deg(10.0, "right")) % 360
    assert np.isclose(d, 180.0)
