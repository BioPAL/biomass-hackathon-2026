import numpy as np
import pandas as pd
import pytest

from biomass_tom_gedi.dataset import Dataset

TARGET_COLS = ["rh_25", "rh_50", "rh_75", "rh_90", "rh_98", "rh_100", "cover", "agbd", "fhd_normal", "pai",
               "pai_z_2", "pai_z_6", "cover_z_6"] + [f"pavd_z_{i}" for i in range(12)]


@pytest.fixture(scope="session")
def synthetic_dataset():
    """3000 fake shots on a 110x66-pixel patch (60 cells, 10 spatial blocks); targets depend on the profile strength."""
    rng = np.random.default_rng(0)
    n, nz = 3000, 81
    z = -60.0 + 2.0 * np.arange(nz)
    rows = rng.integers(0, 110, n)
    cols = rng.integers(0, 66, n)
    lat = -0.66 + rows * 0.0004
    lon = 11.0 + cols * 0.0004 + (rows // 11) * 0.1          # block = f(cell row): 10 blocks
    strength = rng.uniform(0.5, 2.0, n)
    P = np.empty((n, nz, 3))
    for k in range(3):
        P[:, :, k] = (0.01 + strength[:, None] * np.exp(-((z[None, :] - 20) / 25) ** 2)) * rng.lognormal(0, 0.05, (n, nz))
    shots = pd.DataFrame({"latitude": lat, "longitude": lon, "row": rows, "col": cols, "slope_ok": True})
    for c in TARGET_COLS:
        shots[c] = 10 * strength + rng.normal(0, 0.5, n)
    return Dataset(shots, P, z, ["HH", "HV", "VV"], z_ground=4.0, npx=11, min_cell_shots=5)
