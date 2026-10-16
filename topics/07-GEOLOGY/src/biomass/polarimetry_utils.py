"""
biomass_utils.py
Helper functions for BIOMASS SLC processing and Freeman decomposition.
"""

import os
import glob
import numpy as np
import xarray as xr
import rioxarray as riox
from dask.diagnostics import ProgressBar


def find_measurement_folders(tds: str, data_root: str = ".") -> list[str]:
    """Find all BIOMASS measurement subfolders inside data_{TDS}."""
    data_dir = os.path.join(data_root, f"../data/{tds}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    pattern = os.path.join(data_dir, "BIO_S1_SCS__1S_*")
    folders = sorted(glob.glob(pattern))
    if not folders:
        raise FileNotFoundError(f"No folders matching 'BIO_S1_SCS__1S_*' in {data_dir}")
    return folders


def find_tiff_files(folder: str) -> dict[str, str]:
    """Identify amplitude and phase TIFFs inside a measurement folder."""
    tiffs = sorted(
        glob.glob(os.path.join(folder, "*.tif"))
        + glob.glob(os.path.join(folder, "*.tiff"))
    )
    if not tiffs:
        raise FileNotFoundError(f"No .tif/.tiff files in {folder}")

    abs_path = phase_path = None
    for t in tiffs:
        n = os.path.basename(t).lower()
        if any(k in n for k in ("amp", "abs", "magnitude", "amplitude")):
            abs_path = t
        elif any(k in n for k in ("phase", "ph")):
            phase_path = t

    if abs_path is None:
        abs_path = tiffs[0]
    if phase_path is None:
        phase_path = tiffs[-1] if len(tiffs) > 1 else tiffs[0]

    return {"abs": abs_path, "phase": phase_path}


def reconstruct_slc(abs_tiff: str, phase_tiff: str) -> xr.Dataset:
    """Load amp/phase TIFFs → complex SLC → PolSARpro S-matrix Dataset."""
    da_abs = riox.open_rasterio(abs_tiff, chunks={})
    da_phase = riox.open_rasterio(phase_tiff, chunks={})

    slc = da_abs * np.exp(1j * da_phase)
    dims = ("y", "x")

    return xr.Dataset(
        {
            "hh": (dims, slc[0].data),
            "hv": (dims, slc[1].data),
            "vh": (dims, slc[2].data),
            "vv": (dims, slc[3].data),
        },
        coords={"y": np.arange(slc.shape[1]), "x": np.arange(slc.shape[2])},
        attrs={"poltype": "S", "description": "Scattering matrix"},
    ).chunk("auto")


def freeman_decompose(S: xr.Dataset, out_file: str, snr: list[int] = [7, 7]) -> xr.Dataset:
    """Run Freeman decomposition and save to NetCDF."""
    from polsarpro.decompositions import freeman

    if os.path.exists(out_file):
        os.remove(out_file)

    with ProgressBar():
        freeman(S, snr).to_netcdf(out_file)

    return xr.open_dataset(out_file)
