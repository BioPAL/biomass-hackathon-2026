"""Shared I/O: constants, authenticated downloads, netCDF lock and the extracted-stack reader."""

import json
import shutil
import threading
from pathlib import Path

import requests

from .scenes import removed_ids

POLARISATIONS = ["HH", "HV", "VH", "VV"]   # band order of the extracted product tifs
NODATA = -9999.0                           # no-data value of the L1c COGs and LUTs

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tiff,.tif",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "3",
    "VSI_CACHE": "FALSE",
}

# netCDF-C/HDF5 is not built thread-safe in these environments: opening several
# netCDF4.Dataset files from parallel threads segfaults intermittently. Every
# netCDF4 access goes through this lock; the downloads around it stay parallel.
NC_LOCK = threading.Lock()


def download(url, token, dest, timeout=600):
    """Stream an authenticated GET to dest (written to dest.part first)."""
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    with requests.get(url, headers=token.headers(), stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with part.open("wb") as out:
            shutil.copyfileobj(response.raw, out, length=8 * 1024 * 1024)
    part.replace(dest)
    return dest


def load_stack_info(stack_dir, kept_only=False):
    """stack_info.json of an extracted stack, keeping only the products on disk, sorted by date.

    kept_only=True also drops the scenes removed in the manual review (core.scenes).
    """
    stack_dir = Path(stack_dir)
    info = json.loads((stack_dir / "stack_info.json").read_text(encoding="utf-8"))
    removed = removed_ids(stack_dir) if kept_only else set()
    info["products"] = sorted(
        (p for p in info["products"]
         if p.get("status") in ("ok", "skipped (exists)") and (stack_dir / p["file"]).exists()
         and p["id"] not in removed),
        key=lambda p: p["datetime"],
    )
    info["_dir"] = stack_dir
    return info
