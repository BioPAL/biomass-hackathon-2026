"""BIOMASS L2A forest-height products: catalogue search, authenticated crop download, GeoTIFF I/O.

Credentials are asked with getpass (or read from environment variables) and kept in memory only.
"""

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping


def find_scenes(bbox, datetime, catalog_url, collection="BiomassLevel2a") -> pd.DataFrame:
    """All BIOMASS L2A Forest Height products intersecting ``bbox`` (one row each)."""
    from pystac_client import Client

    cat = Client.open(catalog_url)
    rows = []
    for it in cat.search(collections=[collection], bbox=bbox, datetime=datetime, max_items=300, method="GET").items():
        if not it.id.startswith("BIO_FP_FH__L2A"):
            continue
        tf = re.search(r"_T(\d+)_F(\d+)_", it.id)
        rows.append(
            dict(
                id=it.id,
                date=pd.Timestamp(it.properties.get("start_datetime") or it.datetime).tz_localize(None).normalize(),
                track=int(tf.group(1)),
                frame=int(tf.group(2)),
                orbit=it.properties.get("sat:orbit_state", "?"),
                look=it.properties.get("sar:observation_direction", "left"),
                footprint=it.geometry["coordinates"][0],
                height_href=it.assets["enclosure_i_fh_tiff"].get_absolute_href(),
                quality_href=it.assets["enclosure_i_quality_tiff"].get_absolute_href(),
            )
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()


def scene_files(site_dir: Path, scene_id: str):
    d = Path(site_dir) / "scenes"
    return d / f"{scene_id}_height.tif", d / f"{scene_id}_quality.tif", d / f"{scene_id}.json"


def missing_scenes(scenes: pd.DataFrame, site_dir: Path) -> pd.DataFrame:
    if scenes.empty:
        return scenes
    return scenes[[not scene_files(site_dir, s)[2].exists() for s in scenes.id]]


def download_scenes(scenes: pd.DataFrame, bbox, site_dir: Path, log=print):
    """Crop height + quality of every missing scene to ``bbox``. Needs ESA_CLIENT_SECRET in the environment
    (typed once with getpass, kept only in memory). Scenes without valid pixels are recorded as empty."""
    todo = missing_scenes(scenes, site_dir)
    (Path(site_dir) / "scenes").mkdir(parents=True, exist_ok=True)
    for _, s in todo.iterrows():
        hp, qp, mp = scene_files(site_dir, s.id)
        meta = dict(
            id=s.id,
            date=s.date.isoformat(),
            track=int(s.track),
            frame=int(s.frame),
            orbit=str(s.orbit),
            look=str(s.look),
            footprint=s.footprint,
        )
        try:
            token = get_esa_access_token()
            h, tr, crs = crop_remote_biomass_asset(s.height_href, bbox, token)
            q, qtr, qcrs = crop_remote_biomass_asset(s.quality_href, bbox, token)
            del token
            h = np.where(h < 0, np.nan, h)
            n = int(np.isfinite(h).sum())
            meta["valid_pixels"] = n
            if n:
                save_geotiff(hp, h, tr, crs)
                save_geotiff(qp, q, qtr, qcrs)
            log(f"  {s.date.date()} T{s.track:03d} F{s.frame} {s.orbit[:4]}: {n:,} valid pixels")
        except Exception as err:  # keep going: one failing scene must not stop the others
            meta["error"] = f"{type(err).__name__}: {str(err)[:200]}"
            log(f"  {s.id}: FAILED ({meta['error']})")
        Path(mp).write_text(json.dumps(meta, indent=1))
        time.sleep(0.5)


def get_esa_access_token() -> str:
    """Exchange an MAAP offline token for a short-lived token without printing secrets.

    Returns
    -------
    str
        Current Bearer token. Keep in kernel memory only, never commit credentials.
    """
    import os
    from getpass import getpass

    import requests

    credentials = {}
    cfg = Path.home() / "credentials.txt"
    if cfg.exists():
        for line in cfg.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                credentials[key.strip()] = value.strip()
    token_file = Path.home() / "token.txt"
    if token_file.exists() and "OFFLINE_TOKEN" not in credentials:
        credentials["OFFLINE_TOKEN"] = token_file.read_text().strip()
    client_id = credentials.get("CLIENT_ID") or os.environ.get("ESA_CLIENT_ID", "offline-token")
    secret = credentials.get("CLIENT_SECRET") or os.environ.get("ESA_CLIENT_SECRET")
    offline = credentials.get("OFFLINE_TOKEN") or os.environ.get("ESA_OFFLINE_TOKEN")
    if not secret:
        secret = getpass("ESA MAAP CLIENT_SECRET (input hidden): ")
    if not offline:
        offline = getpass("ESA MAAP OFFLINE_TOKEN (input hidden): ")
    url = "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token"
    response = requests.post(
        url,
        data={
            "client_id": client_id,
            "client_secret": secret,
            "grant_type": "refresh_token",
            "refresh_token": offline,
            "scope": "offline_access openid",
        },
        timeout=45,
    )
    response.raise_for_status()
    access_token = response.json().get("access_token")
    if not access_token:
        raise RuntimeError("ESA MAAP did not return an access token.")
    return access_token


def crop_remote_biomass_asset(url: str, bbox: list[float], token: str) -> tuple:
    """Read only a Lopé geographic subset of a remote ESA GeoTIFF.

    Parameters
    ----------
    url : str
        STAC asset URL, HTTPS.
    bbox : list[float]
        Bounding box [west,south,east,north] in degrees.
    token : str
        Short-lived ESA access token (never write to disk).

    Returns
    -------
    tuple
        Crop values, affine transform, and CRS. NoData becomes NaN.
    """
    with rasterio.Env(
        GDAL_HTTP_HEADERS=f"Authorization: Bearer {token}",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    ):
        with rasterio.open(url) as src:
            if src.crs is None:
                raise ValueError("Remote GeoTIFF has no coordinate reference system.")
            roi = transform_geom("EPSG:4326", src.crs.to_string(), mapping(box(*bbox)))
            arr, tr = mask(src, [roi], crop=True, filled=False)
            return np.ma.filled(arr[0].astype("float64"), np.nan), tr, src.crs


def read_raster(path: str | Path) -> tuple[np.ndarray, Any, Any]:
    """Read a georeferenced single-band height raster with NaN for NoData.

    Parameters
    ----------
    path : str or Path
        Source GeoTIFF path (local).

    Returns
    -------
    tuple
        Float raster [m], affine transform, and CRS.
    """
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Missing raster CRS: {path}")
        a = np.ma.filled(src.read(1, masked=True).astype("float64"), np.nan)
        return a, src.transform, src.crs


def save_geotiff(path: str | Path, arr: np.ndarray, transform: Any, crs: Any) -> None:
    """Save a cropped raster with its true geographic grid (not a quicklook).

    Parameters
    ----------
    path : Path or str
        Destination .tif file.
    arr : numpy.ndarray
        Pixel values (height in metres or quality indicator in percent).
    transform : affine.Affine
        Pixel-to-map transformation.
    crs : rasterio.crs.CRS
        Coordinate reference system.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    values = np.where(np.isfinite(arr), arr, -9999).astype("float32")
    with rasterio.open(
        dest,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999,
        compress="deflate",
        predictor=3,
    ) as dst:
        dst.write(values, 1)
