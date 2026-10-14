"""Sentinel-1 and Sentinel-2 products: listed in the MAAP catalogue, served by the Copernicus Data Space."""
from pathlib import Path

import pandas as pd

from . import config
from .auth import CopernicusAuth
from .catalog import BiomassClient
from .query import SearchQuery
from .viz import _add_overlay, _finish_map


class SentinelClient(BiomassClient):
    """Discover Sentinel collections and search for products (e.g. ``sentinel-1-grd``, ``sentinel-2-l2a``).

    Searching needs no account. The files do: ``auth`` is a Copernicus Data Space
    account, which is not the MAAP one.
    """

    def __init__(self, auth: CopernicusAuth = None, catalog_url: str = config.CATALOG_URL):
        super().__init__(auth or CopernicusAuth(), catalog_url)

    def collections(self) -> pd.DataFrame:
        res = self.stac.collection_search(q="Sentinel")
        cols = pd.json_normalize(res.collection_list_as_dict(), record_path=["collections"])
        cols = cols[cols["id"].str.match(r"sentinel-[12]-")]
        return cols.sort_values("id", ignore_index=True)[["id", "title", "extent.temporal.interval"]]

    def search(self, query: SearchQuery) -> list:
        items = super().search(query)
        for item in items:
            item.assets = _https_assets(item.assets)
        return items


def _https_assets(assets: dict) -> dict:
    """The assets with their https address.

    The catalogue gives most files as ``s3://`` links, with the https one as an
    alternate; the ones without any https address are left out.
    """
    kept = {}
    for key, asset in assets.items():
        if not asset.href.startswith("https://"):
            https = asset.extra_fields.get("alternate", {}).get("https", {}).get("href")
            if not https:
                continue
            asset.href = https
        kept[key] = asset
    return kept


def sentinel_items_table(items) -> pd.DataFrame:
    """Like ``items_table``, with the platform and the polarisations (Sentinel-1) or cloud cover in % (Sentinel-2)."""
    rows = []
    for i, it in enumerate(items):
        p = it.properties
        rows.append({"idx": i, "id": it.id, "start": p.get("start_datetime") or p.get("datetime"),
                     "platform": p.get("platform"), "orbit": p.get("sat:orbit_state"),
                     "track": p.get("sat:relative_orbit"), "type": p.get("product:type"),
                     "pols": ",".join(p.get("sar:polarizations") or []) or None,
                     "cloud": p.get("eo:cloud_cover"), "assets": len(it.assets)})
    return pd.DataFrame(rows).dropna(axis="columns", how="all")


def read_optical(path, max_size: int = 1500):
    """Reduced-resolution read of a local Sentinel-2 image (NaN where there is no data).

    Returns ``(data, transform, crs)`` with data as (bands, rows, cols). Band files
    (B02, B04...) come as surface reflectance, using the scale of current products
    (reflectance = digital number / 10000 - 0.1); the true colour image (TCI) comes as
    its 0-255 colours.
    """
    import numpy as np
    import rasterio as rio
    from rasterio.enums import Resampling

    with rio.open(path) as src:
        scale = max(1, -(-max(src.height, src.width) // max_size))
        h, w = -(-src.height // scale), -(-src.width // scale)
        raw = src.read(out_shape=(src.count, h, w), resampling=Resampling.average)
        transform = src.transform * src.transform.scale(src.width / w, src.height / h)
        crs = src.crs
        is_reflectance = src.count == 1 and src.dtypes[0] == "uint16"
    data = raw.astype("float32")
    data[:, (raw == 0).all(axis=0)] = np.nan
    if is_reflectance:
        data = data / 10000 - 0.1
    return data, transform, crs


def plot_optical(path, max_size: int = 1500):
    """Plot a local Sentinel-2 image on its map grid: colour for the TCI, grey reflectance for a band."""
    import matplotlib.pyplot as plt
    import numpy as np

    data, transform, crs = read_optical(path, max_size)
    west, north = transform * (0, 0)
    east, south = transform * (data.shape[2], data.shape[1])
    extent = [west / 1000, east / 1000, south / 1000, north / 1000]
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    if len(data) == 3:
        ax.imshow(np.nan_to_num(np.dstack(data) / 255), extent=extent, interpolation="nearest")
    else:
        lo, hi = np.nanpercentile(data[0], [2, 98])
        im = ax.imshow(data[0], cmap="gray", vmin=lo, vmax=hi, extent=extent, interpolation="nearest")
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.01, label="surface reflectance")
    ax.set_xlabel(f"x (km, {crs})")
    ax.set_ylabel("y (km)")
    ax.set_title(Path(path).name, fontsize=9)
    return fig


def optical_map(paths, max_size: int = 1000, opacity: float = 0.9):
    """Folium map with local Sentinel-2 images laid over it (TCI in colour, bands in grey)."""
    import folium
    import numpy as np
    from rasterio.warp import calculate_default_transform, reproject

    if isinstance(paths, (str, Path)):
        paths = [paths]
    m = folium.Map()
    bounds = []
    for path in paths:
        src, transform, crs = read_optical(path, max_size)
        src = src[:3] if len(src) >= 3 else src[:1]
        rows, cols = src.shape[1:]
        west, north = transform * (0, 0)
        east, south = transform * (cols, rows)
        dst_transform, w, h = calculate_default_transform(crs, "EPSG:4326", cols, rows, west, south, east, north)
        dst = np.full((len(src), h, w), np.nan, dtype="float32")
        reproject(src, dst, src_transform=transform, src_crs=crs, dst_transform=dst_transform,
                  dst_crs="EPSG:4326", src_nodata=np.nan, dst_nodata=np.nan)
        bounds.append(_add_overlay(m, dst, dst_transform, Path(path).name, opacity))
    return _finish_map(m, bounds)
