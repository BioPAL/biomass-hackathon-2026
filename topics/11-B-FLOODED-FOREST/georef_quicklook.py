"""Georeference Biomass L1C preview PNGs using enclosure_map_kml LatLonQuad corners."""

from __future__ import annotations

import re
import urllib.request
from io import BytesIO

import numpy as np
from PIL import Image
from shapely.geometry import Polygon
# KML gx:LatLonQuad: bottom-left, bottom-right, top-right, top-left (lon, lat).
_KML_QUAD_RE = re.compile(
    r"<gx:LatLonQuad>.*?<coordinates>\s*([^<]+?)\s*</coordinates>",
    re.DOTALL,
)

# S1 vs S3 (and catalogue versions) use different STAC asset keys for the same KML.
KML_ASSET_KEY_PRIORITY = (
    "enclosure_map_kml",
    "enclosure_kml",
    "enclosure_coh_map_kml",
    "enclosure_itf_map_kml",
)


def kml_asset_url(product: dict) -> str:
    assets = product.get("assets") or {}
    for key in KML_ASSET_KEY_PRIORITY:
        url = assets.get(key)
        if url:
            return url
    for key, url in assets.items():
        if "kml" in key.lower() and url:
            return url
    raise KeyError(
        f"product {product.get('id')} has no KML asset "
        f"(tried {', '.join(KML_ASSET_KEY_PRIORITY)})"
    )


def parse_kml_latlon_quad(kml_text: str) -> list[tuple[float, float]]:
    match = _KML_QUAD_RE.search(kml_text)
    if not match:
        north = re.search(r"<north>([^<]+)</north>", kml_text)
        south = re.search(r"<south>([^<]+)</south>", kml_text)
        east = re.search(r"<east>([^<]+)</east>", kml_text)
        west = re.search(r"<west>([^<]+)</west>", kml_text)
        if north and south and east and west:
            w = float(west.group(1))
            s = float(south.group(1))
            e = float(east.group(1))
            n = float(north.group(1))
            return [(w, s), (e, s), (e, n), (w, n)]
        raise ValueError("KML has no gx:LatLonQuad or LatLonBox coordinates")
    points: list[tuple[float, float]] = []
    for token in match.group(1).strip().split():
        lon, lat, *_ = token.split(",")
        points.append((float(lon), float(lat)))
    if len(points) != 4:
        raise ValueError(f"expected 4 quad corners, got {len(points)}")
    return points


def fetch_text(url: str, access_token: str, timeout: int = 120) -> str:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_image_array(url: str, access_token: str) -> np.ndarray:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
    with urllib.request.urlopen(req, timeout=120) as response:
        img = Image.open(BytesIO(response.read()))
    return np.asarray(img)


def quad_footprint_polygon(quad: list[tuple[float, float]]) -> Polygon:
    return Polygon(quad)


def common_footprint_bounds(
    quads: list[list[tuple[float, float]]],
) -> tuple[float, float, float, float]:
    """Intersection of all quad polygons; returns (west, south, east, north)."""
    if not quads:
        raise ValueError("no quads")
    common = quad_footprint_polygon(quads[0])
    for quad in quads[1:]:
        common = common.intersection(quad_footprint_polygon(quad))
    if common.is_empty:
        raise ValueError("no common overlap across scene footprints")
    return common.bounds


def preview_row_mapping_flipped(product: dict | None) -> bool:
    """Kept for older notebook cells. Orientation comes from the KML quad, not UTC hour."""
    return False


def lonlat_to_pixel(lon, lat, quad: list[tuple[float, float]], width: int, height: int):
    """
    Inverse of KML gx:LatLonQuad (image BL, BR, TR, TL).

    Morning passes list the northern edge as the image bottom; that is already
    in the quad, so do not flip the PNG by acquisition time.
    """
    bl, br, tr, tl = (np.asarray(p, dtype=np.float64) for p in quad)
    u = np.full(np.shape(lon), 0.5, dtype=np.float64)
    v = np.full(np.shape(lon), 0.5, dtype=np.float64)
    target = np.stack([np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)], axis=-1)

    for _ in range(12):
        p = (
            ((1 - u) * (1 - v))[..., None] * bl
            + (u * (1 - v))[..., None] * br
            + (u * v)[..., None] * tr
            + ((1 - u) * v)[..., None] * tl
        )
        du = (1 - v)[..., None] * (br - bl) + v[..., None] * (tr - tl)
        dv = (1 - u)[..., None] * (tl - bl) + u[..., None] * (tr - br)
        residual = target - p
        det = du[..., 0] * dv[..., 1] - du[..., 1] * dv[..., 0]
        det = np.where(np.abs(det) < 1e-14, np.nan, det)
        u = u + (residual[..., 0] * dv[..., 1] - residual[..., 1] * dv[..., 0]) / det
        v = v + (du[..., 0] * residual[..., 1] - du[..., 1] * residual[..., 0]) / det

    col = u * (width - 1)
    row = (1.0 - v) * (height - 1)
    return row, col


def quad_bounds(quad: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    lons = [p[0] for p in quad]
    lats = [p[1] for p in quad]
    return min(lons), min(lats), max(lons), max(lats)


def warp_quicklook_to_view(
    img: np.ndarray,
    quad: list[tuple[float, float]],
    view: tuple[float, float, float, float],
    out_shape: tuple[int, int] = (480, 480),
    product: dict | None = None,
) -> np.ndarray:
    """
    Resample preview to a north-up lon/lat grid over view=(west, south, east, north).
    """
    from scipy.ndimage import map_coordinates

    west, south, east, north = view
    out_h, out_w = out_shape
    lons = np.linspace(west, east, out_w, dtype=np.float64)
    lats = np.linspace(north, south, out_h, dtype=np.float64)
    lon_g, lat_g = np.meshgrid(lons, lats)

    h, w = img.shape[0], img.shape[1]
    row, col = lonlat_to_pixel(lon_g, lat_g, quad, w, h)

    if img.ndim == 2:
        return map_coordinates(img, [row, col], order=1, mode="constant", cval=0)

    bands = [
        map_coordinates(img[..., i], [row, col], order=1, mode="constant", cval=0)
        for i in range(img.shape[2])
    ]
    return np.stack(bands, axis=-1)


def load_product_quad(product: dict, access_token: str, cache: dict) -> list[tuple[float, float]]:
    pid = product["id"]
    if pid in cache:
        return cache[pid]
    kml_url = kml_asset_url(product)
    quad = parse_kml_latlon_quad(fetch_text(kml_url, access_token))
    cache[pid] = quad
    return quad
