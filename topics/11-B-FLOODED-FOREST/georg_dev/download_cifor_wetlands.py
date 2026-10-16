"""Clip CIFOR Global Wetlands V3 to a lon/lat bbox.

The authoritative GeoTIFF is
https://doi.org/10.17528/CIFOR/DATA.00058
(file TROP-SUBTROP_WetlandV3b_2016_CIFOR.7z). data.cifor.org currently
drops the TLS handshake, so this uses the same public tiled map the
viewer at https://www2.cifor.org/global-wetlands/ draws
(TROP_SUBTROP_WetlandV3_2016_CIFOR3b).

Tiles are a Web Mercator cache of the classified map. Palette colours are
matched to the service legend and written as the published class codes
(10, 20, ... 100). Cache resolution at level 10 is ~153 m; the source
raster is 231 m. Resampling is nearest neighbour.
"""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_origin, from_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds as window_from_bounds

# Belém test extent (west, south, east, north), EPSG:4326
BBOX_WGS84 = (
    -49.1513175964354971,
    -2.2793788909912100,
    -48.7919921875000000,
    -1.7499319314956701,
)
TILE_LEVEL = 10  # ~152.9 m; native product is 231 m

SERVICE = (
    "https://tiles.arcgis.com/tiles/P8Cok4qAP1sTVE59/arcgis/rest/services/"
    "TROP_SUBTROP_WetlandV3_2016_CIFOR3b/MapServer"
)
# Half-circumference used by this ArcGIS cache (matches LOD resolutions).
ORIGIN_X = -20037508.342787
ORIGIN_Y = 20037508.342787
HALF_CIRC = 20037508.342787
LEVEL0_RES = (2 * HALF_CIRC) / 256.0  # 156543.033928 m

# Published pixel codes (Gumbricht et al. 2017 / CIFOR map text).
CLASS_CODE = {
    "Open Waters": 10,
    "Mangroves": 20,
    "Swamps": 30,
    "Fens": 40,
    "Riverine and lacustrine": 50,
    "Floodouts": 60,
    "Floodplains": 70,
    "General Marshes": 80,
    "Marshes in arid climate": 90,
    "Marshes wet meadows": 100,
}

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
OUT_TIF = HERE / "cifor_wetland_v3_belem_test.tif"
TILE_CACHE = TOPIC / "cifor_cache"


def lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    x = lon / 180.0 * HALF_CIRC
    y = math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)) * HALF_CIRC / math.pi
    return x, y


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "biomass-hackathon-cifor/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def legend_rgb() -> dict[str, tuple[int, int, int]]:
    payload = json.loads(fetch(f"{SERVICE}/legend?f=json"))
    colours: dict[str, tuple[int, int, int]] = {}
    for item in payload["layers"][0]["legend"]:
        label = item["label"]
        if label not in CLASS_CODE:
            raise RuntimeError(f"unexpected legend class: {label}")
        raw = __import__("base64").b64decode(item["imageData"])
        with MemoryFile(raw) as mem, mem.open() as src:
            rgba = src.read()
        # Swatch PNGs are slightly filtered; take the most common opaque RGB.
        rgb = rgba[:3].reshape(3, -1).T
        alpha = rgba[3].ravel()
        opaque = rgb[alpha > 200]
        uniq, counts = np.unique(opaque, axis=0, return_counts=True)
        colours[label] = tuple(int(v) for v in uniq[np.argmax(counts)])
    return colours


def nearest_class(rgb: tuple[int, int, int], legend: dict[str, tuple[int, int, int]]) -> int:
    best_code = 0
    best_d = 1e9
    for label, ref in legend.items():
        d = sum((a - b) ** 2 for a, b in zip(rgb, ref))
        if d < best_d:
            best_d = d
            best_code = CLASS_CODE[label]
    # Palette entries are the legend colours (off by a few DN). Reject anything else.
    if best_d > 40**2:
        return 0
    return best_code


def download_clip(bbox=BBOX_WGS84, level: int = TILE_LEVEL, out_path: Path = OUT_TIF) -> Path:
    west, south, east, north = bbox
    if not (west < east and south < north):
        raise ValueError(f"bbox is not west-south-east-north: {bbox}")

    x0, y0 = lonlat_to_mercator(west, south)
    x1, y1 = lonlat_to_mercator(east, north)
    res = LEVEL0_RES / (2**level)
    c0 = math.floor((min(x0, x1) - ORIGIN_X) / (res * 256))
    c1 = math.floor((max(x0, x1) - ORIGIN_X) / (res * 256))
    r0 = math.floor((ORIGIN_Y - max(y0, y1)) / (res * 256))
    r1 = math.floor((ORIGIN_Y - min(y0, y1)) / (res * 256))

    legend = legend_rgb()
    width = (c1 - c0 + 1) * 256
    height = (r1 - r0 + 1) * 256
    mosaic = np.zeros((height, width), dtype=np.uint16)
    TILE_CACHE.mkdir(parents=True, exist_ok=True)

    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            url = f"{SERVICE}/tile/{level}/{row}/{col}"
            cache = TILE_CACHE / f"wetland_v3_l{level}_{row}_{col}.png"
            data = cache.read_bytes() if cache.exists() else fetch(url)
            if not cache.exists():
                cache.write_bytes(data)
            with MemoryFile(data) as mem, mem.open() as src:
                idx = src.read(1)
                cmap = src.colormap(1)
            classes = np.zeros(idx.shape, dtype=np.uint16)
            for value in np.unique(idx):
                r, g, b, a = cmap[int(value)]
                if a < 128:
                    continue
                classes[idx == value] = nearest_class((r, g, b), legend)
            i0 = (row - r0) * 256
            j0 = (col - c0) * 256
            mosaic[i0 : i0 + 256, j0 : j0 + 256] = classes

    left = ORIGIN_X + c0 * res * 256
    top = ORIGIN_Y - r0 * res * 256
    transform_3857 = from_origin(left, top, res, res)
    crs_3857 = CRS.from_epsg(3857)

    # Crop to the requested extent in Web Mercator before the geographic warp.
    win = window_from_bounds(x0, y0, x1, y1, transform_3857)
    win = win.round_offsets().round_lengths()
    row_off, col_off = int(win.row_off), int(win.col_off)
    h, w = int(win.height), int(win.width)
    cropped = mosaic[row_off : row_off + h, col_off : col_off + w]
    cropped_transform = rasterio.windows.transform(win, transform_3857)

    dst_crs = CRS.from_epsg(4326)
    # Keep roughly the cache pixel size in degrees at the bbox centre.
    lat_c = (south + north) / 2.0
    res_deg = res / (111_320.0 * math.cos(math.radians(lat_c)))
    dst_w = max(1, int(round((east - west) / res_deg)))
    dst_h = max(1, int(round((north - south) / res_deg)))
    dst = np.zeros((dst_h, dst_w), dtype=np.uint16)
    dst_transform = from_bounds(west, south, east, north, dst_w, dst_h)
    reproject(
        source=cropped,
        destination=dst,
        src_transform=cropped_transform,
        src_crs=crs_3857,
        src_nodata=0,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": dst_h,
        "width": dst_w,
        "count": 1,
        "dtype": "uint16",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(out_path, "w", **profile) as dst_ds:
        dst_ds.write(dst, 1)
        dst_ds.set_band_description(1, "CIFOR wetland class code")
        dst_ds.update_tags(
            source=SERVICE,
            citation="Gumbricht et al. 2017 Global Change Biology DOI:10.1111/gcb.13689",
            dataset="TROP-SUBTROP Wetland V3b 2016 CIFOR",
            tile_level=str(level),
            cache_resolution_m=f"{res:.3f}",
            note="Class codes from tiled map legend; not the native 231 m GeoTIFF.",
        )

    codes, counts = np.unique(dst, return_counts=True)
    summary = {int(c): int(n) for c, n in zip(codes, counts)}
    print(f"wrote {out_path}")
    print(f"crs EPSG:4326  size {dst_w} x {dst_h}  pixel ~{res_deg:.6f} deg (~{res:.1f} m)")
    print(f"bounds {west:.6f},{south:.6f} : {east:.6f},{north:.6f}")
    print("class counts", summary)
    return out_path


if __name__ == "__main__":
    download_clip()
