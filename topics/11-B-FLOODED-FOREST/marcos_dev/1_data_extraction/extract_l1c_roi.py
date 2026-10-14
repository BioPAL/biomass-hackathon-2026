#!/usr/bin/env python3
"""
Extract ROI crops of BIOMASS Level-1C STA stacks from the ESA MAAP catalogue.

Workflow
--------
1. Ask for ROI name, lat/lon bbox, time span, max number of files, orbit
   direction and the minimum ROI coverage of a stack. Every prompt can also be passed as a
   command-line flag, see ``--help``.
2. Search BiomassLevel1c for S1/S2/S3 STA products touching the bbox, group
   them by stack (``eofeos:stack_id``) and complete each stack with all its
   members (the date filter may cut a stack in half).
3. Recommend stacks: one frame per track (when a ROI falls across two
   adjacent frames of the same datatake, the one covering most of the ROI),
   stacks covering less than --min-coverage of the ROI dropped, then whole
   stacks (best coverage first) up to --max-files.
   Then choose the stacks to keep (stack_selection.py): a table of all
   stacks and an interactive map open in the browser; Enter keeps the
   recommendation. --stacks "0,2,5-7" or --yes skip the question.
4. Download ONE LUT per stack (all members of a stack are coregistered and
   share the same radar grid). The stack footprint is traced on the LUT
   latitude/longitude, restricted to pixels holding data (read from a 1/5
   overview of the reference product, so the no-data strips along the swath
   edges are not counted as covered).
5. Common area = ROI ∩ footprint of every selected stack. It is mapped into
   each stack's radar grid to get a pixel window.
6. For every product, only the tiles of ``i_abs`` / ``i_phase`` covering the
   window are fetched with HTTP range requests (the TIFFs are COGs), combined
   into complex HH/HV/VH/VV and written to disk. Products run in parallel
   threads; each band is processed separately to keep RAM low.

Output layout
-------------
<out-dir>/<ROI_name>/
    stack_selection_map.html     map of all stacks found (interactive selection)
    extraction_summary.json      inputs, common area, stacks, per-file status
    common_area.geojson          ROI ∩ all stack footprints (lat/lon)
    extraction.log
    STA1_ascending_20260704-20260722_WRS-6-310/
        BIO_S1_STA__1S_..._C01_...tif   complex64, 4 bands HH HV VH VV,
        ...                             radar geometry (azimuth x range) crop,
                                        GCPs from the LUT for quick display
        overlap_mask.tif           uint8, 1 = inside the common area
        latlon.tif                 float32 lat / lon of every crop pixel
        geometry_lut.nc            LUT geometry (lat, lon, height, incidence)
                                   on its native decimated grid, cropped
        stack_info.json            window offsets, LUT steps, product list

Requires: pystac-client, rasterio, shapely>=2, netCDF4, numpy, scipy, requests


examples:
python extract_l1c_roi.py            # asks for every input
python extract_l1c_roi.py --name "Lac Tele" --bbox 16.98 0.94 17.48 1.44 \
       --start 2026-05-01 --end 2026-10-13 --max-files 70 --orbit both --min-coverage 0.5 --workers 6

       use --dry-run to just see the roi selection zone

"""

import argparse
import json
import logging
import re
import sys
import tempfile
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np
import rasterio
import requests
import shapely
from pystac_client import Client
from rasterio.control import GroundControlPoint
from rasterio.windows import Window
from scipy.ndimage import binary_erosion, map_coordinates
from shapely.geometry import box, mapping, shape

from stack_selection import choose_stacks

# -------------------- Configuration --------------------

STAC_API_URL = "https://catalog.maap.eo.esa.int/catalogue/"
COLLECTION = "BiomassLevel1c"
TOKEN_URL = "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token"
# Public client credentials from the ESA MAAP examples (same for every user)
DEFAULT_CLIENT_ID = "offline-token"
DEFAULT_CLIENT_SECRET = "p1eL7uonXs6MDxtGbgKdPVRAmnGxHpVE"

POLARISATIONS = ["HH", "HV", "VH", "VV"]
NODATA = -9999.0
# Stacks of the same swath/orbit/cycle starting within this time are adjacent
# frames of the same datatake
SAME_TRACK_SECONDS = 300
# Number of GCPs written into each output GeoTIFF (rows x cols)
GCP_GRID = (12, 6)
# COG overview used to find the pixels with data (1 = 1/2, 2 = 1/5, 3 = 1/10)
VALID_OVERVIEW_LEVEL = 1

SCRIPT_DIR = Path(__file__).resolve().parent

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

log = logging.getLogger("extract")
# outputs are in radar geometry: georeferencing is given by GCPs, set after opening
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# -------------------- Authentication --------------------


class TokenManager:
    """Thread-safe MAAP access token, refreshed shortly before it expires."""

    def __init__(self, offline_token, client_id, client_secret):
        self._offline = offline_token
        self._client_id = client_id
        self._client_secret = client_secret
        self._token = None
        self._expires = 0.0
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if self._token is None or time.time() > self._expires - 60:
                data = {
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": self._offline,
                    "scope": "offline_access openid",
                }
                response = requests.post(TOKEN_URL, data=data, timeout=60)
                response.raise_for_status()
                payload = response.json()
                if not payload.get("access_token"):
                    raise RuntimeError("MAAP token response has no access_token")
                self._token = payload["access_token"]
                self._expires = time.time() + float(payload.get("expires_in", 300))
            return self._token


def load_token_manager(path):
    """Accept a raw offline token file or a KEY=value credentials file."""
    text = Path(path).read_text(encoding="utf-8").strip()
    values = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    offline = values.get("OFFLINE_TOKEN") or text.replace("\n", "")
    return TokenManager(
        offline,
        values.get("CLIENT_ID", DEFAULT_CLIENT_ID),
        values.get("CLIENT_SECRET", DEFAULT_CLIENT_SECRET),
    )


def find_token_file(explicit):
    if explicit:
        return Path(explicit)
    for folder in (Path.cwd(), SCRIPT_DIR, SCRIPT_DIR.parent):
        for name in ("credentials.txt", "token.txt"):
            if (folder / name).exists():
                return folder / name
    raise SystemExit("No token.txt / credentials.txt found, pass --token-file")


# -------------------- Inputs --------------------


def ask(prompt, default=None):
    if not sys.stdin.isatty():
        if default is None:
            raise SystemExit(f"Missing input '{prompt}' (pass it as a flag)")
        return default
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or default


def parse_date(text):
    return datetime.strptime(text.strip(), "%Y-%m-%d").date()


def collect_inputs(args):
    name = args.name or ask("ROI name")
    if not name:
        raise SystemExit("ROI name is required")

    bbox = args.bbox
    if bbox is None:
        raw = ask("ROI bbox as  west south east north  (decimal degrees)")
        bbox = [float(v) for v in re.split(r"[,\s]+", raw.strip()) if v]
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise SystemExit(f"Invalid bbox {bbox}: expected west < east and south < north")

    start = parse_date(args.start or ask("Start date (YYYY-MM-DD)"))
    end = parse_date(args.end or ask("End date   (YYYY-MM-DD)", str(datetime.now(timezone.utc).date())))
    if end < start:
        raise SystemExit("End date is before start date")

    if args.max_files is None or args.min_coverage is None:
        print("Max files and min coverage only shape the recommended selection; you choose the stacks on the map.")
    max_files = args.max_files or int(ask("Max number of files (products) to extract", "70"))
    orbit = args.orbit or ask("Orbit direction (both / ascending / descending)", "both").lower()
    if orbit not in ("both", "ascending", "descending"):
        raise SystemExit(f"Invalid orbit direction '{orbit}'")
    min_coverage = args.min_coverage
    if min_coverage is None:
        min_coverage = float(ask("Min fraction of the ROI a stack must cover (0-1)", "0.5"))

    return {
        "name": name,
        "folder": re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_"),
        "bbox": [west, south, east, north],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "max_files": max_files,
        "orbit": orbit,
        "min_coverage": min_coverage,
        "swaths": [s.upper() for s in args.swaths],
    }


# -------------------- Catalogue search --------------------


def item_version(item):
    return (item.properties.get("version") or "", item.properties.get("created") or "")


def search_stacks(catalog, cfg, workers):
    types = " or ".join(f"product:type='{s}_STA__1S'" for s in cfg["swaths"])
    flt = f"({types})"
    if cfg["orbit"] != "both":
        flt += f" and sat:orbit_state='{cfg['orbit']}'"

    search = catalog.search(
        collections=[COLLECTION],
        bbox=cfg["bbox"],
        datetime=f"{cfg['start']}T00:00:00Z/{cfg['end']}T23:59:59Z",
        filter=flt,
        method="GET",
        limit=200,
    )
    hits = list(search.items())
    stack_ids = sorted({it.properties["eofeos:stack_id"] for it in hits})
    log.info("Search: %d products in %d stacks touch the bbox", len(hits), len(stack_ids))

    def complete(stack_id):
        members = catalog.search(
            collections=[COLLECTION],
            filter=f"eofeos:stack_id='{stack_id}'",
            method="GET",
            limit=50,
        )
        # one product per repeat cycle, newest version wins
        by_cycle = {}
        for it in members.items():
            cycle = int(it.properties["eofeos:repeat_cycle_id"])
            if cycle not in by_cycle or item_version(it) > item_version(by_cycle[cycle]):
                by_cycle[cycle] = it
        return stack_id, [by_cycle[c] for c in sorted(by_cycle)]

    roi = box(*cfg["bbox"])
    stacks = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for stack_id, items in pool.map(complete, stack_ids):
            if not items:
                continue
            ref = next((it for it in items if it.properties.get("eofeos:is_coregistration_primary")), items[0])
            props = ref.properties
            times = sorted(datetime.fromisoformat(it.properties["datetime"].replace("Z", "+00:00")) for it in items)
            footprint = shape(ref.geometry)
            stacks.append({
                "stack_id": stack_id,
                "items": items,
                "reference": ref,
                "swath": props["product:type"][:2],
                "orbit": props["sat:orbit_state"],
                "grid": props.get("grid:code", "grid"),
                "major_cycle": props.get("eofeos:major_cycle_id"),
                "start": times[0],
                "end": times[-1],
                "stac_coverage": footprint.intersection(roi).area / roi.area,
                "stac_footprint": footprint,
            })
    for st in stacks:
        st["name"] = stack_folder_name(st)
    return sorted(stacks, key=lambda s: (s["orbit"], s["swath"], s["start"], s["grid"]))


def best_frame_per_track(stacks):
    """Of adjacent frames from the same datatake, keep the one covering most of the ROI."""
    groups = defaultdict(list)
    for st in stacks:
        groups[(st["swath"], st["orbit"], st["major_cycle"])].append(st)

    kept = []
    for members in groups.values():
        members.sort(key=lambda s: s["start"])
        cluster = [members[0]]
        for st in members[1:]:
            if (st["start"] - cluster[0]["start"]).total_seconds() <= SAME_TRACK_SECONDS:
                cluster.append(st)
            else:
                kept.append(max(cluster, key=lambda s: s["stac_coverage"]))
                cluster = [st]
        kept.append(max(cluster, key=lambda s: s["stac_coverage"]))
    return kept


def limit_files(stacks, max_files):
    """Take whole stacks (best ROI coverage first) until max_files is reached."""
    stacks = sorted(stacks, key=lambda s: (-s["stac_coverage"], s["start"]))
    chosen, total = [], 0
    for st in stacks:
        n = len(st["items"])
        if total + n > max_files:
            if not chosen:
                # limit smaller than one stack: keep a truncated first stack
                st = dict(st, items=st["items"][:max_files])
                chosen.append(st)
            break
        chosen.append(st)
        total += n
    return sorted(chosen, key=lambda s: (s["orbit"], s["swath"], s["start"]))


# -------------------- LUT / geometry --------------------


def load_reference_lut(stack, tokens):
    """Download the LUT of the stack's reference product into RAM."""
    ref = stack["reference"]
    headers = {"Authorization": f"Bearer {tokens.get()}"}
    response = requests.get(ref.assets["enclosure_nc"].href, headers=headers, timeout=600)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        tmp.write(response.content)
        tmp.flush()
        del response
        with netCDF4.Dataset(tmp.name) as ds:
            geo = ds["geometry"]
            lut = {
                name: np.ma.filled(geo[name][:].astype(np.float64), np.nan)
                for name in ("latitude", "longitude", "height", "incidenceAngle")
            }

    # image size from the COG header, data/no-data from a small overview
    href = ref.assets["enclosure_i_abs_tiff"].href
    with rasterio.Env(GDAL_HTTP_HEADERS=f"Authorization: Bearer {tokens.get()}", **GDAL_ENV):
        with rasterio.open(href) as src:
            height, width = src.height, src.width
        with rasterio.open(href, overview_level=VALID_OVERVIEW_LEVEL) as ovr:
            # erode by one overview pixel to stay clear of the no-data edge
            has_data = binary_erosion(ovr.read(1) != NODATA)

    n_az, n_rg = lut["latitude"].shape
    # The LUT is the image grid decimated by an integer step, starting at pixel (0, 0)
    step_az, step_rg = round((height - 1) / (n_az - 1)), round((width - 1) / (n_rg - 1))
    rows = np.minimum(np.arange(n_az) * step_az * has_data.shape[0] // height, has_data.shape[0] - 1)
    cols = np.minimum(np.arange(n_rg) * step_rg * has_data.shape[1] // width, has_data.shape[1] - 1)
    valid = has_data[np.ix_(rows, cols)] & np.isfinite(lut["latitude"]) & np.isfinite(lut["longitude"])

    stack["lut"] = lut
    stack["image_shape"] = (height, width)
    stack["lut_step"] = (step_az, step_rg)
    stack["footprint"] = data_footprint(lut["latitude"], lut["longitude"], valid)
    return stack


def data_footprint(lat, lon, valid):
    """Polygon around the LUT nodes with data: first/last valid range node of each azimuth line."""
    near, far = [], []
    for i in np.nonzero(valid.any(axis=1))[0]:
        js = np.nonzero(valid[i])[0]
        near.append((lon[i, js[0]], lat[i, js[0]]))
        far.append((lon[i, js[-1]], lat[i, js[-1]]))
    polygon = shapely.Polygon(near + far[::-1])
    return polygon if polygon.is_valid else shapely.make_valid(polygon)


def radar_window(stack, area):
    """Pixel window of the image (whole LUT cells) that contains `area`."""
    lat, lon = stack["lut"]["latitude"], stack["lut"]["longitude"]
    valid = np.isfinite(lat) & np.isfinite(lon)
    inside = np.zeros_like(valid)
    inside[valid] = shapely.contains_xy(area, lon[valid], lat[valid])
    if not inside.any():
        raise RuntimeError(f"Common area falls between LUT nodes of {stack['stack_id']}")

    rows, cols = np.nonzero(inside)
    n_az, n_rg = lat.shape
    i0, i1 = max(rows.min() - 1, 0), min(rows.max() + 1, n_az - 1)
    j0, j1 = max(cols.min() - 1, 0), min(cols.max() + 1, n_rg - 1)
    step_az, step_rg = stack["lut_step"]
    height, width = stack["image_shape"]

    row_off, col_off = i0 * step_az, j0 * step_rg
    row_end, col_end = min(i1 * step_az + 1, height), min(j1 * step_rg + 1, width)
    stack["lut_slice"] = (i0, i1 + 1, j0, j1 + 1)
    stack["window"] = Window(col_off, row_off, col_end - col_off, row_end - row_off)


def window_latlon(stack):
    """Bilinear interpolation of the LUT lat/lon onto every pixel of the window."""
    w = stack["window"]
    step_az, step_rg = stack["lut_step"]
    rows = (np.arange(w.row_off, w.row_off + w.height) / step_az).astype(np.float32)
    cols = (np.arange(w.col_off, w.col_off + w.width) / step_rg).astype(np.float32)
    out = np.empty((2, w.height, w.width), dtype=np.float32)
    block = 1024
    for r0 in range(0, w.height, block):
        rr, cc = np.meshgrid(rows[r0:r0 + block], cols, indexing="ij")
        for k, name in enumerate(("latitude", "longitude")):
            out[k, r0:r0 + block] = map_coordinates(stack["lut"][name], [rr, cc], order=1, mode="nearest")
    return out


def stack_gcps(stack):
    """A coarse grid of GCPs (relative to the window) taken from the LUT."""
    i0, i1, j0, j1 = stack["lut_slice"]
    step_az, step_rg = stack["lut_step"]
    w = stack["window"]
    lut = stack["lut"]
    gcps = []
    for i in np.linspace(i0, i1 - 1, GCP_GRID[0]).astype(int):
        for j in np.linspace(j0, j1 - 1, GCP_GRID[1]).astype(int):
            lat, lon, h = lut["latitude"][i, j], lut["longitude"][i, j], lut["height"][i, j]
            if np.isfinite(lat) and np.isfinite(lon):
                gcps.append(GroundControlPoint(
                    row=i * step_az - w.row_off, col=j * step_rg - w.col_off,
                    x=float(lon), y=float(lat), z=float(h) if np.isfinite(h) else 0.0,
                ))
    return gcps


def stack_folder_name(stack):
    return "STA{}_{}_{}-{}_{}".format(
        stack["swath"][1], stack["orbit"],
        stack["start"].strftime("%Y%m%d"), stack["end"].strftime("%Y%m%d"), stack["grid"],
    )


def write_stack_geometry(stack, folder, area):
    """overlap_mask.tif, latlon.tif and geometry_lut.nc for one stack."""
    w = stack["window"]
    latlon = window_latlon(stack)
    mask = np.zeros((w.height, w.width), dtype=np.uint8)
    for r0 in range(0, w.height, 1024):
        mask[r0:r0 + 1024] = shapely.contains_xy(area, latlon[1, r0:r0 + 1024], latlon[0, r0:r0 + 1024])
    stack["mask_fraction"] = float(mask.mean())

    profile = dict(driver="GTiff", width=w.width, height=w.height, tiled=True,
                   blockxsize=512, blockysize=512, compress="deflate")
    gcps = stack_gcps(stack)
    with rasterio.open(folder / "overlap_mask.tif", "w", count=1, dtype="uint8", **profile) as dst:
        dst.write(mask, 1)
        dst.gcps = (gcps, rasterio.CRS.from_epsg(4326))
    with rasterio.open(folder / "latlon.tif", "w", count=2, dtype="float32", **profile) as dst:
        dst.write(latlon)
        dst.gcps = (gcps, rasterio.CRS.from_epsg(4326))
        dst.set_band_description(1, "latitude")
        dst.set_band_description(2, "longitude")
    del latlon, mask

    i0, i1, j0, j1 = stack["lut_slice"]
    step_az, step_rg = stack["lut_step"]
    with netCDF4.Dataset(str(folder / "geometry_lut.nc"), "w") as ds:
        ds.createDimension("azimuth", i1 - i0)
        ds.createDimension("range", j1 - j0)
        ds.source_product = stack["reference"].id
        ds.description = (
            "LUT geometry cropped to the window. LUT node (a, r) is crop pixel "
            "(a*azimuth_step + lut_row_offset, r*range_step + lut_col_offset)."
        )
        ds.azimuth_step, ds.range_step = step_az, step_rg
        ds.lut_row_offset = i0 * step_az - w.row_off
        ds.lut_col_offset = j0 * step_rg - w.col_off
        for name, data in stack["lut"].items():
            var = ds.createVariable(name, "f8", ("azimuth", "range"), fill_value=np.nan, zlib=True)
            var[:] = data[i0:i1, j0:j1]


# -------------------- Product extraction --------------------


def extract_product(item, stack, folder, tokens):
    """Range-read the window of abs/phase, build complex64 and write it."""
    out_path = folder / f"{item.id}.tif"
    w = stack["window"]
    if out_path.exists():
        # a new stack selection changes the common area, hence the window
        with rasterio.open(out_path) as old:
            tags = old.tags()
            same_window = (
                (old.height, old.width) == (w.height, w.width)
                and tags.get("window_row_off") == str(int(w.row_off))
                and tags.get("window_col_off") == str(int(w.col_off))
            )
        if same_window:
            return item.id, "skipped (exists)"
    tmp_path = out_path.with_suffix(".tif.part")
    abs_href = item.assets["enclosure_i_abs_tiff"].href
    phase_href = item.assets["enclosure_i_phase_tiff"].href
    props = item.properties

    profile = dict(driver="GTiff", width=w.width, height=w.height, count=len(POLARISATIONS),
                   dtype="complex64", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")

    with rasterio.Env(GDAL_HTTP_HEADERS=f"Authorization: Bearer {tokens.get()}", **GDAL_ENV):
        with rasterio.open(abs_href) as src_abs, rasterio.open(phase_href) as src_phase:
            if (src_abs.height, src_abs.width) != stack["image_shape"]:
                raise RuntimeError(f"{item.id}: image size differs from the stack reference")
            with rasterio.open(tmp_path, "w", **profile) as dst:
                for band, pol in enumerate(POLARISATIONS, start=1):
                    amplitude = src_abs.read(band, window=w)
                    phase = src_phase.read(band, window=w)
                    nodata = (amplitude == NODATA) | (phase == NODATA)
                    signal = (amplitude * np.exp(1j * phase)).astype(np.complex64)
                    signal[nodata] = np.complex64(np.nan + 1j * np.nan)
                    dst.write(signal, band)
                    dst.set_band_description(band, pol)
                    del amplitude, phase, nodata, signal
                dst.gcps = (stack_gcps(stack), rasterio.CRS.from_epsg(4326))
                dst.update_tags(
                    product_id=item.id,
                    stack_id=stack["stack_id"],
                    datetime=props.get("datetime"),
                    orbit_state=props.get("sat:orbit_state"),
                    repeat_cycle=props.get("eofeos:repeat_cycle_id"),
                    coregistration_primary=str(props.get("eofeos:is_coregistration_primary")),
                    window_row_off=int(w.row_off), window_col_off=int(w.col_off),
                    image_height=stack["image_shape"][0], image_width=stack["image_shape"][1],
                    axes="rows = azimuth, cols = slant range",
                )
    tmp_path.rename(out_path)
    return item.id, "ok"


# -------------------- Main --------------------


def setup_logging(log_file=None):
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    if not log.handlers:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        log.addHandler(console)
    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(fmt)
        log.addHandler(handler)


def print_stacks(stacks, title):
    log.info(title)
    for st in stacks:
        log.info("  %-48s %3d files  ROI coverage %5.1f%%",
                 stack_folder_name(st), len(st["items"]), 100 * st["stac_coverage"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", help="ROI name (also the output folder name)")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--start", help="start date YYYY-MM-DD")
    parser.add_argument("--end", help="end date YYYY-MM-DD")
    parser.add_argument("--max-files", type=int, help="max number of products; whole stacks are taken")
    parser.add_argument("--orbit", choices=["both", "ascending", "descending"])
    parser.add_argument("--swaths", nargs="+", default=["S1", "S2", "S3"])
    parser.add_argument("--min-coverage", type=float,
                        help="drop stacks covering less than this fraction of the ROI (0-1) "
                             "before intersecting, so one marginal stack does not empty the common area")
    parser.add_argument("--workers", type=int, default=6, help="parallel download threads")
    parser.add_argument("--token-file", help="token.txt (raw offline token) or credentials.txt")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR), help="parent folder of the ROI folder")
    parser.add_argument("--stacks", help='stack indices to keep, e.g. "0,2,5-7" or "all" (skips the selection prompt)')
    parser.add_argument("--yes", action="store_true", help="keep the recommended stacks without asking")
    parser.add_argument("--dry-run", action="store_true",
                        help="stop after the common area and windows, without downloading products")
    args = parser.parse_args()

    setup_logging()
    cfg = collect_inputs(args)
    tokens = load_token_manager(find_token_file(args.token_file))
    tokens.get()  # fail early on a bad token

    catalog = Client.open(STAC_API_URL)
    stacks = search_stacks(catalog, cfg, args.workers)
    if not stacks:
        raise SystemExit("No STA stacks found for this bbox / time span")
    log.info("Found %d stacks", len(stacks))

    recommended = [st for st in best_frame_per_track(stacks) if st["stac_coverage"] >= cfg["min_coverage"]]
    recommended = limit_files(recommended, cfg["max_files"]) if recommended else []

    roi_dir = Path(args.out_dir) / cfg["folder"]
    roi_dir.mkdir(parents=True, exist_ok=True)
    stacks = choose_stacks(stacks, recommended, cfg["bbox"], roi_dir / "stack_selection_map.html",
                           preset=args.stacks, interactive=not args.yes)
    if not stacks:
        raise SystemExit(f"No stack selected (none covers at least {cfg['min_coverage']:.0%} of the ROI?)")
    stacks = sorted(stacks, key=lambda s: (s["orbit"], s["swath"], s["start"]))

    setup_logging(roi_dir / "extraction.log")
    n_files = sum(len(st["items"]) for st in stacks)
    print_stacks(stacks, f"Extracting {len(stacks)} stacks, {n_files} files:")

    # one LUT per stack: all members share the reference product's radar grid
    log.info("Downloading %d reference LUTs ...", len(stacks))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda st: load_reference_lut(st, tokens), stacks))

    roi = box(*cfg["bbox"])
    area = roi
    for st in stacks:
        st["lut_coverage"] = st["footprint"].intersection(roi).area / roi.area
        area = area.intersection(st["footprint"])
    if area.is_empty or area.area == 0:
        for st in stacks:
            log.info("  %-48s LUT footprint covers %5.1f%% of ROI", stack_folder_name(st), 100 * st["lut_coverage"])
        raise SystemExit("The selected stacks have no area in common inside the ROI. "
                         "Pick stacks that overlap on the map (or try --min-coverage, --orbit, a smaller bbox).")
    log.info("Common area: %.1f%% of the ROI, bounds %s",
             100 * area.area / roi.area, [round(v, 4) for v in area.bounds])

    (roi_dir / "common_area.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": "common_area"}, "geometry": mapping(area)},
            {"type": "Feature", "properties": {"name": "roi"}, "geometry": mapping(roi)},
        ],
    }, indent=1))

    # per-stack window, geometry files and folder
    for st in stacks:
        radar_window(st, area)
        st["folder"] = roi_dir / stack_folder_name(st)
        st["folder"].mkdir(exist_ok=True)
        write_stack_geometry(st, st["folder"], area)
        w = st["window"]
        log.info("  %-48s window %d x %d px (az x rg), %.0f%% inside common area",
                 st["folder"].name, w.height, w.width, 100 * st["mask_fraction"])

    if args.dry_run:
        log.info("Dry run: products not downloaded")
        return

    # extract every product, several in parallel
    status = {}
    jobs = [(item, st) for st in stacks for item in st["items"]]
    log.info("Extracting %d products with %d threads ...", len(jobs), args.workers)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_product, item, st, st["folder"], tokens): item.id for item, st in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            product_id = futures[future]
            try:
                status[product_id] = future.result()[1]
            except Exception as exc:  # keep going, report at the end
                status[product_id] = f"failed: {exc}"
            log.info("[%d/%d] %s  %s", done, len(jobs), product_id, status[product_id])
    log.info("Extraction finished in %.1f min", (time.time() - t0) / 60)

    # summaries
    stack_summaries = []
    for st in stacks:
        w = st["window"]
        info = {
            "stack_id": st["stack_id"],
            "folder": st["folder"].name,
            "swath": st["swath"],
            "orbit_state": st["orbit"],
            "grid": st["grid"],
            "start": st["start"].isoformat(),
            "end": st["end"].isoformat(),
            "reference_product": st["reference"].id,
            "image_shape": list(st["image_shape"]),
            "lut_step_azimuth_range": list(st["lut_step"]),
            "window": {"row_off": int(w.row_off), "col_off": int(w.col_off),
                       "height": int(w.height), "width": int(w.width)},
            "roi_coverage_lut": round(st["lut_coverage"], 4),
            "fraction_of_window_in_common_area": round(st["mask_fraction"], 4),
            "products": [
                {
                    "id": it.id,
                    "file": f"{it.id}.tif",
                    "datetime": it.properties.get("datetime"),
                    "repeat_cycle": it.properties.get("eofeos:repeat_cycle_id"),
                    "coregistration_primary": it.properties.get("eofeos:is_coregistration_primary"),
                    "status": status.get(it.id),
                }
                for it in st["items"]
            ],
        }
        (st["folder"] / "stack_info.json").write_text(json.dumps(info, indent=1))
        stack_summaries.append(info)

    summary = {
        "roi": {k: cfg[k] for k in ("name", "bbox", "start", "end", "max_files", "orbit", "min_coverage", "swaths")},
        "common_area_fraction_of_roi": round(area.area / roi.area, 4),
        "common_area_bounds": list(area.bounds),
        "stacks": stack_summaries,
    }
    (roi_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=1))

    failed = [p for p, s in status.items() if s.startswith("failed")]
    log.info("Done: %d ok/skipped, %d failed. Output in %s", len(status) - len(failed), len(failed), roi_dir)
    if failed:
        log.info("Re-run the same command to retry the failed products (finished ones are skipped).")


if __name__ == "__main__":
    main()
