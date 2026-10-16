"""Extract calibrated ROI crops of BIOMASS L1C STA stacks (ported from extract_l1c_roi.py).

Workflow
--------
1. Search the STA stacks over the ROI bbox and time span (extraction.stacks),
   recommend some and let the user choose (extraction.selection: table + map).
2. Download ONE LUT per stack, the reference product's: geometry (lat, lon,
   height, incidence), footprint traced on the pixels holding data (from a
   COG overview) and the reference's own phase screens.
3. Common area = ROI ∩ footprint of every selected stack. It is mapped into
   each stack's radar grid to get a pixel window; overlap mask, lat/lon,
   cropped geometry LUT and pixel spacings are written once per stack.
4. Every product (parallel threads): its LUT phase screens (flattening, SKP,
   kz) cropped to the window, the window tiles of i_abs / i_phase fetched with
   HTTP range requests, s = abs * exp(1j * phase) corrected with
   exp(1j * skp) * exp(1j * flat) (lut.phase_correction) and written as
   complex64 HH/HV/VH/VV. The phase screens of all products go to phase_lut.nc.

Output (<roi>/extraction/)
--------------------------
    stack_selection_map.html, extraction_summary.json, common_area.geojson, extraction.log
    <stack>/
        BIO_S1_STA__1S_...tif   complex64, 4 bands HH HV VH VV, SKP + flattening applied,
                                radar geometry crop (rows = azimuth, cols = slant range), GCPs
        overlap_mask.tif        uint8, 1 = inside the common area
        latlon.tif              float32 lat / lon of every crop pixel
        geometry_lut.nc         reference LUT geometry (lat, lon, height, incidence), cropped
        phase_lut.nc            flattening, SKP (+ quality) and kz of every product, cropped
        stack_info.json         window, LUT steps, pixel spacings, products and their corrections
"""

import json
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from pystac_client import Client
from scipy.ndimage import binary_erosion
from shapely.geometry import box, mapping

from ..core.io import GDAL_ENV, NODATA, POLARISATIONS
from . import geometry, lut
from .selection import choose_stacks
from .stacks import recommend, search_stacks, stack_folder_name

log = logging.getLogger("flooded_forest.extraction")
# outputs are in radar geometry: georeferencing is given by GCPs, set after opening
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# COG overview used to find the pixels with data (1 = 1/2, 2 = 1/5, 3 = 1/10)
VALID_OVERVIEW_LEVEL = 1


def _gdal_env(token):
    return rasterio.Env(GDAL_HTTP_HEADERS=f"Authorization: Bearer {token.get()}", **GDAL_ENV)


# -------------------- per stack: reference LUT --------------------


def load_reference(stack, token):
    """Reference product of a stack: LUT geometry + phase screens, image size and data footprint."""
    ref = stack["reference"]
    with lut.fetched_lut(ref, token) as path:
        stack["lut"] = lut.read_layers(path, lut.GEOMETRY_LAYERS)
        stack["ref_screens"] = lut.read_layers(path, lut.PHASE_LAYERS)
        stack["lut_axes"] = lut.read_axes(path)

    # image size from the COG header, data/no-data from a small overview
    href = ref.assets["enclosure_i_abs_tiff"].href
    with _gdal_env(token):
        with rasterio.open(href) as src:
            height, width = src.height, src.width
        with rasterio.open(href, overview_level=VALID_OVERVIEW_LEVEL) as ovr:
            # erode by one overview pixel to stay clear of the no-data edge
            has_data = binary_erosion(ovr.read(1) != NODATA)

    lat, lon = stack["lut"]["latitude"], stack["lut"]["longitude"]
    n_az, n_rg = lat.shape
    # the LUT is the image grid decimated by an integer step, starting at pixel (0, 0)
    step_az, step_rg = round((height - 1) / (n_az - 1)), round((width - 1) / (n_rg - 1))
    rows = np.minimum(np.arange(n_az) * step_az * has_data.shape[0] // height, has_data.shape[0] - 1)
    cols = np.minimum(np.arange(n_rg) * step_rg * has_data.shape[1] // width, has_data.shape[1] - 1)
    valid = has_data[np.ix_(rows, cols)] & np.isfinite(lat) & np.isfinite(lon)

    stack["image_shape"] = (height, width)
    stack["lut_step"] = (step_az, step_rg)
    stack["footprint"] = geometry.data_footprint(lat, lon, valid)
    return stack


# -------------------- per product --------------------


def product_screens(item, stack, token):
    """Phase screens of one product cropped to the stack's LUT slice (reference: already in memory)."""
    if item.id == stack["reference"].id:
        return dict(stack["ref_screens"])
    i0, i1, j0, j1 = stack["lut_slice"]
    with lut.fetched_lut(item, token) as path:
        screens = lut.read_layers(path, lut.PHASE_LAYERS, stack["lut_slice"])
    shapes = {data.shape for data in screens.values()}
    if shapes and shapes != {(i1 - i0, j1 - j0)}:
        raise RuntimeError(f"{item.id}: LUT grid differs from the stack reference")
    return screens


def _same_product(path, window, tag):
    """True when an existing tif has the same window and the same phase corrections."""
    try:
        with rasterio.open(path) as old:
            tags = old.tags()
            return (
                (old.height, old.width) == (window.height, window.width)
                and tags.get("window_row_off") == str(int(window.row_off))
                and tags.get("window_col_off") == str(int(window.col_off))
                and tags.get("phase_calibration") == tag
            )
    except rasterio.errors.RasterioIOError:
        return False


def extract_product(item, stack, folder, token, cached_screens, apply_skp, apply_flat):
    """Range-read the window of abs/phase, apply SKP + flattening, write complex64.

    Returns (product_id, status, screens, correction tag).
    """
    w = stack["window"]
    out_path = folder / f"{item.id}.tif"
    screens = cached_screens.get(item.id) or product_screens(item, stack, token)
    planned = lut.planned_corrections(screens, apply_skp, apply_flat, item.id)
    tag = lut.correction_tag(planned)
    # a new stack selection changes the common area, hence the window
    if out_path.exists() and _same_product(out_path, w, tag):
        return item.id, "skipped (exists)", screens, tag

    ramp = lut.phase_correction(screens, (w.height, w.width), stack["lut_step"], planned)
    tmp_path = out_path.with_suffix(".tif.part")
    props = item.properties
    profile = dict(driver="GTiff", width=w.width, height=w.height, count=len(POLARISATIONS),
                   dtype="complex64", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")

    with _gdal_env(token):
        abs_href = item.assets["enclosure_i_abs_tiff"].href
        phase_href = item.assets["enclosure_i_phase_tiff"].href
        with rasterio.open(abs_href) as src_abs, rasterio.open(phase_href) as src_phase:
            if (src_abs.height, src_abs.width) != tuple(stack["image_shape"]):
                raise RuntimeError(f"{item.id}: image size differs from the stack reference")
            with rasterio.open(tmp_path, "w", **profile) as dst:
                for band, pol in enumerate(POLARISATIONS, start=1):
                    amplitude = src_abs.read(band, window=w)
                    phase = src_phase.read(band, window=w)
                    nodata = (amplitude == NODATA) | (phase == NODATA)
                    signal = (amplitude * np.exp(1j * phase)).astype(np.complex64)
                    if ramp is not None:
                        signal *= ramp
                    signal[nodata] = np.complex64(np.nan + 1j * np.nan)
                    dst.write(signal, band)
                    dst.set_band_description(band, pol)
                    del amplitude, phase, nodata, signal
                dst.gcps = (geometry.stack_gcps(stack), rasterio.CRS.from_epsg(4326))
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
                    phase_calibration=tag,
                    phase_correction=lut.PHASE_CORRECTION if planned else "none",
                )
    tmp_path.rename(out_path)
    return item.id, "ok", screens, tag


# -------------------- main --------------------


def setup_logging(log_file=None):
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.propagate = False
    if not any(type(h) is logging.StreamHandler for h in log.handlers):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        log.addHandler(console)
    if log_file and not any(getattr(h, "baseFilename", None) == str(log_file) for h in log.handlers):
        handler = logging.FileHandler(log_file)
        handler.setFormatter(fmt)
        log.addHandler(handler)


def _print_stacks(stacks, title):
    log.info(title)
    for st in stacks:
        log.info("  %-48s %3d files  ROI coverage %5.1f%%",
                 stack_folder_name(st), len(st["items"]), 100 * st["stac_coverage"])


def _stack_info(st, status, tags, apply_skp, apply_flat):
    w = st["window"]
    return {
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
        "lut_slice": list(st["lut_slice"]),
        "window": {"row_off": int(w.row_off), "col_off": int(w.col_off),
                   "height": int(w.height), "width": int(w.width)},
        "spacings": st["spacings"],
        "roi_coverage_lut": round(st["lut_coverage"], 4),
        "fraction_of_window_in_common_area": round(st["mask_fraction"], 4),
        "phase_correction": {"skp": apply_skp, "flat": apply_flat, "formula": lut.PHASE_CORRECTION},
        "products": [
            {
                "id": it.id,
                "file": f"{it.id}.tif",
                "datetime": it.properties.get("datetime"),
                "repeat_cycle": it.properties.get("eofeos:repeat_cycle_id"),
                "coregistration_primary": it.properties.get("eofeos:is_coregistration_primary"),
                "status": status.get(it.id),
                "phase_calibration": tags.get(it.id),
            }
            for it in st["items"]
        ],
    }


def run_extraction(config, roi, start, end, token, roi_paths, stacks=None, yes=False, orbit="both",
                   max_files=None, min_coverage=None, workers=None, dry_run=False, swaths=("S1", "S2", "S3")):
    """Search, select and extract the calibrated stacks of a ROI into roi_paths.extraction."""
    max_files = config.max_files if max_files is None else max_files
    min_coverage = config.min_coverage if min_coverage is None else min_coverage
    workers = config.workers if workers is None else workers
    apply_skp, apply_flat = config.apply_skp, config.apply_flat
    out_dir = roi_paths.extraction
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging()

    catalog = Client.open(config.stac_api_url)
    found = search_stacks(catalog, config.collection, roi["bbox"], start, end, swaths, orbit, workers)
    if not found:
        raise SystemExit("No STA stacks found for this bbox / time span")
    log.info("Found %d stacks", len(found))

    chosen = choose_stacks(found, recommend(found, min_coverage, max_files), roi["bbox"],
                           out_dir / "stack_selection_map.html", preset=stacks, interactive=not yes)
    if not chosen:
        raise SystemExit(f"No stack selected (none covers at least {min_coverage:.0%} of the ROI?)")
    chosen = sorted(chosen, key=lambda s: (s["orbit"], s["swath"], s["start"]))

    setup_logging(out_dir / "extraction.log")
    _print_stacks(chosen, f"Extracting {len(chosen)} stacks, {sum(len(st['items']) for st in chosen)} files:")
    log.info("Phase corrections: SKP %s, flattening %s", "on" if apply_skp else "off", "on" if apply_flat else "off")

    # one LUT per stack: all members share the reference product's radar grid
    log.info("Downloading %d reference LUTs ...", len(chosen))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda st: load_reference(st, token), chosen))

    roi_box = box(*roi["bbox"])
    area = roi_box
    for st in chosen:
        st["lut_coverage"] = st["footprint"].intersection(roi_box).area / roi_box.area
        area = area.intersection(st["footprint"])
    if area.is_empty or area.area == 0:
        for st in chosen:
            log.info("  %-48s LUT footprint covers %5.1f%% of ROI", stack_folder_name(st), 100 * st["lut_coverage"])
        raise SystemExit("The selected stacks have no area in common inside the ROI. "
                         "Pick stacks that overlap on the map (or try --min-coverage, --orbit, a smaller bbox).")
    log.info("Common area: %.1f%% of the ROI, bounds %s",
             100 * area.area / roi_box.area, [round(v, 4) for v in area.bounds])
    (out_dir / "common_area.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": "common_area"}, "geometry": mapping(area)},
            {"type": "Feature", "properties": {"name": "roi"}, "geometry": mapping(roi_box)},
        ],
    }, indent=1))

    # per-stack window, geometry files (computed once per stack) and folder
    for st in chosen:
        geometry.radar_window(st, area)
        i0, i1, j0, j1 = st["lut_slice"]
        # the reference's phase screens, cropped once (frees the full-size LUT layers)
        st["ref_screens"] = {k: v[i0:i1, j0:j1].copy() for k, v in st["ref_screens"].items()}
        st["folder"] = out_dir / stack_folder_name(st)
        st["folder"].mkdir(exist_ok=True)
        geometry.write_stack_geometry(st, st["folder"], area)
        w, sp = st["window"], st["spacings"]
        log.info("  %-48s window %d x %d px (az x rg), %.0f%% inside common area, "
                 "spacing az %.2f m, ground rg %.2f m (incidence %.1f deg)",
                 st["folder"].name, w.height, w.width, 100 * st["mask_fraction"],
                 sp["azimuth_m"], sp["ground_range_m"], sp["incidence_mean_deg"])

    if dry_run:
        log.info("Dry run: products not downloaded")
        return out_dir

    # extract every product, several in parallel
    status, tags = {}, {}
    screens = {st["stack_id"]: {} for st in chosen}
    cached = {st["stack_id"]: lut.load_phase_lut(st["folder"] / "phase_lut.nc", st["lut_slice"]) for st in chosen}
    jobs = [(item, st) for st in chosen for item in st["items"]]
    log.info("Extracting %d products with %d threads ...", len(jobs), workers)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(extract_product, item, st, st["folder"], token, cached[st["stack_id"]],
                        apply_skp, apply_flat): (item.id, st["stack_id"])
            for item, st in jobs
        }
        for done, future in enumerate(as_completed(futures), start=1):
            product_id, stack_id = futures[future]
            try:
                _, status[product_id], screens[stack_id][product_id], tags[product_id] = future.result()
            except Exception as exc:  # keep going, report at the end
                status[product_id] = f"failed: {exc}"
            log.info("[%d/%d] %s  %s  (%s)", done, len(jobs), product_id, status[product_id],
                     tags.get(product_id, "-"))
    log.info("Extraction finished in %.1f min", (time.time() - t0) / 60)

    # phase screens + summaries
    stack_summaries = []
    for st in chosen:
        by_pid = {it.id: screens[st["stack_id"]][it.id] for it in st["items"] if it.id in screens[st["stack_id"]]}
        if by_pid:
            lut.write_phase_lut(st["folder"] / "phase_lut.nc", st, by_pid, tags)
        info = _stack_info(st, status, tags, apply_skp, apply_flat)
        (st["folder"] / "stack_info.json").write_text(json.dumps(info, indent=1))
        stack_summaries.append(info)

    summary = {
        "roi": {"name": roi["name"], "bbox": roi["bbox"], "start": start, "end": end, "max_files": max_files,
                "orbit": orbit, "min_coverage": min_coverage, "swaths": list(swaths)},
        "common_area_fraction_of_roi": round(area.area / roi_box.area, 4),
        "common_area_bounds": list(area.bounds),
        "phase_correction": {"skp": apply_skp, "flat": apply_flat},
        "stacks": stack_summaries,
    }
    (out_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=1))

    failed = [p for p, s in status.items() if s.startswith("failed")]
    log.info("Done: %d ok/skipped, %d failed. Output in %s", len(status) - len(failed), len(failed), out_dir)
    if failed:
        log.info("Re-run the same command to retry the failed products (finished ones are skipped).")
    return out_dir
