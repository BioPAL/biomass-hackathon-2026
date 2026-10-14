"""Search the MAAP STAC catalogue for STA Level-1C products over a ROI.

Ported from find_stack_files.py. Results are merged per ROI into
assets/catalogue/sta_level1c_by_roi.json.
"""

import json
import time

from pystac_client import Client
from shapely.errors import ShapelyError
from shapely.geometry import box, shape

CATALOGUE_FILENAME = "sta_level1c_by_roi.json"


def item_geometry(item):
    """Shapely footprint from the STAC geometry, bbox fallback."""
    if item.geometry:
        try:
            geom = shape(item.geometry)
            if not geom.is_empty and geom.is_valid:
                return geom, "geometry"
            if not geom.is_empty:
                fixed = geom.buffer(0)
                if not fixed.is_empty and fixed.is_valid:
                    return fixed, "geometry_repaired"
        except (ShapelyError, TypeError, ValueError):
            pass
    if item.bbox and len(item.bbox) >= 4:
        west, south, east, north = item.bbox[:4]
        if west < east and south < north:
            return box(west, south, east, north), "bbox_fallback"
    return None, None


def get_asset_hrefs(item):
    return {key: asset.href for key, asset in item.assets.items() if getattr(asset, "href", None)}


def serialize_item(item, roi_geometry, min_overlap_fraction):
    footprint, geometry_source = item_geometry(item)
    if footprint is None or not footprint.intersects(roi_geometry):
        return None

    intersection = footprint.intersection(roi_geometry)
    if intersection.is_empty:
        return None

    roi_area = roi_geometry.area
    overlap_fraction = (intersection.area / roi_area) if roi_area else 0.0
    if overlap_fraction + 1e-12 < min_overlap_fraction:
        return None

    properties = dict(item.properties or {})
    return {
        "id": item.id,
        "collection": item.collection_id,
        "datetime": properties.get("datetime"),
        "start_datetime": properties.get("start_datetime"),
        "end_datetime": properties.get("end_datetime"),
        "bbox": item.bbox,
        "geometry_source": geometry_source,
        "roi_overlap_fraction": round(overlap_fraction, 6),
        "assets": get_asset_hrefs(item),
        "properties": {
            key: properties.get(key)
            for key in (
                "platform",
                "instruments",
                "product_type",
                "processing:level",
                "processing_level",
                "sar:instrument_mode",
                "created",
            )
            if properties.get(key) is not None
        },
    }


def _open_catalog(config):
    """Open the MAAP catalogue.

    The search is anonymous: the MAAP STAC API rejects authenticated search
    requests ("Invalid token"). The token is only used for the asset downloads.
    """
    return Client.open(config.stac_api_url)


def _search_with_retry(config, roi, token, attempts=3, backoff=10.0):
    """Search one ROI, retrying on transient catalogue errors ("Invalid token")."""
    from pystac_client.exceptions import APIError

    last_error = None
    for attempt in range(1, attempts + 1):
        catalog = _open_catalog(config)
        try:
            search = catalog.search(
                collections=[config.collection],
                bbox=roi["bbox"],
                max_items=config.max_items_per_roi,
            )
            items = list(search.items())
            if config.request_pause_seconds:
                time.sleep(config.request_pause_seconds)
            return items
        except (APIError, ConnectionError, TimeoutError) as exc:
            last_error = exc
            print(f"  catalogue error (attempt {attempt}/{attempts}): {str(exc)[:120]}")
            if attempt < attempts:
                time.sleep(backoff * attempt)
    raise RuntimeError(
        f"MAAP catalogue search failed after {attempts} attempts for '{roi['name']}'"
    ) from last_error


def search_roi(config, roi, token):
    """Search one ROI. Returns {name, bbox, product_count, products}."""
    print(f"Searching {roi['name']} ...")
    items = _search_with_retry(config, roi, token)
    roi_geometry = box(*roi["bbox"])

    found = {}
    for item in items:
        serialized = serialize_item(item, roi_geometry, config.min_overlap_fraction)
        if serialized is not None:
            found[item.id] = serialized

    products = sorted(found.values(), key=lambda x: x["id"])
    print(f"  Found {len(products)} overlapping products")
    return {
        "name": roi["name"],
        "bbox": dict(zip(("west", "south", "east", "north"), roi["bbox"])),
        "product_count": len(products),
        "products": products,
    }


def run_search(config, roi, token, catalogue_path=None):
    """Search and merge the ROI entry into assets/catalogue/sta_level1c_by_roi.json."""
    catalogue_path = catalogue_path or config.catalogue_dir / CATALOGUE_FILENAME
    catalogue_path.parent.mkdir(parents=True, exist_ok=True)

    entry = search_roi(config, roi, token)
    entry["slug"] = roi["slug"]

    if catalogue_path.exists():
        data = json.loads(catalogue_path.read_text(encoding="utf-8"))
    else:
        data = {"matching_rule": "Product footprint intersects ROI", "rois": []}
    data["collection"] = config.collection
    data["rois"] = [r for r in data.get("rois", []) if r.get("name") != entry["name"]]
    data["rois"].append(entry)

    catalogue_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Saved results to: {catalogue_path}")
    return entry
