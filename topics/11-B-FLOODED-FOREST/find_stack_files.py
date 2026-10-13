#!/usr/bin/env python3
"""
Find MAAP STAC products whose footprints overlap ROI bounding boxes.

Input:
  rois.json — list of objects containing:
    {
      "name": "...",
      "bbox": {
        "west": ..., "south": ..., "east": ..., "north": ...
      }
    }

Output:
  sta_level1c_by_roi.json — matching products grouped by ROI.

Notes:
- Set STAC_COLLECTION_ID to the exact STA Level-1C collection ID in your MAAP
  catalogue. The collection in the original example (BiomassLevel1a) may not
  be the correct Level-1C collection.
- This script tests actual geometry intersection when an item exposes a usable
  GeoJSON geometry. If geometry is missing, it falls back to the STAC bbox.
- "overlap" means any non-zero-area intersection with the ROI. Set
  MIN_OVERLAP_FRACTION > 0 to require a minimum fraction of the ROI covered.
"""

import json
import os
import time
from pathlib import Path

import requests
from pystac_client import Client
from shapely.geometry import box, shape
from shapely.errors import ShapelyError

# -------------------- Configuration --------------------
import yaml

# -------------------- ROI and geometry helpers --------------------


def load_rois(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept either a top-level list or {"rois": [...]}
    if isinstance(data, dict):
        data = data.get("rois")
    if not isinstance(data, list):
        raise ValueError("ROI JSON must be a list, or an object with a 'rois' list.")

    rois = []
    for index, roi in enumerate(data):
        name = roi.get("name", f"ROI_{index + 1}")
        bbox = roi.get("bbox")
        if not isinstance(bbox, dict):
            raise ValueError(f"ROI '{name}' is missing its bbox object.")

        try:
            west = float(bbox["west"])
            south = float(bbox["south"])
            east = float(bbox["east"])
            north = float(bbox["north"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"ROI '{name}' has invalid bbox coordinates.") from exc

        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError(f"ROI '{name}' has invalid geographic bounds: {bbox}")

        if (east - west) > 0.5 + 1e-9 or (north - south) > 0.5 + 1e-9:
            print(f"ROI '{name}' exceeds 0.5 degrees in longitude or latitude.")

        rois.append(
            {
                "name": name,
                "bbox": [west, south, east, north],
                "geometry": box(west, south, east, north),
            }
        )

    return rois


def item_geometry(item):
    """Return a Shapely footprint from STAC geometry, or bbox fallback."""
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
    """Return all asset URLs; asset keys are preserved."""
    return {
        key: asset.href
        for key, asset in item.assets.items()
        if getattr(asset, "href", None)
    }


def serialize_item(item, roi_geometry):
    footprint, geometry_source = item_geometry(item)
    if footprint is None:
        return None

    if not footprint.intersects(roi_geometry):
        return None

    intersection = footprint.intersection(roi_geometry)
    if intersection.is_empty:
        return None

    roi_area = roi_geometry.area
    overlap_fraction = (intersection.area / roi_area) if roi_area else 0.0
    if overlap_fraction + 1e-12 < MIN_OVERLAP_FRACTION:
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


# -------------------- Search --------------------


def main():
    if STAC_COLLECTION_ID == "REPLACE_WITH_STA_LEVEL1C_COLLECTION":
        raise SystemExit(
            "Set STAC_COLLECTION_ID to the exact STA Level-1C collection ID, "
            "or run with MAAP_STAC_COLLECTION='your_collection_id'. "
            "Do not use BiomassLevel1a unless the catalogue confirms it is correct."
        )

    rois = load_rois(ROI_JSON_PATH)
    offline_token, client_id, client_secret = load_credentials(CREDENTIALS_PATH)
    access_token = get_maap_access_token(offline_token, client_id, client_secret)

    catalog = Client.open(
        STAC_API_URL,
        headers={"Authorization": f"Bearer {access_token}"},
    )

    results = {
        "collection": STAC_COLLECTION_ID,
        "roi_json": str(ROI_JSON_PATH),
        "matching_rule": (
            "Product footprint intersects ROI"
            if MIN_OVERLAP_FRACTION == 0
            else f"Product footprint covers at least {MIN_OVERLAP_FRACTION:.1%} of ROI"
        ),
        "rois": [],
    }

    for roi in rois:
        print(f"Searching {roi['name']} ...")
        found = {}

        search = catalog.search(
            collections=[STAC_COLLECTION_ID],
            bbox=roi["bbox"],
            max_items=MAX_ITEMS_PER_ROI,
            # fields={
            #     "include": [
            #         "assets.enclosure_ql_pauli_png",
            #         "assets.enclosure_i_abs_tiff",
            #         "assets.enclosure_i_phase_tiff",
            #     ]
            # },
        )

        try:
            for item in search.items():
                serialized = serialize_item(item, roi["geometry"])
                if serialized is not None:
                    found[item.id] = serialized
                if REQUEST_PAUSE_SECONDS:
                    time.sleep(REQUEST_PAUSE_SECONDS)
        except Exception as exc:
            print(f"  Search error for {roi['name']}: {exc}")

        products = sorted(found.values(), key=lambda x: x["id"])
        results["rois"].append(
            {
                "name": roi["name"],
                "bbox": {
                    "west": roi["bbox"][0],
                    "south": roi["bbox"][1],
                    "east": roi["bbox"][2],
                    "north": roi["bbox"][3],
                },
                "product_count": len(products),
                "products": products,
            }
        )
        print(f"  Found {len(products)} overlapping products")

    output_path = Path(OUTPUT_JSON_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {output_path}")
    print("\nProduct IDs by ROI:")


if __name__ == "__main__":

    # IMPORTANT: replace with the exact collection ID for STA Level-1C products.
    STAC_COLLECTION_ID = "BiomassLevel1c"

    # 0.0 = any intersection. For example, 0.01 requires at least 1% of ROI area
    # to overlap the product footprint.
    MIN_OVERLAP_FRACTION = 0.0
    MAX_ITEMS_PER_ROI = 1000
    REQUEST_PAUSE_SECONDS = 0.1

    main()
