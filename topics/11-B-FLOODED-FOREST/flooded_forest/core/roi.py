"""ROI resolution: preset name from rois.json, custom bbox, or interactive prompt."""

import json
import re
import sys

from shapely.geometry import box


def load_rois(path):
    """Return the preset ROI list: [{name, region, bbox:[w,s,e,n]}]."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("rois")
    rois = []
    for index, roi in enumerate(data):
        b = roi.get("bbox")
        rois.append(
            {
                "name": roi.get("name", f"ROI_{index + 1}"),
                "region": roi.get("region", ""),
                "bbox": [float(b["west"]), float(b["south"]), float(b["east"]), float(b["north"])],
            }
        )
    return rois


def slugify(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _match_roi(rois, name):
    hits = [r for r in rois if r["name"].lower() == name.lower()]
    if not hits:
        hits = [r for r in rois if name.lower() in r["name"].lower()]
    if len(hits) != 1:
        available = "\n".join(f"  - {r['name']}" for r in rois)
        raise SystemExit(f"ROI '{name}' not found (or ambiguous) in rois.json. Available:\n{available}")
    return hits[0]


def list_rois(rois):
    for i, r in enumerate(rois):
        w, s, e, n = r["bbox"]
        print(f"  [{i}] {r['name']}  ({r['region']})  bbox {w} {s} {e} {n}")


def resolve_roi(name=None, bbox=None, roi_json_path=None):
    """Return {name, slug, bbox:[w,s,e,n]} from a preset name, a custom bbox, or a prompt."""
    rois = load_rois(roi_json_path) if roi_json_path else []

    if name and not bbox:
        roi = _match_roi(rois, name)
        return {"name": roi["name"], "slug": slugify(roi["name"]), "bbox": roi["bbox"]}

    if bbox is not None:
        values = [float(v) for v in bbox]
        if len(values) != 4:
            raise SystemExit("--bbox expects 4 values: west south east north")
        west, south, east, north = values
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise SystemExit(f"Invalid bbox {values}: expected west < east and south < north")
        if not name:
            name = "roi_{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(west, south, east, north)
        return {"name": name, "slug": slugify(name), "bbox": values}

    # interactive
    if not sys.stdin.isatty():
        raise SystemExit("Pass --roi NAME (see --list-rois) or --bbox WEST SOUTH EAST NORTH")
    print("Preset ROIs:")
    list_rois(rois)
    answer = input("ROI number, or 'c' to enter 4 coordinates: ").strip()
    if answer.lower() in ("c", "custom"):
        raw = input("bbox as  west south east north  (decimal degrees): ")
        return resolve_roi(name=None, bbox=raw.split(), roi_json_path=None)
    roi = rois[int(answer)]
    return {"name": roi["name"], "slug": slugify(roi["name"]), "bbox": roi["bbox"]}


def roi_polygon(roi):
    return box(*roi["bbox"])
