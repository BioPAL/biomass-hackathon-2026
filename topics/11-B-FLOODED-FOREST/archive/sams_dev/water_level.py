#!/usr/bin/env python3
"""Water level of the nearest DAHITI target for a flooded-forest ROI (prototype).

Standalone, self-contained script in sams_dev/. For a ROI (preset name from
rois.json or a custom bbox) it:

1. finds the DAHITI water-level targets nearest to the ROI centroid
   (get-nearest-target, great-circle distance in km) and keeps the first one
   that has data inside PERIOD_START / PERIOD_END (below),
2. prints the distance from every considered area (the ROI, its common area and
   the extracted stacks, when they exist under assets/stacks/<roi>/) to that
   target,
3. prints the DAHITI source metadata (dataset, institution, software, url),
4. downloads the water-level time series and writes the raw data, the plot and
   an info JSON to a temp folder (/tmp/water_level/<roi>/).

DAHITI API: https://dahiti.dgfi.tum.de/en/api/doc/v2/

Examples
--------
  python water_level.py --roi bolobo
  python water_level.py --roi "Lac Télé"
  python water_level.py --bbox 16.6 -2.4 16.8 -2.2 --name "Bolobo site"
"""

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import requests

# ---------------------------------------------------------------------------
# DAHITI API key.
#
# TODO: MOVE THIS BEFORE PUSHING. Never commit API keys: read the key from the
# DAHITI_API_KEY environment variable (or a git-ignored credentials file)
# instead of hardcoding it here.
# ---------------------------------------------------------------------------
API_KEY = "589AD7A19BAE1CBA8FCE33756EBD90BD124ECB2A2F4E638BFADD12C6C4DE84C4"

# Temporal extent of the water-level series that is kept and plotted.
PERIOD_START = datetime(2024, 1, 1)  # inclusive
PERIOD_END = None                    # None -> now
MAX_TRIES = 10                       # targets tried at most to find data in the period

BASE_URL = "https://dahiti.dgfi.tum.de/api/v2/"
TOPIC_ROOT = Path(__file__).resolve().parent.parent
ROIS_JSON = TOPIC_ROOT / "rois.json"
STACKS_DIR = TOPIC_ROOT / "assets" / "stacks"
OUT_DIR = Path("/tmp/water_level")


# ------------------------- ROI helpers -------------------------

def slugify(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def load_rois(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("rois")
    rois = []
    for i, r in enumerate(data):
        b = r["bbox"]
        rois.append({
            "name": r.get("name", f"ROI_{i + 1}"),
            "bbox": [float(b["west"]), float(b["south"]), float(b["east"]), float(b["north"])],
        })
    return rois


def resolve_roi(name=None, bbox=None):
    if bbox is not None:
        west, south, east, north = (float(v) for v in bbox)
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise SystemExit(f"invalid bbox {bbox}: need west < east and south < north")
        return {"name": name or f"roi_{west}_{south}_{east}_{north}",
                "bbox": [west, south, east, north]}
    rois = load_rois(ROIS_JSON) if ROIS_JSON.exists() else []
    if not rois:
        raise SystemExit(f"no --bbox given and {ROIS_JSON} not found")
    key = (name or "").lower()
    hits = [r for r in rois if r["name"].lower() == key] or \
           [r for r in rois if key in r["name"].lower()]
    if len(hits) != 1:
        available = "\n".join(f"  - {r['name']}" for r in rois)
        raise SystemExit(f"ROI '{name}' not found or ambiguous in rois.json. Available:\n{available}")
    return hits[0]


def roi_centroid(bbox):
    """(lat, lon) of the ROI centre."""
    west, south, east, north = bbox
    return (south + north) / 2.0, (west + east) / 2.0


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def considered_areas(roi):
    """[(label, lat, lon)] of the ROI, its common area and its extracted stacks."""
    areas = [("ROI centroid", *roi_centroid(roi["bbox"]))]
    roi_dir = STACKS_DIR / slugify(roi["name"])

    common_path = roi_dir / "common_area.geojson"
    if common_path.exists():
        try:
            from shapely.geometry import shape
            data = json.loads(common_path.read_text(encoding="utf-8"))
            geom = next(shape(f["geometry"]) for f in data["features"]
                        if f.get("properties", {}).get("name") == "common_area")
            areas.append(("common area centroid", geom.centroid.y, geom.centroid.x))
        except Exception as exc:
            print(f"  (common area centroid skipped: {exc})")

    try:
        import numpy as np
        import rasterio
        from shapely.geometry import Polygon
    except ImportError:
        return areas

    for latlon in sorted(roi_dir.glob("*/latlon.tif")):
        try:
            with rasterio.open(latlon) as src:
                lat, lon = src.read(1)[::12, ::2], src.read(2)[::12, ::2]
            ring = np.concatenate([
                np.column_stack([lon[0, :], lat[0, :]]),
                np.column_stack([lon[:, -1], lat[:, -1]]),
                np.column_stack([lon[-1, ::-1], lat[-1, ::-1]]),
                np.column_stack([lon[::-1, 0], lat[::-1, 0]]),
            ])
            centroid = Polygon(ring).centroid
            areas.append((f"stack {latlon.parent.name}", centroid.y, centroid.x))
        except Exception as exc:
            print(f"  (stack {latlon.parent.name} centroid skipped: {exc})")
    return areas


# ------------------------- DAHITI API -------------------------

def _post(endpoint, **params):
    params["api_key"] = API_KEY
    response = requests.post(BASE_URL + endpoint + "/", data=params, timeout=180)
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") not in (200, None):
        raise RuntimeError(f"{endpoint}: {payload.get('message')}")
    return payload


def nearest_target(lat, lon):
    """Nearest DAHITI target: {id, target_name, latitude, longitude, distance, ...}."""
    return _post("get-nearest-target", latitude=lat, longitude=lon)["data"]


def targets_by_distance(lat, lon, skip_id, limit):
    """The `limit` DAHITI targets closest to (lat, lon), `skip_id` excluded."""
    targets = _post("list-targets")["data"]
    ranked = sorted(
        (t for t in targets if t["dahiti_id"] != skip_id),
        key=lambda t: haversine_km(lat, lon, t["latitude"], t["longitude"]),
    )
    return [{"id": t["dahiti_id"], "target_name": t["target_name"],
             "latitude": t["latitude"], "longitude": t["longitude"],
             "country": t.get("country"), "continent": t.get("continent"),
             "distance": haversine_km(lat, lon, t["latitude"], t["longitude"])}
            for t in ranked[:limit]]


def target_info(dahiti_id):
    try:
        return _post("get-target-info", dahiti_id=dahiti_id)["data"]
    except Exception:
        return {}


def download_series(dahiti_id):
    response = requests.post(
        BASE_URL + "download-water-level/",
        data={"api_key": API_KEY, "dahiti_id": dahiti_id, "format": "json"},
        timeout=180,
    )
    response.raise_for_status()
    return response.text


def parse_series(text):
    """(metadata, rows) from a JSON download; rows are (datetime, level, error)."""
    payload = json.loads(text)
    meta = {key: payload.get(key) for key in
            ("dahiti_id", "target_name", "dataset", "institution", "source", "software",
             "url", "creation_date", "longitude", "latitude", "country")}
    rows = []
    for point in payload.get("data", []):
        raw_dt = point.get("datetime") or point.get("date")
        level = point.get("wse", point.get("water_level"))
        error = point.get("wse_u", point.get("error"))
        if raw_dt is None or level is None:
            continue
        try:
            when = datetime.fromisoformat(str(raw_dt))
        except ValueError:
            when = datetime.strptime(str(raw_dt), "%Y-%m-%d")
        rows.append((when, float(level), None if error is None else float(error)))
    return meta, rows


def clip_period(rows):
    """Rows inside PERIOD_START / PERIOD_END (None -> now), and the period end used."""
    end = PERIOD_END or datetime.now()
    return [row for row in rows if PERIOD_START <= row[0] <= end], end


def pick_target(lat, lon):
    """Nearest DAHITI target with data inside the period, and its raw series.

    get-nearest-target gives the closest target; when its series has no data in
    the period, the next nearest targets (list-targets ranked by distance) are
    tried until one has data.
    """
    def attempts(first):
        yield first
        for candidate in targets_by_distance(lat, lon, first["id"], MAX_TRIES):
            yield candidate

    previous_ids = set()
    for target in attempts(nearest_target(lat, lon)):
        if target["id"] in previous_ids:
            continue
        previous_ids.add(target["id"])
        print(f"  trying {target['target_name']} (id {target['id']}, "
              f"{target['distance']:.1f} km) ...")
        raw = download_series(target["id"])
        rows = parse_series(raw)[1]
        if clip_period(rows)[0]:
            return target, raw
        last = f"{rows[-1][0]:%Y-%m-%d}" if rows else "empty"
        print(f"    no data in the period (series ends {last})")
    raise SystemExit(f"no DAHITI target with data in the period among the "
                     f"{len(previous_ids)} nearest targets")


# ------------------------- plot -------------------------

def plot_series(meta, rows, roi_dist, out_png):
    dates = [row[0] for row in rows]
    levels = [row[1] for row in rows]
    errors = [row[2] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(dates, levels, yerr=errors, fmt="-o", ms=4, lw=1.2, color="#2a78d6",
                ecolor="#c3c2b7", elinewidth=0.8, capsize=2, label="DAHITI water level")
    ax.set_xlabel("Date")
    ax.set_ylabel("Water level [m]")
    ax.set_title(f"{meta['target_name']} (DAHITI {meta['dahiti_id']}) - "
                 f"{roi_dist:.1f} km from the ROI centroid")
    ax.grid(color="#d9d8d2", lw=0.5)
    ax.set_axisbelow(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"  plot saved: {out_png}")
    plt.close(fig)


# ------------------------- main -------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roi", help="preset ROI name from rois.json (substring match)")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--name", help="name of a custom --bbox")
    args = parser.parse_args(argv)

    roi = resolve_roi(name=args.name or args.roi, bbox=args.bbox)
    print(f"ROI: {roi['name']}  bbox {roi['bbox']}")

    roi_lat, roi_lon = roi_centroid(roi["bbox"])
    print(f"Searching the nearest DAHITI target with data in the period "
          f"{PERIOD_START:%Y-%m-%d} - now, around the ROI centroid "
          f"({roi_lat:.4f}, {roi_lon:.4f}) ...")
    target, raw = pick_target(roi_lat, roi_lon)

    meta, all_rows = parse_series(raw)

    info = target_info(target["id"])
    target_type = info.get("type", "?")
    tlat = info.get("latitude", target.get("latitude"))
    tlon = info.get("longitude", target.get("longitude"))
    name = info.get("target_name", target.get("target_name"))
    country = max((info.get("country") or "", target.get("country") or "",
                   meta.get("country") or ""), key=len) or "?"
    continent = max((info.get("continent") or "", target.get("continent") or "",
                     meta.get("continent") or ""), key=len) or "?"
    roi_dist = haversine_km(roi_lat, roi_lon, tlat, tlon)

    print(f"Selected DAHITI target: {name} (id {target['id']}, {target_type})")
    print(f"  target position : {tlat:.4f}, {tlon:.4f}  ({country}, {continent})")

    areas = considered_areas(roi)
    print("Distance from the considered areas to the DAHITI target:")
    for label, alat, alon in areas:
        print(f"  {label:<28} {haversine_km(alat, alon, tlat, tlon):8.2f} km")

    out_dir = OUT_DIR / slugify(roi["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{target['id']}_water_level.json"
    raw_path.write_text(raw, encoding="utf-8")
    print(f"  data saved: {raw_path}")

    rows, end = clip_period(all_rows)
    if not rows:
        print("  no data points in the period, nothing to plot")
        return

    print("DAHITI source metadata:")
    print(f"  dataset     : {meta['dataset']}  |  {meta['source']} / {meta['institution']}")
    print(f"  instruments : not exposed by the DAHITI API - multi-mission satellite "
          f"altimetry (processing software v{meta['software']})")
    print(f"  target page : {meta['url']}")
    print(f"Series: {len(rows)} points from {rows[0][0]:%Y-%m-%d} to {rows[-1][0]:%Y-%m-%d} "
          f"(min {min(r[1] for r in rows):.2f} / max {max(r[1] for r in rows):.2f} m, "
          f"period {PERIOD_START:%Y-%m-%d} - {end:%Y-%m-%d})")

    plot_series(meta, rows, roi_dist, out_dir / f"water_level_{target['id']}.png")

    info_path = out_dir / "water_level_info.json"
    info_path.write_text(json.dumps({
        "roi": {"name": roi["name"], "bbox": roi["bbox"]},
        "period": {"start": str(PERIOD_START.date()), "end": str(end.date())},
        "nearest": {
            "id": target["id"],
            "name": name,
            "type": target_type,
            "latitude": tlat,
            "longitude": tlon,
            "distance_roi_centroid_km": round(roi_dist, 3),
            "country": country,
            "continent": continent,
        },
        "dahiti_metadata": meta,
        "distances_km": {label: round(haversine_km(alat, alon, tlat, tlon), 3)
                         for label, alat, alon in areas},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  info saved: {info_path}")


if __name__ == "__main__":
    main()
