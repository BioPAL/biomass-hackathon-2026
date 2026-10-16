#!/usr/bin/env python3
"""DAHITI water level of the target nearest to a ROI (pipeline step 10).

For the ROI the pipeline works on (preset name from rois.json, 4 coordinates or a
prompt) it:

1. finds the DAHITI water-level targets nearest to the ROI centroid
   (get-nearest-target, great-circle distance in km) and keeps the first one that
   has data inside the pipeline period (--start / --end, default: today),
2. prints the distance from every considered area (the ROI, its common area and
   the extracted stacks, when they exist under outputs/<roi>/extraction/) to that
   target,
3. plots every point of the target since --since (default: water_level_since in
   config.yaml, 2024-01-01) and marks the dates of the extracted stack scenes as
   faint vertical lines,
4. writes the raw data and an info JSON to outputs/<roi>/water_level/.

Outputs
-------
    outputs/<roi>/extraction/figures/water_level.png   the figure (next to timeseries.png)
    outputs/<roi>/water_level/water_level_<id>.json    raw DAHITI download (json format)
    outputs/<roi>/water_level/water_level.json         roi, period, target, distances, metadata

Standalone examples
-------------------
    python tools/water_level_roi.py --roi bolobo --start 2026-05-01
    python tools/water_level_roi.py --bbox 16.6 -2.4 16.8 -2.2 --name "Bolobo site" --start 2026-05-01

In the pipeline: run_pipeline.py step 10 (skip it with --skip-water-level). The API
key is read from the credentials file (DAHITI_API_KEY=..., see config.yaml
credentials_path) or from the DAHITI_API_KEY environment variable. Never commit it.

DAHITI API: https://dahiti.dgfi.tum.de/en/api/doc/v2/
"""

import argparse
import json
import math
import sys
from datetime import date, datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import requests
from matplotlib.lines import Line2D

from flooded_forest.core import auth, config as ffconfig, roi as roi_mod
from flooded_forest.core.io import load_stack_info
from flooded_forest.plotting.common import save
from flooded_forest.plotting.stacks import STACK_COLORS

BASE_URL = "https://dahiti.dgfi.tum.de/api/v2/"
SERIES_COLOR = "#2a78d6"   # DAHITI water level (same colour as the prototype in archive/sams_dev)
SCENE_ALPHA = 0.35         # faintness of the extracted-scene vertical lines


# ------------------------- ROI helpers -------------------------

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


def as_datetime(value, label):
    """datetime of a YYYY-MM-DD string (or of a date / datetime object)."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.max if label == "end" else time.min)
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        raise SystemExit(f"invalid {label} date {value!r}: expected YYYY-MM-DD")


def considered_areas(paths, roi):
    """[(label, lat, lon)] of the ROI, its common area and its extracted stacks."""
    areas = [("ROI centroid", *roi_centroid(roi["bbox"]))]

    common_path = paths.extraction / "common_area.geojson"
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

    for latlon in sorted(paths.extraction.glob("*/latlon.tif")):
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


def stack_scenes(paths):
    """{stack label: [scene datetimes]} of the extracted stacks, chronological per stack."""
    scenes = {}
    if not paths.extraction.exists():
        return scenes
    for stack_dir in paths.stack_dirs():
        dates = [scene_datetime(p["datetime"]) for p in load_stack_info(stack_dir)["products"]]
        if dates:
            label = stack_dir.name.split("_2")[0]  # STA1_descending_20260916-... -> STA1_descending
            scenes[label] = sorted(dates)
    return scenes


# ------------------------- DAHITI API -------------------------

def _post(api_key, endpoint, **params):
    params["api_key"] = api_key
    response = requests.post(BASE_URL + endpoint + "/", data=params, timeout=180)
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") not in (200, None):
        raise RuntimeError(f"{endpoint}: {payload.get('message')}")
    return payload


def nearest_target(api_key, lat, lon):
    """Nearest DAHITI target: {id, target_name, latitude, longitude, distance, ...}."""
    return _post(api_key, "get-nearest-target", latitude=lat, longitude=lon)["data"]


def all_targets(api_key):
    """Every DAHITI target: {dahiti_id, target_name, latitude, longitude, country, ...}."""
    return _post(api_key, "list-targets")["data"]


def targets_by_distance(api_key, lat, lon, skip_id, limit):
    """The `limit` DAHITI targets closest to (lat, lon), `skip_id` excluded."""
    ranked = sorted(
        (t for t in all_targets(api_key) if t["dahiti_id"] != skip_id),
        key=lambda t: haversine_km(lat, lon, t["latitude"], t["longitude"]),
    )
    return [{"id": t["dahiti_id"], "target_name": t["target_name"],
             "latitude": t["latitude"], "longitude": t["longitude"],
             "country": t.get("country"), "continent": t.get("continent"),
             "distance": haversine_km(lat, lon, t["latitude"], t["longitude"])}
            for t in ranked[:limit]]


def target_info(api_key, dahiti_id):
    """Type, position and names of one target ({} when the request fails)."""
    try:
        return _post(api_key, "get-target-info", dahiti_id=dahiti_id)["data"]
    except Exception:
        return {}


def download_series(api_key, dahiti_id):
    """Raw water-level download of one target (json format)."""
    response = requests.post(
        BASE_URL + "download-water-level/",
        data={"api_key": api_key, "dahiti_id": dahiti_id, "format": "json"},
        timeout=180,
    )
    response.raise_for_status()
    return response.text


def parse_series(text):
    """(metadata, rows) from a JSON download; rows are (datetime, level, error)."""
    payload = json.loads(text)
    meta = {key: payload.get(key) for key in
            ("dahiti_id", "target_name", "dataset", "institution", "source", "software",
             "url", "creation_date", "longitude", "latitude", "country", "continent")}
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


def pick_target(api_key, lat, lon, start, end, max_tries):
    """Nearest DAHITI target with data inside [start, end], and its raw series.

    get-nearest-target gives the closest target; when its series has no data in
    the period, the next nearest targets (list-targets ranked by distance) are
    tried until one has data.
    """
    def attempts(first):
        yield first
        for candidate in targets_by_distance(api_key, lat, lon, first["id"], max_tries):
            yield candidate

    previous_ids = set()
    for target in attempts(nearest_target(api_key, lat, lon)):
        if target["id"] in previous_ids:
            continue
        previous_ids.add(target["id"])
        print(f"  trying {target['target_name']} (id {target['id']}, "
              f"{target['distance']:.1f} km) ...")
        raw = download_series(api_key, target["id"])
        rows = parse_series(raw)[1]
        if any(start <= row[0] <= end for row in rows):
            return target, raw
        last = f"{rows[-1][0]:%Y-%m-%d}" if rows else "empty"
        print(f"    no data in the period (series ends {last})")
    raise RuntimeError(f"no DAHITI target with data in the period among the "
                       f"{len(previous_ids)} nearest targets")


def scene_datetime(raw):
    """datetime of a stack_info.json product ('2026-09-16T17:02:43...'), naive UTC."""
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).replace(tzinfo=None)


# ------------------------- plot -------------------------

def plot_series(meta, rows, since, roi_dist, scenes, out_dir, show=False):
    """Water level since `since`, with the extracted scene dates as faint vertical lines."""
    dates = [row[0] for row in rows]
    levels = [row[1] for row in rows]
    errors = [0.0 if row[2] is None else row[2] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(dates, levels, yerr=errors, fmt="-o", ms=4, lw=1.2, color=SERIES_COLOR,
                ecolor="#c3c2b7", elinewidth=0.8, capsize=2, label="DAHITI water level")

    handles = [Line2D([], [], color=SERIES_COLOR, marker="o", ms=4, lw=1.2,
                      label="DAHITI water level")]
    n_lines = 0
    for i, (label, scene_dates) in enumerate(sorted(scenes.items())):
        color = STACK_COLORS[i % len(STACK_COLORS)]
        for when in scene_dates:
            ax.axvline(when, color=color, lw=0.6, alpha=SCENE_ALPHA, zorder=0)
        n_lines += len(scene_dates)
        handles.append(Line2D([], [], color=color, lw=0.6, alpha=0.6,
                              label=f"{label} scenes ({len(scene_dates)})"))
    all_dates = dates + [when for scene_dates in scenes.values() for when in scene_dates]
    ax.set_xlim(since, max(all_dates))
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=False)
    ax.set_xlabel("Date")
    ax.set_ylabel("Water level [m]")
    ax.set_title(f"{meta['target_name']} (DAHITI {meta['dahiti_id']}) - "
                 f"{roi_dist:.1f} km from the ROI centroid")
    ax.grid(color="#d9d8d2", lw=0.5)
    ax.set_axisbelow(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    if n_lines:
        print(f"  {n_lines} extracted scene dates marked as faint vertical lines "
              f"({len(scenes)} stacks)")
    else:
        print("  no extracted stacks found: no scene-date lines on the figure")
    return save(fig, out_dir, "water_level", show)


# ------------------------- pipeline entry point -------------------------

def run_water_level(cfg, roi, paths, start, end, api_key=None, since=None, show=False, max_tries=10):
    """DAHITI water-level figure + data of the target nearest to a pipeline ROI.

    The target is the nearest one with data inside the pipeline period [start, end];
    the figure plots every point since `since` (default: cfg.water_level_since) and
    marks the extracted scene dates as faint vertical lines. The figure goes to
    paths.figures (with the other ROI figures), the raw series and the info JSON to
    paths.water_level.

    Returns the figure path, or None when the key is missing, the API fails or no
    target has data in the period (never raises, so the pipeline keeps going).
    """
    try:
        api_key = api_key or auth.dahiti_api_key(cfg.credentials_path)
        start_dt = as_datetime(start, "--start")
        end_dt = as_datetime(end or date.today(), "--end")
        since_dt = as_datetime(since or cfg.water_level_since, "--since")

        lat, lon = roi_centroid(roi["bbox"])
        print(f"  searching the nearest DAHITI target with data in the period "
              f"{start_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d} around the ROI centroid "
              f"({lat:.4f}, {lon:.4f}) ...")
        target, raw = pick_target(api_key, lat, lon, start_dt, end_dt, max_tries)

        meta, all_rows = parse_series(raw)
        info = target_info(api_key, target["id"])
        target_type = info.get("type", "?")
        tlat = info.get("latitude", target.get("latitude"))
        tlon = info.get("longitude", target.get("longitude"))
        name = info.get("target_name", target.get("target_name"))
        country = max((info.get("country") or "", target.get("country") or "",
                       meta.get("country") or ""), key=len) or "?"
        continent = max((info.get("continent") or "", target.get("continent") or "",
                         meta.get("continent") or ""), key=len) or "?"
        roi_dist = haversine_km(lat, lon, tlat, tlon)

        print(f"  selected DAHITI target: {name} (id {target['id']}, {target_type})")
        print(f"    target position : {tlat:.4f}, {tlon:.4f}  ({country}, {continent})")
        print(f"  DAHITI source metadata: {meta['dataset']}  |  "
              f"{meta['source']} / {meta['institution']}  |  {meta['url']}")

        areas = considered_areas(paths, roi)
        print("  distance from the considered areas to the DAHITI target:")
        for label, alat, alon in areas:
            print(f"    {label:<28} {haversine_km(alat, alon, tlat, tlon):8.2f} km")

        paths.water_level.mkdir(parents=True, exist_ok=True)
        raw_path = paths.water_level / f"water_level_{target['id']}.json"
        raw_path.write_text(raw, encoding="utf-8")
        print(f"    data saved: {raw_path}")

        rows = [row for row in all_rows if row[0] >= since_dt]
        if not rows:
            print(f"  no data points since {since_dt:%Y-%m-%d}, nothing to plot")
            return None

        scenes = stack_scenes(paths)
        print(f"  series: {len(rows)} points from {rows[0][0]:%Y-%m-%d} to {rows[-1][0]:%Y-%m-%d} "
              f"(min {min(r[1] for r in rows):.2f} / max {max(r[1] for r in rows):.2f} m, "
              f"since {since_dt:%Y-%m-%d})")
        figure = plot_series(meta, rows, since_dt, roi_dist, scenes, paths.figures, show)

        info_path = paths.water_level / "water_level.json"
        info_path.write_text(json.dumps({
            "roi": {"name": roi["name"], "bbox": roi["bbox"]},
            "period": {
                "since": f"{since_dt:%Y-%m-%d}",
                "pipeline": {"start": f"{start_dt:%Y-%m-%d}", "end": f"{end_dt:%Y-%m-%d}"},
                "plotted": {"first": f"{rows[0][0]:%Y-%m-%d}", "last": f"{rows[-1][0]:%Y-%m-%d}"},
            },
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
            "stack_scenes": {label: [f"{when:%Y-%m-%d}" for when in dates]
                             for label, dates in sorted(scenes.items())},
            "figure": str(figure),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"    info saved: {info_path}")
        return figure
    except Exception as exc:
        print(f"  WARNING: water level figure skipped ({exc})")
        return None


# ------------------------- main -------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roi", help="ROI name from rois.json (substring match, see --list-rois)")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--name", help="name of a custom --bbox (also the output folder name)")
    parser.add_argument("--list-rois", action="store_true", help="print the preset ROIs and exit")
    parser.add_argument("--start", help="pipeline period start YYYY-MM-DD "
                                        "(the target must have data in it; default: --since)")
    parser.add_argument("--end", help="pipeline period end YYYY-MM-DD (default: today)")
    parser.add_argument("--since", help="plot every point since this date YYYY-MM-DD "
                                        "(default: water_level_since in config.yaml)")
    parser.add_argument("--max-tries", type=int, default=10,
                        help="DAHITI targets tried at most to find data in the period")
    parser.add_argument("--api-key", help="DAHITI API key (default: DAHITI_API_KEY in the "
                                          "credentials file or the environment)")
    parser.add_argument("--show", action="store_true", help="display the figure interactively")
    parser.add_argument("--config", help="alternative config.yaml")
    args = parser.parse_args(argv)

    cfg = ffconfig.load_config(args.config)
    cfg.ensure_dirs()
    if args.list_rois:
        roi_mod.list_rois(roi_mod.load_rois(cfg.roi_json_path))
        return
    if args.roi or args.bbox:
        roi = roi_mod.resolve_roi(name=args.name or args.roi, bbox=args.bbox,
                                  roi_json_path=cfg.roi_json_path)
    else:
        roi = roi_mod.resolve_roi(roi_json_path=cfg.roi_json_path)
    paths = cfg.roi_paths(roi["slug"])
    print(f"ROI: {roi['name']}  bbox {roi['bbox']}  ->  {paths.root}")

    figure = run_water_level(cfg, roi, paths, args.start or args.since or cfg.water_level_since,
                             args.end, api_key=args.api_key, since=args.since,
                             show=args.show, max_tries=args.max_tries)
    if figure is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
