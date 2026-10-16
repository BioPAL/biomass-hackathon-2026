#!/usr/bin/env python3
"""Visualise BIOMASS major-cycle KML footprints on a globe (Plotly)."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go

from flooded_forest.core import config as ffconfig

# Default: KML files downloaded from https://explorer.maap.eo.esa.int/ (or pass paths)
DEFAULT_KML_FILES = [
    r"/mnt/c/Users/sam_s/Downloads/BIO_OPER_TOM-GC1__MajorCycle2.kml",
    r"/mnt/c/Users/sam_s/Downloads/BIO_OPER_TOM-GC1__MajorCycle1(1).kml",
    r"/mnt/c/Users/sam_s/Downloads/BIO_OPER_TOM-GC1__MajorCycle3-v2.kml",
]

COLORS = ["#ff4d4d", "#4da6ff", "#33cc99", "#ffcc33", "#cc66ff"]
SAMPLE_EVERY = 5  # keep 1 out of every N footprints for fast rendering


def extract_footprints(kml_path):
    tree = ET.parse(kml_path)
    geometries = []
    for elem in tree.getroot().iter():
        tag = elem.tag.split("}")[-1]
        if tag not in {"Polygon", "LineString", "LinearRing"}:
            continue
        coords = []
        for child in elem.iter():  # first coordinate block = outer boundary
            if child.tag.split("}")[-1] == "coordinates" and child.text:
                for item in child.text.split():
                    vals = item.split(",")
                    if len(vals) >= 2:
                        try:
                            lon, lat = float(vals[0]), float(vals[1])
                            if -180 <= lon <= 180 and -90 <= lat <= 90:
                                coords.append((lon, lat))
                        except ValueError:
                            pass
                break
        if coords:
            geometries.append((tag, coords))
    return geometries


def main(kml_files=None, output_name="simplified_major_cycles_globe.html"):
    cfg = ffconfig.load_config()
    figures_dir = cfg.outputs_dir / "tools"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / output_name

    fig = go.Figure()
    for idx, kml_path in enumerate(kml_files or DEFAULT_KML_FILES):
        path = Path(kml_path)
        if not path.exists():
            print(f"File not found: {kml_path}")
            continue
        print(f"Processing: {path.name}...")
        geometries = extract_footprints(path)
        print(f"  Extracted {len(geometries):,} footprints.")
        color = COLORS[idx % len(COLORS)]

        for i, (geom_type, coords) in enumerate(geometries[::SAMPLE_EVERY]):
            lon = [p[0] for p in coords]
            lat = [p[1] for p in coords]
            fig.add_trace(go.Scattergeo(
                lon=lon, lat=lat, mode="lines", opacity=0.35,
                name=f"{path.name} - {i}", showlegend=(i == 0),
                line=dict(width=1, color=color),
            ))

    fig.update_layout(
        title="Major Cycles Globe Footprints",
        height=850,
        template="plotly_dark",
        geo=dict(
            projection=dict(type="orthographic"),
            showland=True, showocean=True, showcoastlines=True,
            landcolor="rgb(35, 35, 35)", oceancolor="rgb(15, 15, 15)",
            coastlinecolor="rgb(80, 80, 80)",
        ),
    )
    fig.write_html(out_path)
    print(f"Saved visualization to {out_path}")
    fig.show()


if __name__ == "__main__":
    main(kml_files=sys.argv[1:] or None)
