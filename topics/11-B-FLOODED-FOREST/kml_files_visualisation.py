from pathlib import Path
import xml.etree.ElementTree as ET
import plotly.graph_objects as go

# --------------------------------------------------
# Configuration
# --------------------------------------------------

KML_FILES = [
    r"/mnt/c/Users/sam_s/Downloads/BIO_OPER_TOM-GC1__MajorCycle2.kml",
    r"/mnt/c/Users/sam_s/Downloads/BIO_OPER_TOM-GC1__MajorCycle1(1).kml",
    r"/mnt/c/Users/sam_s/Downloads/BIO_OPER_TOM-GC1__MajorCycle3-v2.kml",
    # r"/mnt/c/Users/sam_s/Downloads/BIO_OPER_TOM-GC1__MajorCycle4-v2.kml",
    # r"/mnt/c/Users/sam_s/Downloads/BIO_OPER_TOM-GC1__MajorCycle5-v2.kml",
]

COLORS = [
    "#ff4d4d",
    "#4da6ff",
    "#33cc99",
    "#ffcc33",
    "#cc66ff",
]

# Keep 1 out of every N footprints to ensure fast rendering
SAMPLE_EVERY = 5

OUTPUT_HTML = "simplified_major_cycles_globe.html"


# --------------------------------------------------
# Clean Coordinate Extraction
# --------------------------------------------------


def extract_footprints(kml_path):
    tree = ET.parse(kml_path)
    root = tree.getroot()

    geometries = []

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag in {"Polygon", "LineString", "LinearRing"}:
            coords = []
            # Grab only the first coordinate block (outer boundary) to prevent nesting duplication
            for child in elem.iter():
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
                    break  # Stop after outer boundary
            if coords:
                geometries.append((tag, coords))

    return geometries


# --------------------------------------------------
# Main Plotting Logic
# --------------------------------------------------

fig = go.Figure()

for idx, kml_path in enumerate(KML_FILES):
    path_obj = Path(kml_path)
    if not path_obj.exists():
        print(f"File not found: {kml_path}")
        continue

    print(f"Processing: {path_obj.name}...")
    geometries = extract_footprints(kml_path)
    print(f"  Extracted {len(geometries):,} footprints.")

    # Downsample
    sampled = geometries[::SAMPLE_EVERY]
    color = COLORS[idx % len(COLORS)]

    for i, (geom_type, coords) in enumerate(sampled):
        lon = [p[0] for p in coords]
        lat = [p[1] for p in coords]
        is_polygon = geom_type in {"Polygon", "LinearRing"}

        fig.add_trace(
            go.Scattergeo(
                lon=lon,
                lat=lat,
                mode="lines",
                # fill="toself" if is_polygon else None,
                fillcolor=color,
                opacity=0.35,
                name=f"{path_obj.name} - {i}",
                showlegend=(i == 0),  # Show legend once per file
                line=dict(width=1, color=color),
            )
        )

fig.update_layout(
    title="Major Cycles Globe Footprints",
    height=850,
    template="plotly_dark",
    geo=dict(
        projection=dict(type="orthographic"),
        showland=True,
        showocean=True,
        # showcountries=True,
        showcoastlines=True,
        landcolor="rgb(35, 35, 35)",
        oceancolor="rgb(15, 15, 15)",
        countrycolor="rgb(60, 60, 60)",
        coastlinecolor="rgb(80, 80, 80)",
    ),
)

fig.write_html(OUTPUT_HTML)
print(f"Saved visualization to {OUTPUT_HTML}")
fig.show()
