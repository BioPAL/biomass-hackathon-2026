"""
Interactive choice of the STA stacks to extract, used by extraction.extract
between the stack search and the download.

- prints one line per stack (not per file) with its index
- writes an interactive folium map (footprints, ROI, quicklook of each stack's
  primary acquisition in the popup) and opens it in the browser
- asks which stacks to keep: "0,2,5-7", "all", or Enter for the recommendation
"""

import sys
import webbrowser
from collections import defaultdict

import folium
from shapely.geometry import box, mapping

from ..core.scenes import parse_selection

ORBIT_COLORS = {"ascending": "#2a78d6", "descending": "#eb6834"}
INK = "#2b2b2a"
ROI_COLOR = "#eda100"  # visible on both the map and the satellite layer


# -------------------- Table --------------------


def print_table(stacks, recommended_ids):
    print()
    print(f"  {'idx':>3}  rec  {'stack':<48} {'files':>5}  {'ROI cov.':>8}")
    for i, st in enumerate(stacks):
        flag = " * " if st["stack_id"] in recommended_ids else "   "
        print(f"  {i:>3}  {flag}  {st['name']:<48} {len(st['items']):>3}/7  {100 * st['stac_coverage']:7.1f}%")
    print("  (* = recommended: best frame per track, min coverage and max files applied)")
    print()


# -------------------- Map --------------------


def _popup_html(i, st, recommended):
    ref = st["reference"]
    quicklook = ref.assets.get("quicklook_jpg")
    dates = ", ".join(it.properties["datetime"][5:10] for it in st["items"])
    img = f'<br><img src="{quicklook.href}" width="280">' if quicklook else ""
    return (
        f"<b>[{i}] {st['name']}</b><br>"
        f"{len(st['items'])} files &middot; ROI coverage {100 * st['stac_coverage']:.1f}%"
        f"{' &middot; <b>recommended</b>' if recommended else ''}<br>"
        f"dates (MM-DD): {dates}<br>"
        f"<small>{st['stack_id']}</small>"
        f"{img}<br><small>quicklook of {ref.id}</small>"
    )


def build_map(stacks, bbox, recommended_ids, path):
    """Folium map of all stack footprints over the ROI, saved as HTML."""
    west, south, east, north = bbox
    # OpenStreetMap blocks tiles for pages opened from a local file, so use Esri tiles
    fmap = folium.Map(location=[(south + north) / 2, (west + east) / 2], tiles=None, zoom_start=8)
    for name, service in (("Map", "World_Topo_Map"), ("Satellite", "World_Imagery")):
        folium.TileLayer(
            f"https://server.arcgisonline.com/ArcGIS/rest/services/{service}/MapServer/tile/{{z}}/{{y}}/{{x}}",
            attr=f"Esri {service.replace('_', ' ')}", name=name,
        ).add_to(fmap)

    for i, st in enumerate(stacks):
        recommended = st["stack_id"] in recommended_ids
        color = ORBIT_COLORS.get(st["orbit"], INK)
        style = {
            "color": color,
            "weight": 3 if recommended else 1.5,
            "dashArray": None if recommended else "6 4",
            "fillOpacity": 0.12 if recommended else 0.03,
        }
        layer = folium.FeatureGroup(name=f"[{i}] {st['name']}{'  *' if recommended else ''}")
        folium.GeoJson(
            mapping(st["stac_footprint"]),
            style_function=lambda _, s=style: s,
            highlight_function=lambda _: {"weight": 5, "fillOpacity": 0.25},
            tooltip=f"[{i}] {st['name']}  ({len(st['items'])} files, {100 * st['stac_coverage']:.0f}% of ROI)",
            popup=folium.Popup(_popup_html(i, st, recommended), max_width=320),
        ).add_to(layer)
        layer.add_to(fmap)

    # stacks of the same track repeat the same footprint: one index label per position
    labels = defaultdict(list)
    for i, st in enumerate(stacks):
        c = st["stac_footprint"].intersection(box(*bbox)).representative_point()
        if c.is_empty:
            c = st["stac_footprint"].representative_point()
        labels[(round(c.y, 2), round(c.x, 2))].append(i)
    for (lat, lon), idx in labels.items():
        text = ", ".join(str(i) for i in idx)
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                icon_anchor=(0, 10),
                html=(f'<div style="font:600 12px sans-serif;color:{INK};background:white;'
                      f'border:1px solid {INK};border-radius:4px;padding:1px 5px;white-space:nowrap;'
                      f'display:inline-block">{text}</div>'),
            ),
        ).add_to(fmap)

    folium.Rectangle(
        bounds=[[south, west], [north, east]], color=ROI_COLOR, weight=3, dash_array="8 6", fill=False, tooltip="ROI",
    ).add_to(fmap)

    legend = f"""
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;
                border:1px solid #c3c2b7;border-radius:6px;font:12px sans-serif;color:{INK}">
      <div><span style="display:inline-block;width:22px;border-top:3px solid {ORBIT_COLORS['ascending']}"></span> ascending</div>
      <div><span style="display:inline-block;width:22px;border-top:3px solid {ORBIT_COLORS['descending']}"></span> descending</div>
      <div><span style="display:inline-block;width:22px;border-top:3px solid {INK}"></span> recommended &nbsp;
           <span style="display:inline-block;width:22px;border-top:2px dashed {INK}"></span> other</div>
      <div><span style="display:inline-block;width:22px;border-top:3px dashed {ROI_COLOR}"></span> ROI &middot; click a footprint for its quicklook</div>
    </div>"""
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(fmap)

    bounds = [st["stac_footprint"].bounds for st in stacks] + [tuple(bbox)]
    fmap.fit_bounds([[min(b[1] for b in bounds), min(b[0] for b in bounds)],
                     [max(b[3] for b in bounds), max(b[2] for b in bounds)]])
    fmap.save(str(path))
    return path


# -------------------- Selection --------------------


def choose_stacks(stacks, recommended, bbox, map_path, preset=None, interactive=True):
    """Return the stacks to extract.

    preset      selection string from the command line, used without asking
    interactive False -> keep the recommendation without asking
    """
    recommended_ids = {st["stack_id"] for st in recommended}
    print_table(stacks, recommended_ids)
    rec_idx = [i for i, st in enumerate(stacks) if st["stack_id"] in recommended_ids]

    if preset is not None:
        return [stacks[i] for i in parse_selection(preset, len(stacks))]
    if not interactive or not sys.stdin.isatty():
        return recommended

    build_map(stacks, bbox, recommended_ids, map_path)
    print(f"Map of the stacks: {map_path}")
    try:
        webbrowser.open(map_path.resolve().as_uri())
    except Exception:
        pass  # no browser (e.g. remote JupyterLab): open the HTML file from the file browser

    default = ",".join(str(i) for i in rec_idx)
    while True:
        answer = input(f"Stacks to keep (e.g. 0,2,5-7 | all | Enter = recommended [{default}]): ").strip()
        try:
            chosen = parse_selection(answer or default, len(stacks))
        except ValueError as exc:
            print(f"  {exc}")
            continue
        if not chosen:
            print("  nothing selected")
            continue
        n_files = sum(len(stacks[i]["items"]) for i in chosen)
        for i in chosen:
            print(f"    [{i}] {stacks[i]['name']}")
        if input(f"  {len(chosen)} stacks, {n_files} files. Extract these? (y/n) [y]: ").strip().lower() in ("", "y", "yes"):
            return [stacks[i] for i in chosen]
