#!/usr/bin/env python3
"""Refine ROIs to the intersection of the best ascending/descending C01 frames.

Reads assets/catalogue/sta_level1c_by_roi.json (written by the pipeline search),
downloads the KML footprint of every product, plots them per ROI (colour by
orbit direction), picks the ascending/descending pair with the largest overlap
and stores the intersection as 'roi_refined' (WKT) in the JSON.
Figures go to assets/figures/.
"""

import json
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from shapely.geometry import box

from flooded_forest import auth, config as ffconfig

CATALOGUE_PATH = "sta_level1c_by_roi.json"


def main():
    cfg = ffconfig.load_config()
    catalogue_path = cfg.catalogue_dir / CATALOGUE_PATH
    if not catalogue_path.exists():
        raise SystemExit(f"{catalogue_path} not found — run the pipeline search first")

    data = json.loads(catalogue_path.read_text(encoding="utf-8"))
    token = auth.load_token(cfg)
    figures_dir = cfg.assets_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for roi in data.get("rois", []):
        products = roi.get("products", [])
        name = roi["name"]
        print(f"ROI: {name} | products: {len(products)}")
        if not products:
            continue

        slug = name.replace(" ", "_")[:40]
        fig, ax = plt.subplots(figsize=(10, 10))
        legend_handles = []
        footprints = {}  # product id -> (orbit direction, shapely geometry)

        for product in products:
            asset_url = product["assets"].get("enclosure_itf_map_kml")
            if not asset_url:
                continue
            try:
                # orbit direction from the public STAC item
                item_url = (
                    "https://catalog.maap.eo.esa.int/catalogue/collections/"
                    f"{product['collection']}/items/{product['id']}"
                )
                metadata = json.loads(urllib.request.urlopen(item_url).read().decode())
                orbit = metadata.get("properties", {}).get("sat:orbit_state", "unknown")
                edge_color = {"ascending": "blue", "descending": "orange"}.get(orbit, "gray")

                req = urllib.request.Request(asset_url, headers={"Authorization": f"Bearer {token.get()}"})
                with urllib.request.urlopen(req) as response, tempfile.NamedTemporaryFile(
                    suffix=".kml", delete=False
                ) as tmp:
                    tmp.write(response.read())
                    tmp_path = tmp.name
                geometry = gpd.read_file(tmp_path).geometry.iloc[0]

                footprints[product["id"]] = (orbit, geometry)
                gpd.GeoSeries([geometry]).plot(
                    ax=ax, facecolor="lightgray", edgecolor=edge_color, alpha=0.3, linewidth=1.5
                )
                date = (product.get("datetime") or "?")[:10]
                label = f"{date} ({orbit})"
                if not any(h.get_label() == label for h in legend_handles):
                    legend_handles.append(mpatches.Patch(
                        facecolor="lightgray", edgecolor=edge_color, alpha=0.5, label=label))
            except Exception as exc:
                print(f"  -> error loading KML for {product.get('id')}: {exc}")

        bbox = roi["bbox"]
        roi_gdf = gpd.GeoDataFrame(
            geometry=[box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])], crs="EPSG:4326"
        )
        roi_gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2.5)
        legend_handles.insert(0, mpatches.Patch(
            facecolor="none", edgecolor="red", linewidth=2.5, label=f"ROI: {name}"))

        ax.set_title(f"ROI: {name}\nProduct footprints & ROI boundary", fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{slug}_footprints.png", dpi=300)
        plt.show()
        plt.close(fig)

        # best ascending/descending C01 pair by overlap area
        c01 = {k: v for k, v in footprints.items() if "C01" in k}
        asc = [k for k, v in c01.items() if v[0] == "ascending"]
        desc = [k for k, v in c01.items() if v[0] == "descending"]
        if not (asc and desc):
            print("  -> no ascending/descending C01 pair, skipping refinement")
            continue

        best_pair, max_overlap = None, 0.0
        for a in asc:
            for d in desc:
                overlap = c01[a][1].intersection(c01[d][1]).area
                if overlap > max_overlap:
                    max_overlap, best_pair = overlap, (a, d)
        print(f"  best pair: {best_pair} | overlap: {max_overlap:.4e}")

        roi_polygon = c01[best_pair[0]][1].intersection(c01[best_pair[1]][1])
        roi["roi_refined"] = roi_polygon.wkt

    catalogue_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"JSON updated with 'roi_refined' where possible: {catalogue_path}")


if __name__ == "__main__":
    main()
