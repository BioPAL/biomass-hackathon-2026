import tempfile
import json
import urllib.request
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import box
import xml.etree.ElementTree as ET

# 1. Load the JSON file
json_path = "./sta_level1c_by_roi.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Get your access token
offline_token, client_id, client_secret = load_credentials(CREDENTIALS_PATH)
access_token = get_maap_access_token(offline_token, client_id, client_secret)

# 3. Iterate through ROIs and plot products vs ROI boundary (No Backgrounds)
for roi in data.get("rois", []):
    products = roi.get("products", [])
    product_count = len(products)
    roi_name = roi["name"]

    print(f"ROI: {roi_name} | Products found: {product_count}")
    if product_count == 0:
        continue

    # Create the plot for the current ROI
    fig, ax = plt.subplots(figsize=(10, 10))
    legend_handles = []

    local_products = {}
    # Fetch and plot each product's KML footprint
    for product in products:
        asset_url = product["assets"].get("enclosure_itf_map_kml")
        if not asset_url:
            continue

        url = (
            "https://catalog.maap.eo.esa.int/catalogue/collections/"
            f"BiomassLevel1c/items/{product['id']}"
        )

        response = urllib.request.urlopen(url)  # Note: kept your request pattern
        metadata = json.loads(response.read().decode())

        orbit_direction = metadata.get("properties", {}).get(
            "sat:orbit_state", "unknown"
        )

        # Map orbit direction to boundary color
        if orbit_direction == "ascending":
            edge_color = "blue"
        elif orbit_direction == "descending":
            edge_color = "orange"
        else:
            edge_color = "gray"

        # Extract date and format label
        date_str = product.get("datetime", "Unknown Date")[:10]

        try:
            # Securely download KML file with Bearer token
            req = urllib.request.Request(
                asset_url, headers={"Authorization": f"Bearer {access_token}"}
            )
            with urllib.request.urlopen(req) as response, tempfile.NamedTemporaryFile(
                suffix=".kml", delete=False
            ) as tmp:
                tmp.write(response.read())
                tmp_path = tmp.name

            # Read KML geometry using GeoPandas (native EPSG:4326)
            prod_gdf = gpd.read_file(tmp_path)
            prod_gdf.plot(
                ax=ax,
                facecolor="lightgray",
                edgecolor=edge_color,
                alpha=0.3,
                linewidth=1.5,
            )
            local_products[product["id"]] = (orbit_direction, prod_gdf)
            # Include orbit direction in legend label to prevent duplicates per type/date
            label = f"{date_str} ({orbit_direction})"
            if not any(h.get_label() == label for h in legend_handles):
                patch = mpatches.Patch(
                    facecolor="lightgray", edgecolor=edge_color, alpha=0.5, label=label
                )
                legend_handles.append(patch)

        except Exception as e:
            print(f"  -> Error loading KML for product {product.get('id')}: {e}")

    # Plot ROI boundary LAST so it appears ON TOP of footprints
    bbox = roi["bbox"]
    roi_geom = box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    roi_gdf = gpd.GeoDataFrame(geometry=[roi_geom], crs="EPSG:4326")
    roi_gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2.5)

    # Add ROI to the top of the legend handles
    roi_patch = mpatches.Patch(
        facecolor="none", edgecolor="red", linewidth=2.5, label=f"ROI: {roi_name}"
    )
    legend_handles.insert(0, roi_patch)

    ax.set_title(f"ROI: {roi_name}\nProduct Footprints & ROI Boundary", fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Configure legend and grid
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{roi_name}_map.png", dpi=300)
    plt.show()
    # break

    # take the ascending and descending with the highest overlap

    sel_local_products_keys = [k for k in local_products.keys() if "C01" in k]
    from shapely import wkt

    max_overlap = 0.0
    best_pair = None

    # Separate keys by orbit direction
    asc_keys = [
        k for k in sel_local_products_keys if local_products[k][0] == "ascending"
    ]
    desc_keys = [
        k for k in sel_local_products_keys if local_products[k][0] == "descending"
    ]

    # Find the pair with the maximum intersection area
    for asc in asc_keys:
        poly_asc = local_products[asc][1]
        if isinstance(poly_asc, str):
            poly_asc = wkt.loads(poly_asc)

        for desc in desc_keys:
            poly_desc = local_products[desc][1]
            if isinstance(poly_desc, str):
                poly_desc = wkt.loads(poly_desc)

            overlap = poly_asc.intersection(poly_desc).area[0]
            if overlap > max_overlap:
                max_overlap = overlap
                best_pair = (asc, desc)

    print(f"Best Pair (Ascending, Descending): {best_pair}")
    print(f"Maximum Overlap Area: {max_overlap}")

    # Retrieve geometries for the best pair
    best_asc, best_desc = best_pair

    poly_asc = local_products[best_asc][1]
    if isinstance(poly_asc, str):
        poly_asc = wkt.loads(poly_asc)

    poly_desc = local_products[best_desc][1]
    if isinstance(poly_desc, str):
        poly_desc = wkt.loads(poly_desc)

    # Define the ROI as the intersection
    roi_polygon = poly_asc.intersection(poly_desc)

    # If the intersection result is a GeoSeries, extract the first element
    if hasattr(roi_polygon, "iloc"):
        roi_polygon = roi_polygon.iloc[0]

    # Now you can get WKT and GeoJSON
    roi_wkt = roi_polygon.wkt
    roi_geojson = roi_polygon.__geo_interface__

    print("ROI WKT:", roi_wkt)
    roi["roi_refined"] = roi_wkt


with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("JSON successfully updated with the refined ROI field for all ROIs.")
