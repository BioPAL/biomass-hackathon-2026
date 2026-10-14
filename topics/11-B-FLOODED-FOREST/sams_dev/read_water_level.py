import io
import matplotlib.pyplot as plt
import pandas as pd
import requests
from shapely import wkt

# 1. Setup API and ROI
API_KEY = "589AD7A19BAE1CBA8FCE33756EBD90BD124ECB2A2F4E638BFADD12C6C4DE84C4"

import io
import json
import matplotlib.pyplot as plt
import pandas as pd
import requests
from shapely import wkt

BASE_URL = "https://dahiti.dgfi.tum.de/api/v2/"

# 1. Setup ROI and find Centroid
roi_wkt = "POLYGON((-62.0 -4.0, -59.0 -4.0, -59.0 -2.0, -62.0 -2.0, -62.0 -4.0))"
polygon = wkt.loads(roi_wkt)
centroid = polygon.centroid

# 2. Get Nearest Target (uses form-encoded POST)
target_url = f"{BASE_URL}get-nearest-target/"
target_args = {
    "api_key": API_KEY,
    "longitude": centroid.x,
    "latitude": centroid.y,
}

response = requests.post(target_url, data=target_args)
response.raise_for_status()
target_data = json.loads(response.text)
dahiti_id = target_data.get("dahiti_id")
print(f"Found DAHITI ID: {dahiti_id} {target_data.get('data')['target_name']}")



wl_url = f"{BASE_URL}download-water-level/"
wl_args = {"api_key": API_KEY, "dahiti_id": dahiti_id}

wl_response = requests.post(wl_url, data=wl_args)
wl_response.raise_for_status()

# Parse JSON data into DataFrame
data_json = json.loads(wl_response.text)
df_wl = pd.DataFrame(data_json)

# Filter for data since 2024
df_wl["date"] = pd.to_datetime(df_wl["date"])
df_wl = df_wl[df_wl["date"] >= "2024-01-01"]

# 4. Download Water Occurrence Mask (using JSON POST pattern)
mask_url = f"{BASE_URL}download-water-occurrence-mask/"
mask_args = {"api_key": API_KEY, "dahiti_id": dahiti_id}
mask_response = requests.post(mask_url, json=mask_args)
# mask_response.content contains the raster bytes if available

# 5. Generate Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time Series Plot
axes[0].plot(
    df_wl["date"],
    df_wl["height"],
    marker="o",
    linestyle="-",
    color="royalblue",
)
axes[0].set_title(f"Water Level Time Series (ID: {dahiti_id})")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Water Level (m)")
axes[0].grid(True)
axes[0].tick_params(axis="x", rotation=45)

# Spatial Plot Placeholder
axes[1].set_title("Spatial ROI / Water Mask")
axes[1].axis("off")

plt.tight_layout()
plt.show()