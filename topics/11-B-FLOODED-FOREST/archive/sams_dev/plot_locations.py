import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

# Load ROIs from JSON
json_path = (
    "/home/samuel/codes/biomass-hackathon-2026/topics/11-B-FLOODED-FOREST/rois.json"
)
with open(json_path, "r") as f:
    rois = json.load(f)

# Initialize figure with smaller size and higher DPI
fig = plt.figure(figsize=(8, 6), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Crop extent to -30 to 30
ax.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.LAND, facecolor="#f2f2f2")
ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")
ax.add_feature(cfeature.COASTLINE, edgecolor="gray", linewidth=0.6)
ax.add_feature(cfeature.RIVERS, edgecolor="#3399ff", linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray", linewidth=0.5)

# Plot each ROI bounding box and center point
for roi in rois:
    bbox = roi["bbox"]
    west, south, east, north = (
        bbox["west"],
        bbox["south"],
        bbox["east"],
        bbox["north"],
    )

    width = east - west
    height = north - south
    center_lon = west + width / 2
    center_lat = south + height / 2

    # Bounding box with thicker lines
    rect = Rectangle(
        (west, south),
        width,
        height,
        transform=ccrs.PlateCarree(),
        facecolor="red",
        edgecolor="darkred",
        alpha=0.6,
        linewidth=2.0,
    )
    ax.add_patch(rect)

    # Larger marker for the location center
    ax.plot(
        center_lon,
        center_lat,
        marker="o",
        color="darkred",
        markersize=6,
        transform=ccrs.PlateCarree(),
    )

    # # Label the ROI (no bold)
    # ax.text(
    #     east + 0.5,
    #     north,
    #     roi["name"],
    #     transform=ccrs.PlateCarree(),
    #     fontsize=7,
    #     color="black",
    #     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    # )

ax.set_title("Regions of Interest Map", fontsize=10)
plt.tight_layout()
plt.show()