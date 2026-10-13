import json
import urllib.request
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image

# 1. Load the JSON file
json_path = "/home/samuel/codes/HydroPol/sta_level1c_by_roi.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Get your access token (using your existing functions)
offline_token, client_id, client_secret = load_credentials(CREDENTIALS_PATH)
access_token = get_maap_access_token(offline_token, client_id, client_secret)

# 3. Iterate through ROIs and check product counts
for roi in data.get("rois", []):
    products = roi.get("products", [])
    product_count = len(products)

    print(f"ROI: {roi['name']} | Products found: {product_count}")

    if product_count > 0:
        for first_product in products:
            # first_product = products[0]
            product_date = first_product.get("datetime")
            print(f"  -> First Product Date: {product_date}")

            # Use the Pauli RGB quicklook asset URL from the product
            asset_url = first_product["assets"].get("enclosure_ql_pauli_png")

            if asset_url:
                # 4. Fetch and display the RGB composite image securely with Bearer token
                req = urllib.request.Request(
                    asset_url, headers={"Authorization": f"Bearer {access_token}"}
                )
                with urllib.request.urlopen(req) as response:
                    img = Image.open(BytesIO(response.read()))

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(img)
                ax.set_title(
                    f"ROI: {roi['name']}\nDate: {product_date} (Pauli RGB Composite)"
                )
                ax.axis("off")
                plt.show()
            break


dates = [p.get("datetime") for p in products]
print(f"Total products: {len(dates)}")
print(f"First Date: {dates[0]}")
print(f"Last Date: {dates[-1]}")


import rasterio
from rasterio.session import DummySession

# Identify the first and last product
first_product = products[0]
last_product = products[-1]

# Replace 'data' or 'cog' with the actual asset key for your real data product
# (You can check available keys via: print(first_product['assets'].keys()))
first_asset_url = first_product["assets"]["data"]["href"]  # Example asset key
last_asset_url = last_product["assets"]["data"]["href"]


# Function to load raster data securely with Bearer token
def load_raster_data(url, token):
    with rasterio.Env(
        GDAL_HTTP_HEADER=f"Authorization: Bearer {token}",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    ):
        with rasterio.open(url) as src:
            data = src.read()  # Reads all bands into a NumPy array
            transform = src.transform
            profile = src.profile
            return data, profile, transform


# Load data for the first date
first_data, first_profile, first_transform = load_raster_data(
    first_asset_url, access_token
)
print(f"First dataset shape: {first_data.shape}")

# Load data for the last date
last_data, last_profile, last_transform = load_raster_data(last_asset_url, access_token)
print(f"Last dataset shape: {last_data.shape}")
