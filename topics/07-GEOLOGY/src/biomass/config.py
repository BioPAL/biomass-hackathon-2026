"""Constants for the ESA MAAP services."""
from pathlib import Path

CATALOG_URL = "https://catalog.maap.eo.esa.int/catalogue/"
IAM_URL = "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token"

# Public client credentials published by ESA in its example notebook (not the user's token).
CLIENT_ID = "offline-token"
CLIENT_SECRET = "p1eL7uonXs6MDxtGbgKdPVRAmnGxHpVE"

TOKEN_FILENAME = "token.txt"
TOKEN_FILE_ENV = "BIOMASS_TOKEN_FILE"
DEFAULT_OUTPUT_DIR = Path("data")

REQUEST_TIMEOUT = 120

# Copernicus Data Space (Sentinel-1 and Sentinel-2 files; the MAAP catalogue lists them)
COPERNICUS_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
COPERNICUS_CLIENT_ID = "cdse-public"
COPERNICUS_USER_ENV = "COPERNICUS_USER"
COPERNICUS_PASSWORD_ENV = "COPERNICUS_PASSWORD"
COPERNICUS_CREDENTIALS_FILENAME = "copernicus.txt"
COPERNICUS_CREDENTIALS_FILE_ENV = "COPERNICUS_CREDENTIALS_FILE"

# NASA Earthdata (NISAR products, distributed by the Alaska Satellite Facility)
NASA_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/ASF"
NASA_CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
NASA_TOKEN_FILENAME = "nasa_token.txt"
NASA_TOKEN_FILE_ENV = "NASA_TOKEN_FILE"
