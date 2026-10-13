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
