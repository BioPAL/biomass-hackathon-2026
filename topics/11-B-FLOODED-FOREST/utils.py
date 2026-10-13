import json
import os
import time
from pathlib import Path

import requests
from pystac_client import Client
from shapely.geometry import box, shape
from shapely.errors import ShapelyError

# -------------------- Configuration --------------------
import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_credentials(path):
    """Read OFFLINE_TOKEN, CLIENT_ID and CLIENT_SECRET from a KEY=value file."""
    values = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    required = ("OFFLINE_TOKEN", "CLIENT_ID", "CLIENT_SECRET")
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError(f"Missing {', '.join(missing)} in credentials file: {path}")

    return values["OFFLINE_TOKEN"], values["CLIENT_ID"], values["CLIENT_SECRET"]


def get_maap_access_token(offline_token, client_id, client_secret):
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": offline_token,
        "scope": "offline_access openid",
    }
    response = requests.post(TOKEN_URL, data=data, timeout=60)
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("MAAP token response did not contain access_token")
    return token


config_path = (
    "/home/samuel/codes/biomass-hackathon-2026/topics/11-B-FLOODED-FOREST/config.yaml"
)

# Usage example:config = load_config(config_path)
config = load_config(config_path)

CREDENTIALS_PATH = config.get("credentials_path")
ROI_JSON_PATH = config.get("roi_json_path")
OUTPUT_JSON_PATH = config.get("output_json_path")
STAC_API_URL = config.get("stac_api_url")
TOKEN_URL = config.get("token_url")


import numpy as np

db_data = lambda x: 10 * np.log10(np.maximum(x, 1e-10))
