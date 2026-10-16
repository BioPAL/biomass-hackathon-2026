"""MAAP credentials and access token, and the DAHITI API key."""

import os
import threading
import time
from pathlib import Path

import requests

# Public client credentials from the ESA MAAP examples (same for every user)
DEFAULT_CLIENT_ID = "offline-token"
DEFAULT_CLIENT_SECRET = "p1eL7uonXs6MDxtGbgKdPVRAmnGxHpVE"


def read_keyvalue(path):
    """KEY=value pairs of a credentials file; blank lines and #comments ignored."""
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def load_credentials(path):
    """Accept a raw offline-token file or a KEY=value credentials file.

    Returns (offline_token, client_id, client_secret).
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    values = read_keyvalue(path)

    offline = values.get("OFFLINE_TOKEN") or text.replace("\n", "")
    if not offline:
        raise ValueError(f"No OFFLINE_TOKEN found in {path}")
    return (
        offline,
        values.get("CLIENT_ID", DEFAULT_CLIENT_ID),
        values.get("CLIENT_SECRET", DEFAULT_CLIENT_SECRET),
    )


def dahiti_api_key(credentials_path=None):
    """DAHITI API key: DAHITI_API_KEY=... in the credentials file, or the environment.

    Get a key after registering on https://dahiti.dgfi.tum.de/. Never commit it: the
    credentials file is git-ignored (see config.yaml credentials_path).
    """
    if credentials_path and Path(credentials_path).exists():
        key = read_keyvalue(credentials_path).get("DAHITI_API_KEY")
        if key:
            return key
    key = os.environ.get("DAHITI_API_KEY")
    if not key:
        raise ValueError(
            f"No DAHITI_API_KEY in {credentials_path or 'the credentials file'} or in the "
            "environment: add the line DAHITI_API_KEY=... to the credentials file "
            "(dahiti.dgfi.tum.de registration) or export DAHITI_API_KEY"
        )
    return key


class Token:
    """MAAP access token, refreshed shortly before it expires. Thread-safe (parallel downloads)."""

    def __init__(self, config, credentials_path=None):
        offline, client_id, client_secret = load_credentials(credentials_path or config.credentials_path)
        self._offline = offline
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = config.token_url
        self._token = None
        self._expires = 0.0
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if self._token is None or time.time() > self._expires - 60:
                self._refresh()
            return self._token

    def headers(self):
        return {"Authorization": f"Bearer {self.get()}"}

    def _refresh(self):
        response = requests.post(
            self._token_url,
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "grant_type": "refresh_token",
                "refresh_token": self._offline,
                "scope": "offline_access openid",
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("access_token"):
            raise RuntimeError("MAAP token response did not contain access_token")
        self._token = payload["access_token"]
        self._expires = time.time() + float(payload.get("expires_in", 300))


def load_token(config, credentials_path=None):
    """Build a Token from the credentials file configured in config.yaml."""
    return Token(config, credentials_path)
