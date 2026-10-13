"""Authentication against the ESA MAAP IAM using an offline (refresh) token."""
import os
import time
from pathlib import Path

import requests

from . import config
from .exceptions import AuthError


class TokenProvider:
    """Exchanges the offline token in ``token.txt`` for short-lived access tokens.

    The access token is cached and renewed only when it is about to expire.
    The token file is looked up in ``token_file``, then ``$BIOMASS_TOKEN_FILE``,
    then ``token.txt`` in the current folder or any parent folder.
    """

    EXPIRY_MARGIN = 30  # seconds

    def __init__(self, token_file=None, session=None):
        self.token_file = self._find_token_file(token_file)
        self._session = session or requests.Session()
        self._access_token = None
        self._expires_at = 0.0

    @staticmethod
    def _find_token_file(token_file):
        explicit = token_file or os.environ.get(config.TOKEN_FILE_ENV)
        if explicit:
            path = Path(explicit)
            if not path.is_file():
                raise AuthError(f"Token file not found: {path.resolve()}")
            return path
        cwd = Path.cwd()
        for folder in (cwd, *cwd.parents):
            candidate = folder / config.TOKEN_FILENAME
            if candidate.is_file():
                return candidate
        raise AuthError(f"No {config.TOKEN_FILENAME} found in {cwd} or its parents")

    def access_token(self) -> str:
        if self._access_token and time.time() < self._expires_at - self.EXPIRY_MARGIN:
            return self._access_token
        return self._refresh()

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.access_token()}"}

    def _refresh(self) -> str:
        offline = self.token_file.read_text().strip().replace("\n", "")
        if not offline:
            raise AuthError(f"{self.token_file} is empty")
        response = self._session.post(config.IAM_URL, data={
            "client_id": config.CLIENT_ID,
            "client_secret": config.CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": offline,
            "scope": "offline_access openid",
        }, timeout=60)
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            raise AuthError(f"IAM rejected the token (it may have expired, 90 days): {e}") from e
        payload = response.json()
        token = payload.get("access_token")
        if not token:
            raise AuthError("IAM response did not contain an access_token")
        self._access_token = token
        self._expires_at = time.time() + float(payload.get("expires_in", 60))
        return token
