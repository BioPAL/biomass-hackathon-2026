"""Authentication against the ESA MAAP IAM using an offline (refresh) token."""
import base64
import getpass
import json
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
    def _find_token_file(token_file, env_var=config.TOKEN_FILE_ENV, filename=config.TOKEN_FILENAME):
        explicit = token_file or os.environ.get(env_var)
        if explicit:
            path = Path(explicit)
            if not path.is_file():
                raise AuthError(f"Token file not found: {path.resolve()}")
            return path
        cwd = Path.cwd()
        for folder in (cwd, *cwd.parents):
            candidate = folder / filename
            if candidate.is_file():
                return candidate
        raise AuthError(f"No {filename} found in {cwd} or its parents")

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


class CopernicusAuth:
    """Copernicus Data Space account (username and password), for the Sentinel-1 and
    Sentinel-2 files.

    Credentials are taken from the arguments, then ``$COPERNICUS_USER`` and
    ``$COPERNICUS_PASSWORD``, then a ``copernicus.txt`` file (e-mail on the first line,
    password on the second) found like ``token.txt``, or at ``credentials_file`` /
    ``$COPERNICUS_CREDENTIALS_FILE``. Otherwise they are asked for the first time a
    download needs them. Access tokens last a few minutes and are renewed as needed.
    """

    EXPIRY_MARGIN = 30  # seconds

    def __init__(self, username=None, password=None, session=None, credentials_file=None):
        self._username = username or os.environ.get(config.COPERNICUS_USER_ENV)
        self._password = password or os.environ.get(config.COPERNICUS_PASSWORD_ENV)
        if not (self._username and self._password):
            file_user, file_password = self._read_credentials_file(credentials_file)
            self._username = self._username or file_user
            self._password = self._password or file_password
        self._session = session or requests.Session()
        self._access_token = None
        self._expires_at = 0.0
        self._refresh_token = None
        self._refresh_expires_at = 0.0

    def access_token(self) -> str:
        now = time.time()
        if self._access_token and now < self._expires_at - self.EXPIRY_MARGIN:
            return self._access_token
        if self._refresh_token and now < self._refresh_expires_at - self.EXPIRY_MARGIN:
            response = self._grant(grant_type="refresh_token", refresh_token=self._refresh_token)
            if response.ok:
                return self._store(response)
        if not self._username:
            self._username = input("Copernicus Data Space username (e-mail): ")
        if not self._password:
            self._password = getpass.getpass("Copernicus Data Space password: ")
        response = self._grant(grant_type="password", username=self._username, password=self._password)
        if response.status_code in (400, 401):
            self._password = None  # ask again next time instead of repeating a wrong password
            raise AuthError("Copernicus Data Space did not accept the username and password")
        response.raise_for_status()
        return self._store(response)

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.access_token()}"}

    @staticmethod
    def _read_credentials_file(credentials_file):
        """(e-mail, password) from the credentials file; (None, None) if there is none."""
        try:
            path = TokenProvider._find_token_file(
                credentials_file, config.COPERNICUS_CREDENTIALS_FILE_ENV, config.COPERNICUS_CREDENTIALS_FILENAME)
        except AuthError:
            if credentials_file or os.environ.get(config.COPERNICUS_CREDENTIALS_FILE_ENV):
                raise  # asked for a file that is not there
            return None, None
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if len(lines) < 2:
            raise AuthError(f"{path} must hold the e-mail on the first line and the password on the second")
        return lines[0], lines[1]

    def _grant(self, **data):
        return self._session.post(config.COPERNICUS_TOKEN_URL, timeout=60,
                                  data={"client_id": config.COPERNICUS_CLIENT_ID, **data})

    def _store(self, response) -> str:
        payload = response.json()
        now = time.time()
        self._access_token = payload["access_token"]
        self._expires_at = now + float(payload.get("expires_in", 60))
        self._refresh_token = payload.get("refresh_token")
        self._refresh_expires_at = now + float(payload.get("refresh_expires_in", 0))
        return self._access_token


class EarthdataToken:
    """NASA Earthdata Login token for the NISAR products, read from ``nasa_token.txt``.

    The token is used as it is (there is nothing to refresh): when it expires, after
    60 days, generate a new one at https://urs.earthdata.nasa.gov ("Generate Token").
    The file is looked up in ``token_file``, then ``$NASA_TOKEN_FILE``, then
    ``nasa_token.txt`` in the current folder or any parent folder.
    """

    def __init__(self, token_file=None):
        self.token_file = TokenProvider._find_token_file(
            token_file, config.NASA_TOKEN_FILE_ENV, config.NASA_TOKEN_FILENAME)

    def access_token(self) -> str:
        token = self.token_file.read_text().strip().replace("\n", "")
        if not token:
            raise AuthError(f"{self.token_file} is empty")
        expires_at = self._expiry(token)
        if expires_at is not None and expires_at < time.time():
            raise AuthError(f"The token in {self.token_file} expired on "
                            f"{time.strftime('%Y-%m-%d', time.gmtime(expires_at))}: generate a new one")
        return token

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.access_token()}"}

    @staticmethod
    def _expiry(token: str):
        """Expiry time written inside the token, or None if it cannot be read."""
        try:
            payload = token.split(".")[1]
            return float(json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["exp"])
        except (IndexError, KeyError, TypeError, ValueError):
            return None
