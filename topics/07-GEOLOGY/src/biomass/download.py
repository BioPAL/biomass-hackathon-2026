"""Resumable file downloads from the MAAP data servers."""
import time
from pathlib import Path

import requests
from tqdm.auto import tqdm

from . import config
from .auth import TokenProvider
from .exceptions import DownloadError


class Downloader:
    """Downloads assets of STAC items into ``outdir/<item id>/``.

    Existing files are skipped; interrupted downloads (``*.part``) are resumed
    with an HTTP Range request when the server supports it.
    """

    def __init__(self, auth: TokenProvider, outdir=config.DEFAULT_OUTPUT_DIR,
                 retries: int = 3, chunk_size: int = 1 << 20, session=None, backoff: float = 3.0):
        self.auth = auth
        self.outdir = Path(outdir)
        self.retries = retries
        self.chunk_size = chunk_size
        self.backoff = backoff
        self._session = session or requests.Session()

    def download(self, items, asset_keys) -> list:
        done = []
        for item in items:
            for key in asset_keys:
                if key not in item.assets:
                    print(f"{item.id}: has no asset '{key}'")
                    continue
                href = item.assets[key].href
                name = Path(href.split("?")[0]).name or key
                done.append(self.download_file(href, self.outdir / item.id / name))
        print(f"\n{len(done)} files in {self.outdir.resolve()}")
        return done

    def download_file(self, url: str, dest) -> Path:
        dest = Path(dest)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  already exists, skipping: {dest.name}")
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                self._fetch(url, part, dest.name)
                part.replace(dest)
                return dest
            except Exception as e:  # network errors, HTTP errors, token refresh failures
                last_error = e
                print(f"  attempt {attempt}/{self.retries} failed: {e}")
                time.sleep(self.backoff * attempt)
        raise DownloadError(f"Could not download {url}") from last_error

    def _fetch(self, url: str, part: Path, label: str) -> None:
        headers = self.auth.headers()
        done = part.stat().st_size if part.exists() else 0
        if done:
            headers["Range"] = f"bytes={done}-"
        with self._session.get(url, headers=headers, stream=True, timeout=config.REQUEST_TIMEOUT) as r:
            r.raise_for_status()
            if done and r.status_code != 206:  # server ignored Range: start over
                done = 0
            total = done + int(r.headers.get("content-length", 0))
            with open(part, "ab" if done else "wb") as f, \
                    tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=label, leave=False) as bar:
                for chunk in r.iter_content(chunk_size=self.chunk_size):
                    f.write(chunk)
                    bar.update(len(chunk))
