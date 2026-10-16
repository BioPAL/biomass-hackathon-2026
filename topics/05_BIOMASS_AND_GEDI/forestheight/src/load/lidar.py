"""NASA LVIS airborne LiDAR (AfriSAR 2016, L3 RH98 25 m grids): find and download."""

from pathlib import Path


def find_lvis(site_dir: Path, filename: str):
    """LVIS file in the site folder (case-insensitive: NASA names use 'RH98')."""
    for p in Path(site_dir).glob("*.tif"):
        if p.name.lower() == filename.lower():
            return p
    return None


def download_lvis(url: str, site_dir: Path):
    """Download one LVIS GeoTIFF from NASA Earthdata (interactive login, nothing saved)."""
    import earthaccess

    earthaccess.login(strategy="interactive", persist=False)
    return earthaccess.download([url], str(site_dir))
