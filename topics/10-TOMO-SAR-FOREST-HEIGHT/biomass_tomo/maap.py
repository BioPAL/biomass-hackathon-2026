"""
ESA MAAP access for BIOMASS Level-1c stacks: authentication, STAC search, grouping
of products into tomographic stacks, and download that rebuilds the product folders.

Credentials are never printed. The offline (refresh) token is read from a text file
(default: token.txt next to the notebooks, or $MAAP_TOKEN_FILE). The OIDC client used by
the ESA MAAP course material is the public "offline-token" client; override it with
$MAAP_CLIENT_ID / $MAAP_CLIENT_SECRET if ESA changes it.
"""
import os
import pathlib
import re
import time
from collections import defaultdict

import requests

CATALOG_URL = "https://catalog.maap.eo.esa.int/catalogue/"
TOKEN_URL = "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token"
L1C_COLLECTION = "BiomassLevel1c"
STACK_PRODUCT_TYPE = "STA__1S"          # single-look L1c stack product (S1/S2/S3_STA__1S)

# BIO_S1_STA__1S_20260424T225341_20260424T225402_T_G01_M03_C01_T029_F158_02_DS3JCN
ID_PATTERN = re.compile(
    r"BIO_(?P<swath>S\d)_(?P<ptype>STA__1[SM])_(?P<start>\d{8}T\d{6})_\d{8}T\d{6}_"
    r"\w_G(?P<coverage>\d\d)_M(?P<major>\d\d)_C(?P<cycle>\d\d)_T(?P<track>\d{3})_F(?P<frame>\d{3})_"
    r"(?P<version>\d\d)_\w+")

# Assets needed to rebuild a stack (quicklooks, KMLs and schemas are skipped)
ASSETS_REQUIRED = ("enclosure_i_abs_tiff", "enclosure_i_phase_tiff", "enclosure_vrt", "enclosure_nc",
                   "enclosure_annot_xml_1", "enclosure_annot_xml_2")
ASSETS_OPTIONAL = ("enclosure_orb_xml_1", "enclosure_orb_xml_2", "enclosure_att_xml_1", "enclosure_att_xml_2")


# authentication
def _read_kv(path):
    out = {}
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip().upper()] = v.strip()
    return out


def _client(credentials_file):
    """OIDC client id/secret from $MAAP_CLIENT_ID/$MAAP_CLIENT_SECRET or the credentials file
    (CLIENT_ID=... / CLIENT_SECRET=..., the public values given in the ESA MAAP course notebooks)."""
    kv = _read_kv(credentials_file) if credentials_file and pathlib.Path(credentials_file).exists() else {}
    cid = os.environ.get("MAAP_CLIENT_ID", kv.get("CLIENT_ID", "offline-token"))
    secret = os.environ.get("MAAP_CLIENT_SECRET", kv.get("CLIENT_SECRET"))
    if not secret:
        raise RuntimeError(f"No MAAP client secret: set $MAAP_CLIENT_SECRET or CLIENT_SECRET=... in {credentials_file}")
    return cid, secret


class MaapSession:
    """Short-lived access tokens from the offline token, refreshed automatically.

    token_file: plain offline token (token.txt) or a key=value file with OFFLINE_TOKEN=...
    credentials_file: key=value file with CLIENT_ID / CLIENT_SECRET (default maap_credentials.txt)
    """

    def __init__(self, token_file=None, credentials_file="maap_credentials.txt"):
        path = pathlib.Path(token_file or os.environ.get("MAAP_TOKEN_FILE", "token.txt")).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"MAAP offline token file not found: {path}")
        text = path.read_text().strip()
        self._offline = _read_kv(path).get("OFFLINE_TOKEN") if "OFFLINE_TOKEN" in text else text.replace("\n", "")
        self._credentials_file = credentials_file
        self._token, self._expires = None, 0.0

    def token(self):
        if self._token is None or time.time() > self._expires - 60:
            cid, secret = _client(self._credentials_file)
            r = requests.post(TOKEN_URL, data={"client_id": cid, "client_secret": secret,
                                               "grant_type": "refresh_token", "refresh_token": self._offline,
                                               "scope": "offline_access openid"}, timeout=60)
            r.raise_for_status()
            js = r.json()
            if not js.get("access_token"):
                raise RuntimeError("MAAP IAM returned no access token (offline token expired?)")
            self._token, self._expires = js["access_token"], time.time() + float(js.get("expires_in", 300))
        return self._token

    def headers(self):
        return {"Authorization": f"Bearer {self.token()}"}


# catalogue
def _search(retries=4, wait_s=10, **kw):
    """STAC search with retries: the MAAP catalogue intermittently answers with an HTML error page."""
    from pystac_client import Client
    from pystac_client.exceptions import APIError
    for attempt in range(retries):
        try:
            return list(Client.open(CATALOG_URL).search(**kw).items())
        except APIError:
            if attempt == retries - 1:
                raise
            time.sleep(wait_s * (attempt + 1))


def parse_id(product_id):
    m = ID_PATTERN.search(product_id)
    return m.groupdict() if m else None


def search_l1c(bbox=None, intersects=None, datetime=None, collection=L1C_COLLECTION, max_items=2000):
    """All L1c single-look stack products whose footprint intersects the AOI."""
    kw = dict(collections=[collection], method="GET", max_items=max_items)
    if intersects is not None:
        kw["intersects"] = intersects
    elif bbox is not None:
        kw["bbox"] = list(bbox)
    if datetime is not None:
        kw["datetime"] = datetime
    items = _search(**kw)
    return [it for it in items if (p := parse_id(it.id)) and p["ptype"] == "STA__1S"]


def group_stacks(items, aoi=None):
    """
    Group products into tomographic stacks: same swath, major cycle, track, frame and
    product version. Returns a list of dicts sorted by (#acquisitions, AOI coverage).
    aoi: optional shapely geometry (lon/lat); coverage = fraction of the AOI inside the
    intersection of all footprints of the stack.
    """
    from shapely.geometry import shape
    groups = defaultdict(list)
    for it in items:
        p = parse_id(it.id)
        groups[(p["swath"], p["major"], p["track"], p["frame"], p["version"])].append(it)
    stacks = []
    for key, its in groups.items():
        # one product per repeat cycle (keep the most recently published duplicate)
        by_cycle = {}
        for it in sorted(its, key=lambda i: i.properties.get("published", "")):
            by_cycle[parse_id(it.id)["cycle"]] = it
        its = [by_cycle[c] for c in sorted(by_cycle)]
        common = shape(its[0].geometry)
        for it in its[1:]:
            common = common.intersection(shape(it.geometry))
        cover = float(common.intersection(aoi).area / aoi.area) if aoi is not None and aoi.area > 0 else None
        primary = [it.id for it in its if it.properties.get("eofeos:is_coregistration_primary")]
        stacks.append(dict(
            key="_".join(f"{a}{b}" for a, b in zip(("", "M", "T", "F", "v"), key)),
            swath=key[0], major_cycle=int(key[1]), track=int(key[2]), frame=int(key[3]), version=key[4],
            n_acquisitions=len(its), cycles=[int(parse_id(i.id)["cycle"]) for i in its],
            dates=[parse_id(i.id)["start"] for i in its], primary=primary[0] if primary else None,
            stack_id=its[0].properties.get("eofeos:stack_id"), aoi_coverage=cover,
            mission_phase=its[0].properties.get("eofeos:mission_phase"), items=its))
    stacks.sort(key=lambda s: (s["n_acquisitions"], s["aoi_coverage"] or 0), reverse=True)
    return stacks


# download
def _relative_path(href, product_id):
    """Path inside the product folder, e.g. 'measurement/..._i_abs.tiff'."""
    path = href.split("?")[0]
    marker = f"{product_id}/{product_id}/"
    if marker in path:
        return path.split(marker, 1)[1]
    return path.rsplit("/", 1)[1]


def download_item(item, out_dir, session, assets=ASSETS_REQUIRED + ASSETS_OPTIONAL, chunk=8 << 20):
    """Download the assets of one product into out_dir/<product_id>/<relative path>; skips complete files."""
    pdir = pathlib.Path(out_dir) / item.id
    for key in assets:
        if key not in item.assets:
            if key in ASSETS_REQUIRED:
                raise KeyError(f"{item.id}: required asset {key} missing")
            continue
        asset = item.assets[key]
        dest = pdir / _relative_path(asset.href, item.id)
        size = asset.extra_fields.get("file:size")
        if dest.exists() and (size is None or dest.stat().st_size == int(size)):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(asset.href, headers=session.headers(), stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for block in r.iter_content(chunk_size=chunk):
                    f.write(block)
        if size is not None and tmp.stat().st_size != int(size):
            raise IOError(f"{dest.name}: size {tmp.stat().st_size} != catalogue {size}")
        tmp.replace(dest)
    return pdir


def download_stack(stack, out_dir, session, progress=print):
    out_dir = pathlib.Path(out_dir)
    for i, item in enumerate(stack["items"], 1):
        progress(f"[{i}/{stack['n_acquisitions']}] {item.id}")
        download_item(item, out_dir, session)
    return out_dir


def complete_stack(stack, pad_deg=0.05):
    """Re-query the catalogue over the centre of the stack footprint and its own time window, so a
    stack cut short by max_items in a wide search gets all its acquisitions. Returns a new stack dict."""
    seed = stack["items"][0]
    lon = (seed.bbox[0] + seed.bbox[2]) / 2
    lat = (seed.bbox[1] + seed.bbox[3]) / 2
    # the stack_id embeds the start/stop of the whole stack: ..._ID_20260424T225147_20260512T225219_...
    t = re.findall(r"(\d{8}T\d{6})", stack.get("stack_id") or "")[:2] or sorted(stack["dates"])
    iso = lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:15]}Z"
    items = _search(collections=[L1C_COLLECTION], method="GET", max_items=200,
                    bbox=[lon - pad_deg, lat - pad_deg, lon + pad_deg, lat + pad_deg],
                    datetime=[iso(t[0])[:10] + "T00:00:00Z", iso(t[-1])[:10] + "T23:59:59Z"])
    if stack.get("stack_id"):
        same = [it for it in items if it.properties.get("eofeos:stack_id") == stack["stack_id"]]
        items = same or items
    l1c = [it for it in items if parse_id(it.id) and parse_id(it.id)["ptype"] == "STA__1S"]
    full = [s for s in group_stacks(l1c) if s["key"] == stack["key"]]
    return full[0] if full and full[0]["n_acquisitions"] >= stack["n_acquisitions"] else stack


def product_on_disk(out_dir, product_id):
    """Folder of an already downloaded (and extracted) product, or None."""
    for d in pathlib.Path(out_dir).glob(f"{product_id}*"):
        hits = [p for p in [d, *d.glob(product_id)] if (p / "measurement").is_dir() and any((p / "measurement").glob("*_i_abs.tiff"))]
        if hits:
            return hits[0]
    return None


def download_zip(item, out_dir, session, keep_zip=False, chunk=8 << 20):
    """Download the whole product as one zip (asset 'product') and extract it to
    out_dir/<product_id>_product/ (the layout produced by unzipping the MAAP product)."""
    import zipfile
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = out_dir / f"{item.id}_product.zip"
    if not zpath.exists():
        tmp = zpath.with_suffix(".zip.part")
        with requests.get(item.assets["product"].href, headers=session.headers(), stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for block in r.iter_content(chunk_size=chunk):
                    f.write(block)
        tmp.replace(zpath)
    dest = out_dir / f"{item.id}_product"
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        if bad:
            raise IOError(f"{zpath.name}: corrupt member {bad}")
        z.extractall(dest)
    if not keep_zip:
        zpath.unlink()
    return dest
