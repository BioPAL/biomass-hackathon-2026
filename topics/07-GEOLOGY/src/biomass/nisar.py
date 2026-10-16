"""NISAR products from NASA's catalogue (distributed by the Alaska Satellite Facility)."""
from pathlib import Path

import pandas as pd
import requests

from . import config
from .auth import EarthdataToken
from .catalog import BiomassClient
from .exceptions import QueryError
from .query import SearchQuery
from .viz import _add_overlay, _finish_map

POLARISATIONS = {"SH", "SV", "DH", "DV", "QP"}
BANDWIDTHS = {5, 20, 40, 77}  # MHz
CMR_PAGE_SIZE = 2000

GCOV_GRIDS = "science/LSAR/GCOV/grids"
IDENTIFICATION = "science/LSAR/identification"


class NisarClient(BiomassClient):
    """Discover NISAR collections and search for products.

    In NASA's catalogue each product type is a collection of its own (e.g.
    ``NISAR_L2_GCOV_PROVISIONAL_V1_1``), so queries leave ``product_type`` and
    ``extra_filters`` empty. Results come oldest first.
    """

    def __init__(self, auth: EarthdataToken = None, catalog_url: str = config.NASA_STAC_URL):
        super().__init__(auth or EarthdataToken(), catalog_url)

    def collections(self) -> pd.DataFrame:
        res = self.stac.collection_search(q="NISAR")
        cols = pd.json_normalize(res.collection_list_as_dict(), record_path=["collections"])
        cols = cols[cols["id"].str.match(r"NISAR_L\d")]
        return cols.sort_values("id", ignore_index=True)[["id", "title", "extent.temporal.interval"]]

    def queryables(self, collection: str) -> pd.DataFrame:
        raise QueryError("NASA's catalogue has no queryables: choose the product type with the collection")

    def search(self, query: SearchQuery, polarisation=None, bandwidth: int = None) -> list:
        """Search for products, optionally keeping only one acquisition mode.

        ``polarisation`` is the code of the main band (frequency A), or a list of codes:
        "SH"/"SV" (HH or VV only), "DH" (HH+HV), "DV" (VV+VH) or "QP" (all four).
        ``bandwidth`` is that of the main band in MHz: 5, 20, 40 or 77.
        """
        if query.cql_filter():
            raise QueryError("NISAR searches take no product_type or extra_filters: "
                             "choose the product type with the collection")
        if polarisation is None and bandwidth is None:
            items = super().search(query)
        else:
            names, matched = self._granule_names(query, _name_patterns(polarisation, bandwidth))
            items = self._items(query.collection, names)
            print(f"{len(items)} products fetched | {matched} match in total")
        for item in items:
            item.assets = _named_assets(item.id, item.assets)
        return items

    def _granule_names(self, query: SearchQuery, patterns: list):
        """Names of the products whose name fits any of ``patterns``, oldest first, and how many there are.

        The STAC interface cannot filter on the product name, so this asks NASA's own
        search service (CMR), which can.
        """
        short_name, version = query.collection.rsplit("_", 1)
        params = {"short_name": short_name, "version": version, "sort_key": "start_date",
                  "temporal": ",".join(query.datetime_range()),
                  "readable_granule_name[]": patterns, "options[readable_granule_name][pattern]": "true",
                  "page_size": min(query.max_items or CMR_PAGE_SIZE, CMR_PAGE_SIZE)}
        bbox = query.resolved_bbox()
        if bbox:
            params["bounding_box"] = ",".join(map(str, bbox))
        r = requests.get(config.NASA_CMR_GRANULES_URL, params=params, timeout=config.REQUEST_TIMEOUT)
        r.raise_for_status()
        names = [e.get("producer_granule_id") or e["title"] for e in r.json()["feed"]["entry"]]
        return names, int(r.headers.get("cmr-hits", len(names)))

    def _items(self, collection: str, names: list) -> list:
        """The catalogue items with these names, in the same order."""
        items = {}
        for i in range(0, len(names), 25):  # a few at a time: the ids travel in the address
            chunk = names[i:i + 25]
            found = self.stac.search(collections=[collection], ids=chunk, method="GET", limit=len(chunk))
            items.update((item.id, item) for item in found.items())
        return [items[name] for name in names if name in items]


def _name_patterns(polarisation, bandwidth) -> list:
    """Wildcard patterns for product names, which hold ``_<mode>_<pols>_``, e.g. ``_4005_DHDH_``.

    Both parts give the main band (frequency A) first and the second band after it.
    """
    pols = [polarisation] if isinstance(polarisation, str) else list(polarisation or ["??"])
    for code in pols:
        if code != "??" and code.upper() not in POLARISATIONS:
            raise QueryError(f"polarisation must be one of {sorted(POLARISATIONS)}, not {code!r}")
    if bandwidth is not None and bandwidth not in BANDWIDTHS:
        raise QueryError(f"bandwidth must be one of {sorted(BANDWIDTHS)} (MHz), not {bandwidth!r}")
    mode = "??" if bandwidth is None else f"{bandwidth:02d}"
    return [f"*_{mode}??_{code.upper()}??_*" for code in pols]


def _named_assets(item_id: str, assets: dict) -> dict:
    """The https assets under short names instead of their addresses.

    ``product`` is the .h5 file, ``browse`` the colour image in the product's own map
    grid, ``browse_latlon`` the same in lon/lat and ``thumbnail`` its small version;
    images of a single term are e.g. ``browse_a_hhhh``. The ``s3://`` copies are left
    out: they only work from inside NASA's cloud.
    """
    named = {}
    for asset in assets.values():
        href = asset.href
        name = href.rsplit("/", 1)[-1]
        if not href.startswith("https://") or not name.startswith(item_id):
            continue
        suffix, _, extension = name[len(item_id):].lower().rpartition(".")
        if extension == "h5":
            key = "product"
        elif extension != "png":
            continue
        elif suffix in ("_latlon_thumbnail", "_thumbnail"):
            key = "thumbnail"
        elif suffix == "_latlon":
            key = "browse_latlon"
        else:
            key = "browse" + suffix.replace("_native", "")
        named.setdefault(key, asset)
    return named


def nisar_items_table(items) -> pd.DataFrame:
    """Like ``items_table``, with what the product name tells: orbit, track, frame and mode.

    ``mode`` is the bandwidth in MHz of the main and second bands (4005 = 40 and 5) and
    ``pols`` their polarisations (DHDH = HH+HV in both; NA = band not acquired).
    """
    rows = []
    for i, it in enumerate(items):
        # NISAR_L2_PR_GCOV_<cycle>_<track>_<A|D>_<frame>_<mode>_<pols>_...
        name = it.id.split("_")
        known = len(name) > 9
        rows.append({"idx": i, "id": it.id, "start": it.properties.get("start_datetime"),
                     "orbit": {"A": "ascending", "D": "descending"}.get(name[6]) if known else None,
                     "track": name[5] if known else None, "frame": name[7] if known else None,
                     "mode": name[8] if known else None, "pols": name[9] if known else None,
                     "assets": len(it.assets)})
    return pd.DataFrame(rows)


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def inspect_gcov(path) -> dict:
    """Print what a local GCOV .h5 file holds; return it as a dict.

    A GCOV file has one grid per frequency: ``A`` is the main band and ``B``, when
    present, a second band (often at a coarser spacing). Each holds backscatter terms
    such as ``HHHH`` (HH power) and ``HVHV``, terrain corrected (gamma0), in linear units.
    """
    import h5py

    with h5py.File(path, "r") as f:
        ident = f[IDENTIFICATION]
        info = {key: _text(ident[key][()]) for key in
                ("granuleId", "radarBand", "orbitPassDirection", "lookDirection", "trackNumber",
                 "frameNumber", "zeroDopplerStartTime", "productVersion") if key in ident}
        info["frequencies"] = {}
        for freq, grid in f[GCOV_GRIDS].items():
            terms = [_text(t) for t in grid["listOfCovarianceTerms"][()]]
            info["frequencies"][freq[-1]] = {
                "terms": terms, "shape": grid[terms[0]].shape, "epsg": int(grid["projection"][()]),
                "spacing_m": abs(float(grid["xCoordinateSpacing"][()]))}
    print(info.get("granuleId"))
    print(f"Band: {info.get('radarBand')} | orbit: {info.get('orbitPassDirection')} | look: "
          f"{info.get('lookDirection')} | track {info.get('trackNumber')} frame {info.get('frameNumber')} "
          f"| start: {info.get('zeroDopplerStartTime')}")
    for freq, g in info["frequencies"].items():
        print(f"  frequency {freq}: {g['shape'][1]} x {g['shape'][0]} px at {g['spacing_m']:g} m | "
              f"EPSG:{g['epsg']} | terms: {', '.join(g['terms'])}")
    return info


def read_gcov(path, frequency: str = None, max_size: int = 1500):
    """Reduced-resolution read of a local GCOV .h5 file as dB (NaN where there is no data).

    Returns ``(db, names, transform, crs)``: the (terms, rows, cols) array, the name of
    each term, and the affine transform and CRS of the reduced grid. ``frequency`` is
    "A" or "B"; by default the coarsest one in the file, which is the quickest to read
    (under a second, against some 20 s for frequency A at 10 m). ``max_size=None`` reads
    every pixel.
    """
    import h5py
    import numpy as np
    from affine import Affine

    with h5py.File(path, "r") as f:
        grids = f[GCOV_GRIDS]
        if frequency is None:
            frequency = max(grids, key=lambda name: abs(float(grids[name]["xCoordinateSpacing"][()])))[-1]
        grid = grids[f"frequency{frequency.upper()}"]
        names = [_text(t) for t in grid["listOfCovarianceTerms"][()]]
        names = [n for n in names if n[:2] == n[2:]]  # backscatter terms only (HHHH, HVHV, VVVV...)
        rows, cols = grid[names[0]].shape
        step = 1 if max_size is None else max(1, -(-max(rows, cols) // max_size))
        power = np.stack([grid[n][::step, ::step] for n in names]).astype("float32")
        x, y = grid["xCoordinates"][()], grid["yCoordinates"][()]
        dx, dy = float(grid["xCoordinateSpacing"][()]) * step, float(grid["yCoordinateSpacing"][()]) * step
        crs = f"EPSG:{int(grid['projection'][()])}"
    transform = Affine(dx, 0, x[0] - dx / 2, 0, dy, y[0] - dy / 2)  # coordinates are pixel centres
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(power)
    db[~np.isfinite(db)] = np.nan
    return db, names, transform, crs


def plot_gcov(path, frequency: str = None, max_size: int = 1500):
    """Plot every backscatter term of a local GCOV .h5 file in dB, on its map grid."""
    import matplotlib.pyplot as plt
    import numpy as np

    db, names, transform, crs = read_gcov(path, frequency, max_size)
    west, north = transform * (0, 0)
    east, south = transform * (db.shape[2], db.shape[1])
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 6), sharey=True,
                             squeeze=False, constrained_layout=True)
    for ax, band, name in zip(axes[0], db, names):
        lo, hi = np.nanpercentile(band, [2, 98])
        im = ax.imshow(band, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest",
                       extent=[west / 1000, east / 1000, south / 1000, north / 1000])
        ax.set_title(name)
        ax.set_xlabel(f"x (km, {crs})")
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.01, label="gamma0 (dB)")
    axes[0][0].set_ylabel("y (km)")
    fig.suptitle(Path(path).name, fontsize=9)
    return fig


def gcov_map(paths, band: str = "rgb", frequency: str = None, max_size: int = 1000, opacity: float = 0.9):
    """Folium map with local GCOV .h5 files laid over it.

    ``band`` is a term ("HHHH", "HVHV", "VVVV"...) shown in grey, or "rgb" for a colour
    composite: R=HHHH, G=HVHV and B=VVVV, or B=HHHH/HVHV when the file has no VV.
    """
    import folium
    import numpy as np
    from rasterio.warp import calculate_default_transform, reproject

    if isinstance(paths, (str, Path)):
        paths = [paths]
    m = folium.Map()
    bounds = []
    for path in paths:
        db, names, transform, crs = read_gcov(path, frequency, max_size)
        layers = dict(zip(names, db))
        if band.upper() in layers:
            src = layers[band.upper()][None]
        elif band.lower() == "rgb" and {"HHHH", "HVHV"} <= set(layers):
            blue = layers["VVVV"] if "VVVV" in layers else layers["HHHH"] - layers["HVHV"]
            src = np.stack([layers["HHHH"], layers["HVHV"], blue])
        else:
            print(f"{Path(path).name}: has {names}, not {band}; showing {names[0]}")
            src = db[:1]
        rows, cols = src.shape[1:]
        west, north = transform * (0, 0)
        east, south = transform * (cols, rows)
        dst_transform, w, h = calculate_default_transform(crs, "EPSG:4326", cols, rows, west, south, east, north)
        dst = np.full((len(src), h, w), np.nan, dtype="float32")
        reproject(src, dst, src_transform=transform, src_crs=crs, dst_transform=dst_transform,
                  dst_crs="EPSG:4326", src_nodata=np.nan, dst_nodata=np.nan)
        bounds.append(_add_overlay(m, dst, dst_transform, Path(path).name, opacity))
    return _finish_map(m, bounds)
