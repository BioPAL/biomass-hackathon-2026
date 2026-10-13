"""Maps and plots of search results and downloaded rasters."""
from pathlib import Path

from .query import SearchQuery


def footprint_map(items, query: SearchQuery = None):
    """Folium map of item footprints (orange); the requested area is drawn in blue."""
    import folium
    import geopandas as gpd

    feats = [{"type": "Feature", "geometry": it.geometry,
              "properties": {"id": it.id, "idx": i, "start": it.properties.get("start_datetime")}}
             for i, it in enumerate(items) if it.geometry]
    if not feats:
        print("No results")
        return None
    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=3)
    folium.GeoJson(gdf.__geo_interface__,
                   tooltip=folium.GeoJsonTooltip(fields=["idx", "id", "start"]),
                   style_function=lambda _: {"color": "#ff7f0e", "weight": 1, "fillOpacity": 0.1}).add_to(m)
    b = query.resolved_bbox() if query else None
    if b:
        folium.Rectangle([[b[1], b[0]], [b[3], b[2]]], color="#1f77b4", fill=False).add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    return m


def _read_db(path, max_size):
    """Reduced-resolution read of a local GeoTIFF as dB (NaN where there is no data).

    Returns ``(db, names, gcps)``: the (bands, rows, cols) array, the polarisation of
    each band and the GCPs rescaled to the reduced grid.
    """
    import numpy as np
    import rasterio as rio
    from rasterio.control import GroundControlPoint
    from rasterio.enums import Resampling

    with rio.open(path) as src:
        h, w = min(src.height, max_size), min(src.width, max_size)
        amp = src.read(out_shape=(src.count, h, w), masked=True,
                       resampling=Resampling.average).astype("float32").filled(np.nan)
        names = [src.tags(i + 1).get("POLARIMETRIC_INTERP", f"band {i + 1}") for i in range(src.count)]
        gcps = [GroundControlPoint(row=g.row * h / src.height, col=g.col * w / src.width,
                                   x=g.x, y=g.y, z=g.z or 0)
                for g in src.get_gcps()[0]]
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 20 * np.log10(amp)
    db[~np.isfinite(db)] = np.nan
    return db, names, gcps


def _stretch(band):
    """Scale a band to 0-1 between its 2nd and 98th percentiles."""
    import numpy as np

    lo, hi = np.nanpercentile(band, [2, 98])
    return np.clip((band - lo) / ((hi - lo) or 1), 0, 1)


def plot_raster(path, max_size: int = 1500):
    """Plot every polarisation of a local GeoTIFF in dB, in image (radar) geometry."""
    import matplotlib.pyplot as plt
    import numpy as np

    db, names, _ = _read_db(path, max_size)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 9), sharey=True,
                             squeeze=False, constrained_layout=True)
    for ax, band, name in zip(axes[0], db, names):
        lo, hi = np.nanpercentile(band, [2, 98])
        im = ax.imshow(band, cmap="gray", vmin=lo, vmax=hi, aspect="auto", interpolation="nearest")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.01, label="dB")
    fig.suptitle(Path(path).name, fontsize=9)
    return fig


def raster_map(paths, band: str = "rgb", max_size: int = 1000, opacity: float = 0.9):
    """Folium map with local GeoTIFFs laid over it, geolocated with their GCPs.

    ``band`` is a polarisation ("HH", "HV", "VH", "VV") shown in grey, or "rgb" for a
    colour composite (R=HH, G=HV, B=VV). The geolocation is approximate (quick look).
    """
    import folium
    import numpy as np
    from rasterio.warp import calculate_default_transform, reproject

    if isinstance(paths, (str, Path)):
        paths = [paths]
    m = folium.Map()
    bounds = []
    for path in paths:
        db, names, gcps = _read_db(path, max_size)
        if not gcps:
            print(f"{Path(path).name}: no GCPs, skipped")
            continue
        wanted = ["HH", "HV", "VV"] if band.lower() == "rgb" else [band.upper()]
        if not set(wanted) <= set(names):
            print(f"{Path(path).name}: has {names}, not {wanted}; showing {names[0]}")
            wanted = [names[0]]
        src = np.stack([db[names.index(n)] for n in wanted])
        transform, w, h = calculate_default_transform(
            "EPSG:4326", "EPSG:4326", src.shape[2], src.shape[1], gcps=gcps)
        dst = np.full((len(wanted), h, w), np.nan, dtype="float32")
        reproject(src, dst, gcps=gcps, src_crs="EPSG:4326", dst_transform=transform,
                  dst_crs="EPSG:4326", src_nodata=np.nan, dst_nodata=np.nan)
        grey = [_stretch(b) for b in dst]
        valid = np.isfinite(dst).all(axis=0)
        rgba = np.dstack([np.nan_to_num(b) for b in (grey if len(grey) == 3 else grey * 3)] + [valid])
        west, north = transform * (0, 0)
        east, south = transform * (w, h)
        folium.raster_layers.ImageOverlay(rgba, bounds=[[south, west], [north, east]], opacity=opacity,
                                          mercator_project=True, name=Path(path).name).add_to(m)
        bounds.append((south, west, north, east))
    if not bounds:
        print("Nothing to show")
        return None
    s, w_, n, e = zip(*bounds)
    m.fit_bounds([[min(s), min(w_)], [max(n), max(e)]])
    folium.LayerControl().add_to(m)
    return m
