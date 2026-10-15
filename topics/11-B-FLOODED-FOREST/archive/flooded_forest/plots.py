"""Plots: quick Pauli maps from catalogue assets, and plots from extracted stacks.

Two sources:
- plot_catalogue_pauli: streams a few tiles of one product's abs/phase COGs
  (ported from single_scene_processor.py) — works without extracting stacks.
- plot_stack_outputs: reads the files written by
  marcos_dev/1_data_extraction/extract_l1c_roi.py (ported from
  visualize_stacks.ipynb) — overview, Pauli RGB per stack, HH/HV time series.
"""

import json
import warnings
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.windows import Window
from scipy.ndimage import uniform_filter
from shapely.geometry import Polygon, box, shape

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

INK = "#2b2b2a"
HALO = [pe.withStroke(linewidth=3.5, foreground=INK)]
STACK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

BAND_ORDER = ("HH", "HV", "VH", "VV")  # verify against the printed band descriptions
NODATA = -9999.0


def _save(fig, out_dir, tag, show=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  saved {out}")
    if show:
        plt.show()
    plt.close(fig)
    return out


# -------------------- catalogue quick Pauli (remote COG tiles) --------------------


def _to_complex(amp, phase):
    if np.nanmax(np.abs(phase)) > 2 * np.pi + 0.1:  # auto-detect degrees
        phase = np.deg2rad(phase)
    return (amp * np.exp(1j * phase)).astype("complex64")


def _read_channels(src_abs, src_phase, win):
    """Complex HH, HV, VV for a window (HV/VH averaged, 9999 -> NaN)."""
    n = src_abs.count
    names = BAND_ORDER if n == 4 else ("HH", "HV", "VV")
    ch = {}
    for nm, i in ((nm, i + 1) for i, nm in enumerate(names)):
        amp = src_abs.read(i, window=win).astype("float32")
        amp[amp == NODATA] = np.nan
        ch[nm] = _to_complex(amp, src_phase.read(i, window=win).astype("float32"))
    hv = (ch["HV"] + ch["VH"]) / 2 if "VH" in ch else ch["HV"]
    return ch["HH"], hv, ch["VV"]


def _stretch_power(p, lo=2, hi=98):
    db = 10 * np.log10(p + 1e-12)
    a, b = np.nanpercentile(db, [lo, hi])
    return np.clip((db - a) / (b - a + 1e-12), 0, 1)


def _plot_grid(config, tiles, render, title, tag, out_dir, cmap=None, vmin=None, vmax=None,
               cbar_label=None, show=False):
    grid = int(np.sqrt(len(tiles)))
    fig, axes = plt.subplots(grid, grid, figsize=(12, 12))
    im = None
    for ax, t in zip(axes.ravel(), tiles):
        im = ax.imshow(render(*t), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
    fig.suptitle(title)
    if cbar_label and im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label=cbar_label)
    return _save(fig, out_dir, tag, show)


def plot_catalogue_pauli(config, roi_entry, token, product_index=0, tile=512, grid=3, looks=5,
                         out_dir=None, show=False):
    """Pauli RGB / components / dominant mechanism of one catalogue product (remote tiles)."""
    products = roi_entry.get("products", [])
    if not products:
        print("No products to plot")
        return []
    out_dir = out_dir or config.roi_dir(roi_entry["slug"]) / "figures"

    product = products[product_index]
    abs_url = product["assets"].get("enclosure_i_abs_tiff")
    phase_url = product["assets"].get("enclosure_i_phase_tiff")
    if not (abs_url and phase_url):
        print("  -> missing abs/phase assets")
        return []

    env = {
        "GDAL_HTTP_HEADERS": f"Authorization: Bearer {token.get()}",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_VERSION": "2",
    }
    with rasterio.Env(**env):
        with rasterio.open(f"/vsicurl/{abs_url}") as sa, rasterio.open(f"/vsicurl/{phase_url}") as sp:
            print("Abs bands:", sa.count, sa.dtypes, "| Phase bands:", sp.count, sp.dtypes)
            width, height = sa.width, sa.height
            xs = np.linspace(0, width - tile, grid).astype(int)
            ys = np.linspace(0, height - tile, grid).astype(int)
            tiles = []
            for win in (Window(x, y, tile, tile) for y in ys for x in xs):
                hh, hv, vv = _read_channels(sa, sp, win)
                surface, double, volume = (
                    np.abs(hh + vv) ** 2 / 2,
                    np.abs(hh - vv) ** 2 / 2,
                    np.abs(hv) ** 2,
                )
                if looks > 1:
                    surface, double, volume = (
                        uniform_filter(p, looks) for p in (surface, double, volume)
                    )
                tiles.append((surface, double, volume))

    title = f"{roi_entry['name']} | {product.get('datetime')}"
    figures = []
    figures.append(_plot_grid(
        config, tiles, lambda s, d, v: np.dstack([_stretch_power(d), _stretch_power(v), _stretch_power(s)]),
        f"{title}\nPauli RGB (R=double, G=volume, B=surface)", "pauli_rgb", out_dir, show=show))
    for tag, k in (("surface", 0), ("double_bounce", 1), ("volume", 2)):
        figures.append(_plot_grid(
            config, tiles, lambda s, d, v, k=k: 10 * np.log10((s, d, v)[k] + 1e-12),
            f"{title}\n{tag} [dB]", f"{tag}_db", out_dir, cmap="viridis", vmin=-30, vmax=0,
            cbar_label="dB", show=show))
    figures.append(_plot_grid(
        config, tiles, lambda s, d, v: np.argmax(np.stack([s, d, v]), axis=0),
        f"{title}\nDominant mechanism (blue=surface/water, red=double, green=volume/forest)",
        "dominant_mechanism", out_dir,
        cmap=ListedColormap(["tab:blue", "tab:red", "tab:green"]),
        vmin=0, vmax=2, show=show))
    return figures


# -------------------- plots from extracted stacks --------------------


def _multilook(a, looks):
    """Block average, ignoring NaN (no-data)."""
    la, lr = looks
    h, w = a.shape[0] // la * la, a.shape[1] // lr * lr
    return np.nanmean(a[:h, :w].reshape(h // la, la, w // lr, lr), axis=(1, 3))


def _footprint(lat, lon):
    """Outline of the extracted window, from the border of its lat/lon grid."""
    ring = np.concatenate([
        np.column_stack([lon[0, :], lat[0, :]]),
        np.column_stack([lon[:, -1], lat[:, -1]]),
        np.column_stack([lon[-1, ::-1], lat[-1, ::-1]]),
        np.column_stack([lon[::-1, 0], lat[::-1, 0]]),
    ])
    return Polygon(ring)


def _pick_product(stack, which="primary"):
    if which == "primary":
        return next(p for p in stack["products"] if p["coregistration_primary"])
    return next(p for p in stack["products"] if int(p["repeat_cycle"]) == int(which))


def _load_geometry(roi_dir, stack, looks):
    """Multilooked lat, lon and common-area mask of a stack."""
    folder = roi_dir / stack["folder"]
    with rasterio.open(folder / "latlon.tif") as src:
        lat, lon = _multilook(src.read(1), looks), _multilook(src.read(2), looks)
    with rasterio.open(folder / "overlap_mask.tif") as src:
        inside = _multilook(src.read(1).astype(np.float32), looks) > 0.5
    return lat, lon, inside


def _read_common_area(path):
    """Common area geometry from common_area.geojson (json + shapely, no geopandas)."""
    data = json.loads(Path(path).read_text())
    for feature in data["features"]:
        if feature.get("properties", {}).get("name") == "common_area":
            return shape(feature["geometry"])
    raise ValueError(f"no 'common_area' feature in {path}")


def plot_stack_outputs(config, roi_dir, looks=(12, 2), dim_outside=0.35, show=False):
    """Figures from an extracted ROI folder.

    Per stack folder (<roi>/<stack>/figures/): Pauli RGB of the primary
    acquisition, HH backscatter of every date. ROI level (<roi>/figures/):
    overview map and HH/HV time series.
    """
    roi_dir = Path(roi_dir)
    summary = json.loads((roi_dir / "extraction_summary.json").read_text())
    roi_bbox = summary["roi"]["bbox"]
    roi = box(*roi_bbox)
    common = _read_common_area(roi_dir / "common_area.geojson")
    stacks = summary["stacks"]
    aspect = 1 / np.cos(np.radians(np.mean(roi_bbox[1::2])))

    geometry = [_load_geometry(roi_dir, st, looks) for st in stacks]

    figures = []
    for st, geo in zip(stacks, geometry):
        stack_dir = roi_dir / st["folder"]
        print(f"Stack {st['folder']}:")
        figures.append(_plot_stack_pauli(roi_dir, st, geo, roi, roi_bbox, common, aspect,
                                         dim_outside, looks, show))
        figures.append(_plot_stack_hh_dates(roi_dir, st, geo, common, looks, show))
    figures.append(_plot_overview(roi_dir / "figures", summary, roi, roi_bbox, common,
                                  stacks, geometry, aspect, show))
    figures.append(_plot_timeseries(roi_dir / "figures", roi_dir, stacks, geometry, looks, show))
    return figures


def _plot_stack_pauli(roi_dir, stack, geo, roi, roi_bbox, common, aspect, dim_outside, looks, show):
    """Pauli RGB of the stack's primary acquisition, geocoded, in the stack folder."""
    lat, lon, inside = geo
    product = _pick_product(stack)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.pcolormesh(lon, lat, _pauli_rgba(roi_dir, stack, product, inside, looks, dim_outside),
                  shading="nearest", rasterized=True)
    _draw_outlines(ax, roi, common)
    _map_axes(ax, roi_bbox, aspect)
    swath, orbit = stack["folder"].split("_")[:2]
    date = product["datetime"][:10]
    tag = ", primary" if product["coregistration_primary"] else ""
    ax.set_title(f"{swath} {orbit} · {date} (C{int(product['repeat_cycle']):02d}{tag})", loc="left")

    pauli_key = [Patch(color=c, label=l) for c, l in (
        ("#e0302a", "R  |HH−VV|²  double bounce"),
        ("#2fa84f", "G  |HV+VH|²  volume"),
        ("#2a64d6", "B  |HH+VV|²  surface"),
    )] + [
        Line2D([], [], ls="--", color=INK, lw=1.2, label="ROI"),
        Line2D([], [], color="white", lw=1.5, path_effects=HALO, label="Common area"),
    ]
    ax.legend(handles=pauli_key, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
    return _save(fig, roi_dir / stack["folder"] / "figures", "pauli_rgb", show)


def _plot_stack_hh_dates(roi_dir, stack, geo, common, looks, show):
    """HH backscatter of every date of the stack, same colour scale, common-area zoom."""
    lat, lon, inside = geo
    bounds = getattr(common, "geoms", [common])[0].bounds if hasattr(common, "geoms") else common.bounds
    panel_ratio = (bounds[3] - bounds[1]) * (1 / np.cos(np.radians(np.mean(bounds[1::2])))) / (bounds[2] - bounds[0])
    cols = 7 if panel_ratio > 1.5 else 4
    rows = -(-len(stack["products"]) // cols)

    hh_db = []
    for product in stack["products"]:
        with rasterio.open(roi_dir / stack["folder"] / product["file"]) as src:
            hh = _multilook(np.abs(src.read(1)) ** 2, looks)
        db = 10 * np.log10(hh)
        db[~inside] = np.nan
        hh_db.append(db)
    vmin, vmax = np.nanpercentile(np.stack(hh_db), [2, 98])

    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols + 1.5, 2.3 * panel_ratio * rows + 1),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    for ax, product, db in zip(axes.flat, stack["products"], hh_db):
        mesh = ax.pcolormesh(lon, lat, db, shading="nearest", cmap="gray", vmin=vmin, vmax=vmax,
                             rasterized=True)
        _map_axes(ax, None, 1 / np.cos(np.radians(np.mean(bounds[1::2]))), pad=0.01, bounds=bounds)
        ax.set_title(f"{product['datetime'][:10]}  C{int(product['repeat_cycle']):02d}",
                     loc="left", fontsize=10)
        ax.label_outer()
    for ax in axes.flat[len(hh_db):]:
        ax.set_visible(False)

    fig.suptitle(f"{stack['folder']}: HH backscatter in the common area", x=0.01, ha="left")
    fig.colorbar(mesh, ax=axes, shrink=0.8, label="HH intensity (dB, uncalibrated)")
    return _save(fig, roi_dir / stack["folder"] / "figures", "hh_dates", show)


def _plot_overview(out_dir, summary, roi, roi_bbox, common, stacks, geometry, aspect, show):
    fig, ax = plt.subplots(figsize=(7, 7))
    for part in getattr(common, "geoms", [common]):
        ax.fill(*part.exterior.xy, color="#c3c2b7", alpha=0.6, lw=0)
    ax.plot(*roi.exterior.xy, "--", color=INK, lw=1.5)

    handles = [
        Line2D([], [], ls="--", color=INK, lw=1.5, label="ROI"),
        Patch(color="#c3c2b7", alpha=0.6,
              label=f"Common area ({100 * summary['common_area_fraction_of_roi']:.0f}% of ROI)"),
    ]
    for i, (st, (lat, lon, _)) in enumerate(zip(stacks, geometry)):
        color = STACK_COLORS[i % len(STACK_COLORS)]
        ax.plot(*_footprint(lat, lon).exterior.xy, color=color, lw=2)
        handles.append(Line2D([], [], color=color, lw=2, label=st["folder"]))

    _map_axes(ax, roi_bbox, aspect, pad=0.3)
    ax.set_title(f"{summary['roi']['name']}: extracted stacks", loc="left")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    return _save(fig, out_dir, "stack_overview", show)


def _stretch_db(db, inside, low=2, high=98):
    """Map dB to 0-1 using percentiles taken inside the common area."""
    lo, hi = np.nanpercentile(db[inside], [low, high])
    return np.clip((db - lo) / (hi - lo), 0, 1)


def _pauli_rgba(roi_dir, stack, product, inside, looks, dim_outside):
    """Pauli RGB: R = |HH-VV|^2/2, G = |HV+VH|^2/2, B = |HH+VV|^2/2."""
    with rasterio.open(roi_dir / stack["folder"] / product["file"]) as src:
        hh, hv, vh, vv = (src.read(b) for b in range(1, 5))
    channels = [np.abs(hh - vv) ** 2 / 2, np.abs(hv + vh) ** 2 / 2, np.abs(hh + vv) ** 2 / 2]
    del hh, hv, vh, vv
    rgb = np.stack([_stretch_db(10 * np.log10(_multilook(c, looks)), inside) for c in channels], axis=-1)
    no_data = np.isnan(rgb).any(axis=-1)
    rgb[~inside] *= dim_outside
    return np.dstack([np.nan_to_num(rgb), (~no_data).astype(float)])


def _plot_timeseries(out_dir, roi_dir, stacks, geometry, looks, show):
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (st_i, (_, _, inside_i)) in enumerate(zip(stacks, geometry)):
        color = STACK_COLORS[i % len(STACK_COLORS)]
        for band, pol, style in ((1, "HH", "-"), (2, "HV", "--")):
            dates, values = [], []
            for product in st_i["products"]:
                with rasterio.open(roi_dir / st_i["folder"] / product["file"]) as src:
                    power = _multilook(np.abs(src.read(band)) ** 2, looks)
                dates.append(np.datetime64(product["datetime"][:19]))
                values.append(10 * np.log10(np.nanmean(power[inside_i])))
            ax.plot(dates, values, style, color=color, lw=2, marker="o", ms=5,
                    label=f"{st_i['folder'].split('_2')[0]} {pol}")

    ax.set_ylabel("Mean intensity (dB, uncalibrated)")
    ax.grid(axis="y", color="#d9d8d2", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.autofmt_xdate()
    return _save(fig, out_dir, "timeseries", show)


def _map_axes(ax, roi_bbox, aspect, pad=0.08, bounds=None):
    w, s, e, n = bounds or roi_bbox
    ax.set_xlim(w - pad, e + pad)
    ax.set_ylim(s - pad, n + pad)
    ax.set_aspect(aspect)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(color="#d9d8d2", lw=0.5)
    ax.set_axisbelow(True)


def _draw_outlines(ax, roi, common):
    ax.plot(*roi.exterior.xy, "--", color=INK, lw=1.2)
    for part in getattr(common, "geoms", [common]):
        ax.plot(*part.exterior.xy, color="white", lw=1.5, path_effects=HALO)

