"""Figures of the extracted stacks, kept scenes only (radar geometry, no ground projection).

- <roi>/extraction/<stack>/figures/hh_dates.png  HH intensity |HH|² of every kept scene, common area
- <roi>/extraction/figures/timeseries.png        mean HH / HV intensity in the common area per kept scene
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from ..core.io import load_stack_info
from .common import CMAP, multilook, save

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

STACK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
LOOKS = (12, 2)  # azimuth x range multilook for display


def _inside(info, looks):
    with rasterio.open(info["_dir"] / "overlap_mask.tif") as src:
        return multilook(src.read(1).astype(np.float32), looks) > 0.5


def plot_hh_dates(info, looks=LOOKS, show=False):
    """HH intensity of every kept scene of a stack, same colour scale, crop pixel axes."""
    inside = _inside(info, looks)
    hh_db = []
    for product in info["products"]:
        with rasterio.open(info["_dir"] / product["file"]) as src:
            db = 10 * np.log10(multilook(np.abs(src.read(1)) ** 2, looks))
        db[~inside] = np.nan
        hh_db.append(db)
    if not hh_db:
        return None
    vmin, vmax = np.nanpercentile(np.stack(hh_db), [2, 98])

    sp = info["spacings"]
    px_aspect = sp["azimuth_m"] / sp["ground_range_m"]          # ground shape of one crop pixel
    rows, cols = info["window"]["height"], info["window"]["width"]
    panel_ratio = rows * px_aspect / cols
    n_cols = 7 if panel_ratio > 1.5 else 4
    n_rows = -(-len(hh_db) // n_cols)
    extent = [0, looks[1] * hh_db[0].shape[1], looks[0] * hh_db[0].shape[0], 0]

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharex=True, sharey=True, layout="constrained",
                             figsize=(2.3 * n_cols + 1.5, 2.3 * min(panel_ratio, 4) * n_rows + 1))
    for ax, product, db in zip(axes.flat, info["products"], hh_db):
        mesh = ax.imshow(db, cmap=CMAP, vmin=vmin, vmax=vmax, extent=extent, aspect=px_aspect,
                         interpolation="nearest")
        ax.set_title(f"{product['datetime'][:10]}  C{int(product['repeat_cycle']):02d}", loc="left", fontsize=9)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(hh_db):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("azimuth pixel", fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("range pixel", fontsize=7)
    fig.suptitle(f"{info['_dir'].name}: HH intensity |HH|² in the common area, {looks[0]} x {looks[1]} looks "
                 f"(az x rg), kept scenes", x=0.01, ha="left")
    fig.colorbar(mesh, ax=axes, shrink=0.8, label="HH intensity |HH|² [dB, uncalibrated]")
    return save(fig, info["_dir"] / "figures", "hh_dates", show)


def plot_timeseries(infos, out_dir, looks=LOOKS, show=False):
    """Mean HH / HV intensity inside the common area, per kept scene and per stack."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, info in enumerate(infos):
        color = STACK_COLORS[i % len(STACK_COLORS)]
        inside = _inside(info, looks)
        for band, pol, style in ((1, "HH", "-"), (2, "HV", "--")):
            dates, values = [], []
            for product in info["products"]:
                with rasterio.open(info["_dir"] / product["file"]) as src:
                    power = multilook(np.abs(src.read(band)) ** 2, looks)
                dates.append(np.datetime64(product["datetime"][:19]))
                values.append(10 * np.log10(np.nanmean(power[inside])))
            ax.plot(dates, values, style, color=color, lw=2, marker="o", ms=5,
                    label=f"{info['_dir'].name.split('_2')[0]} {pol}")
    ax.set_ylabel("mean intensity |HH|², |HV|² in the common area\n[dB, uncalibrated]")
    ax.set_title("Mean HH (solid) and HV (dashed) intensity per kept scene", loc="left", fontsize=10)
    ax.grid(axis="y", color="#d9d8d2", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.autofmt_xdate()
    return save(fig, out_dir, "timeseries", show)


def plot_stack_outputs(roi_paths, show=False):
    """HH per kept scene of every stack of the last extraction and the ROI time series."""
    infos = [load_stack_info(d, kept_only=True) for d in roi_paths.stack_dirs()]
    infos = [info for info in infos if info["products"]]
    if not infos:
        print(f"No extracted products in {roi_paths.extraction}")
        return []
    figures = []
    for info in infos:
        print(f"  {info['_dir'].name}:")
        figures.append(plot_hh_dates(info, show=show))
    figures.append(plot_timeseries(infos, roi_paths.figures, show=show))
    return [f for f in figures if f is not None]
