"""Figure helpers shared by every plot. All maps are in radar geometry (crop pixel axes)."""

import os
import sys
import threading
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

CMAP = "jet"  # one classic colour map for every scalar map and matrix
NON_GUI_BACKENDS = {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"}


def save(fig, out_dir, tag, show=False, dpi=300, close=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"    saved {out}")
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return out


def cell_geometry(win, az_index, rg_index):
    """Extent (crop pixels) and imshow aspect of maps on the averaging grid (coherence, PolSAR).

    One output cell = d_az x d_rg crop pixels; px_aspect = ground height / width of one crop
    pixel (azimuth vs ground range), so the cells are drawn with their true ground shape.
    """
    if az_index.size >= 2 and rg_index.size >= 2:
        d_az, d_rg = float(np.diff(az_index[:2])[0]), float(np.diff(rg_index[:2])[0])
    else:  # single-cell grid
        d_az, d_rg = float(win["subs_F_a"]), float(win["subs_F_r"])
    return {
        "extent": [rg_index[0] - d_rg / 2, rg_index[-1] + d_rg / 2,
                   az_index[-1] + d_az / 2, az_index[0] - d_az / 2],
        "px_aspect": (win["output_spacing_az_m"] / d_az) / (win["output_spacing_rg_m"] / d_rg),
        "aspect": win["output_spacing_az_m"] / win["output_spacing_rg_m"],  # of one output cell
    }


def show_map(ax, img, geo, good=None, labels=True, **kw):
    """(Naz, Nrg) image or RGB(A) in radar geometry; 2-D images are NaN outside `good`."""
    if img.ndim == 2 and good is not None:
        img = np.where(good, img, np.nan)
    im = ax.imshow(img, extent=geo["extent"], aspect=geo["px_aspect"], interpolation="nearest", **kw)
    ax.tick_params(labelsize=6)
    if labels:
        ax.set_xlabel("range pixel", fontsize=7)
        ax.set_ylabel("azimuth pixel", fontsize=7)
    return im


def multilook(a, looks):
    """Block average, ignoring NaN (no-data)."""
    la, lr = looks
    h, w = a.shape[0] // la * la, a.shape[1] // lr * lr
    return np.nanmean(a[:h, :w].reshape(h // la, la, w // lr, lr), axis=(1, 3))


def hsi_to_rgb(hue_deg, saturation, intensity):
    """HSI -> RGB (Gonzalez & Woods sector formulas), arrays of any shape -> (..., 3) in [0, 1].

    hue [deg, 0 = red, 120 = green, 240 = blue], saturation and intensity in [0, 1].
    """
    h = np.mod(np.asarray(hue_deg, dtype=np.float64), 360.0)
    s = np.clip(np.asarray(saturation, dtype=np.float64), 0, 1)
    i = np.clip(np.asarray(intensity, dtype=np.float64), 0, 1)
    rgb = np.zeros(h.shape + (3,))
    for sector, (first, second, third) in enumerate(((0, 1, 2), (1, 2, 0), (2, 0, 1))):
        # sector 0: RG (0-120 deg), 1: GB (120-240 deg), 2: BR (240-360 deg)
        sel = (h >= 120 * sector) & (h < 120 * (sector + 1))
        hs = np.deg2rad(h[sel] - 120 * sector)
        low = i[sel] * (1 - s[sel])
        lead = i[sel] * (1 + s[sel] * np.cos(hs) / np.cos(np.pi / 3 - hs))
        rgb[sel, third] = low
        rgb[sel, first] = lead
        rgb[sel, second] = 3 * i[sel] - (low + lead)
    bad = ~(np.isfinite(h) & np.isfinite(s) & np.isfinite(i))
    rgb[bad] = np.nan
    return np.clip(rgb, 0, 1)


def gui_available():
    """True when figures can be shown in desktop windows (interactive backend and a display)."""
    if matplotlib.get_backend().lower() in NON_GUI_BACKENDS or "inline" in matplotlib.get_backend().lower():
        return False
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    return True


def fit_to_screen(fig, max_w=13.0, max_h=8.5):
    """Shrink a (saved) figure so that its window fits on a laptop screen."""
    w, h = fig.get_size_inches()
    scale = min(1.0, max_w / w, max_h / h)
    if scale < 1.0:
        fig.set_size_inches(w * scale, h * scale)


def ask_with_figures(figs, prompt):
    """Show figures in desktop windows and ask in the terminal; the windows stay zoomable while typing.

    Without a GUI the question is asked anyway (the figures are on disk). Returns the answer text.
    """
    shown = bool(figs) and gui_available()
    if shown:
        for fig in figs:
            fit_to_screen(fig)
        plt.show(block=False)
        plt.pause(0.3)
    answer = {}

    def read():
        try:
            answer["text"] = input(prompt)
        except EOFError:
            answer["text"] = ""

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    # keep the GUI event loop running (zoom / pan) without raising the windows over the terminal
    with plt.rc_context({"figure.raise_window": False}):
        while thread.is_alive():
            if shown and plt.get_fignums():
                plt.pause(0.15)
            else:
                thread.join(0.15)
    return answer.get("text", "")
