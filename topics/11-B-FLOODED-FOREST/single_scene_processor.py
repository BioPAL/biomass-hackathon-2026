"""

process a single scene


load a given stack file 
do a pauli decomposition 

water component 
"""





import json
import urllib.request
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt


# 1. Load the JSON file
json_path = "./sta_level1c_by_roi.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)


# 2. Get your access token (using your existing functions)
offline_token, client_id, client_secret = load_credentials(CREDENTIALS_PATH)
access_token = get_maap_access_token(offline_token, client_id, client_secret)

ABS_KEY = "enclosure_i_abs_tiff"
PHASE_KEY = "enclosure_i_phase_tiff"
TILE = 512            # subset size in pixels
GRID = 3              # 3x3 grid
LOOKS = 1             # boxcar multilook window for speckle reduction (1 = off)
ABS_IS_INTENSITY = False   # set True if the abs asset is power rather than amplitude
BAND_ORDER = ("HH", "HV", "VH", "VV")  # verify against the printed band descriptions




def get_href(asset):
    return asset.get("href") if isinstance(asset, dict) else asset


def to_complex(amp, phase):
    if ABS_IS_INTENSITY:
        amp = np.sqrt(amp)
    # Auto-detect degrees vs radians
    if np.nanmax(np.abs(phase)) > 2 * np.pi + 0.1:
        phase = np.deg2rad(phase)
    return (amp * np.exp(1j * phase)).astype("complex64")


def pauli_decomposition(hh, hv, vv):
    """Pauli vector k = 1/sqrt(2) * [HH+VV, HH-VV, 2HV]. Returns powers."""
    p_surface = np.abs(hh + vv) ** 2 / 2      # odd-bounce: water, bare surface
    p_double = np.abs(hh - vv) ** 2 / 2       # double-bounce: trunk-ground, flooded forest
    p_volume = 2 * np.abs(hv) ** 2            # volume: canopy
    if LOOKS > 1:                              # multilook the power to suppress speckle
        p_surface, p_double, p_volume = (
            uniform_filter(p, LOOKS) for p in (p_surface, p_double, p_volume)
        )
    return p_surface, p_double, p_volume


def stretch(p, lo=2, hi=98):
    db = 10 * np.log10(p + 1e-12)
    a, b = np.nanpercentile(db, [lo, hi])
    return np.clip((db - a) / (b - a + 1e-12), 0, 1)


def tile_windows(width, height):
    xs = np.linspace(0, width - TILE, GRID).astype(int)
    ys = np.linspace(0, height - TILE, GRID).astype(int)
    return [Window(x, y, TILE, TILE) for y in ys for x in xs]


def read_channels(src_a, src_p, win):
    """Read all pol channels for a window and return complex HH, HV, VV."""
    n = src_a.count
    names = BAND_ORDER if n == 4 else ("HH", "HV", "VV")
    idx = {nm: i + 1 for i, nm in enumerate(names)}
    ch = {nm: to_complex(src_a.read(i, window=win).astype("float32"),
                         src_p.read(i, window=win).astype("float32"))
          for nm, i in idx.items()}
    hv = (ch["HV"] + ch["VH"]) / 2 if "VH" in ch else ch["HV"]   # reciprocity
    return ch["HH"], hv, ch["VV"]


def read_pauli_tiles(abs_url, phase_url, token):
    hdr = f"Authorization: Bearer {token}"
    with rasterio.Env(GDAL_HTTP_HEADERS=hdr, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(f"/vsicurl/{abs_url}") as sa, rasterio.open(f"/vsicurl/{phase_url}") as sp:
            print("Abs bands:", sa.count, sa.dtypes)
            print("Phase bands:", sp.count, sp.dtypes)
            assert (sa.width, sa.height) == (sp.width, sp.height), "abs/phase grids differ"
            tiles = []
            for win in tile_windows(sa.width, sa.height):
                hh, hv, vv = read_channels(sa, sp, win)
                add a mask that removes the 9999
                # tiles.append((win, *pauli_decomposition(hh, hv, vv)))
                tiles.append((win, (hh, hv, vv)))
            return tiles



plt.matshow(db_data(np.abs(tiles[0][1][0])))



def plot_grid(tiles, render, title, cmap=None, vmin=None, vmax=None, cbar_label=None):
    fig, axes = plt.subplots(GRID, GRID, figsize=(12, 12))
    im = None
    for ax, t in zip(axes.ravel(), tiles):
        im = ax.imshow(render(*t[1:]), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"x={t[0].col_off}, y={t[0].row_off}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    if cbar_label and im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label=cbar_label)
    plt.show()


def plot_all(tiles, title):
    # 1. Pauli RGB: R = double-bounce, G = volume, B = surface
    plot_grid(tiles, lambda s, d, v: np.dstack([stretch(d), stretch(v), stretch(s)]),
              f"{title}\nPauli RGB (R=double, G=volume, B=surface)")

    # 2-4. Individual components in dB
    for name, k in (("Surface (HH+VV)", 0), ("Double-bounce (HH-VV)", 1), ("Volume (HV)", 2)):
        plot_grid(tiles, lambda s, d, v, k=k: 10 * np.log10((s, d, v)[k] + 1e-12),
                  f"{title}\n{name} [dB]", cmap="viridis", vmin=-30, vmax=0, cbar_label="dB")

    # 5. Dominant mechanism: 0 = surface/water, 1 = double-bounce, 2 = volume/forest
    def dominant(s, d, v):
        return np.argmax(np.stack([s, d, v]), axis=0)
    plot_grid(tiles, dominant, f"{title}\nDominant mechanism (blue=surface/water, red=double, green=volume/forest)",
              cmap=plt.matplotlib.colors.ListedColormap(["tab:blue", "tab:red", "tab:green"]), vmin=0, vmax=2)




# --- main loop ---
for roi in data.get("rois", []):
    products = roi.get("products", [])
    print(f"ROI: {roi['name']} | Products found: {len(products)}")
    if not products:
        continue

    product = products[0]
    product_date = product.get("datetime")
    abs_url = get_href(product["assets"].get(ABS_KEY))
    phase_url = get_href(product["assets"].get(PHASE_KEY))
    if not (abs_url and phase_url):
        print(f"  -> Missing {ABS_KEY} or {PHASE_KEY}")
        continue

    tiles = read_pauli_tiles(abs_url, phase_url, access_token)
    plot_all(tiles, f"ROI: {roi['name']} | {product_date}")

    dates = [p.get("datetime") for p in products]
    print(f"Total products: {len(dates)} | First: {dates[0]} | Last: {dates[-1]}")

    break




Pauli RGB, with double-bounce in red, volume in green and surface in blue.
Surface (HH+VV), double-bounce (HH−VV) and volume (HV) as separate dB maps.
Dominant mechanism, which is the simplest water vs forest map: blue is surface/water, green is volume/forest, red is double-bounce (trunk-ground or flooded forest).

Things to check:

Amplitude vs intensity. I assumed abs is amplitude. If it’s power, set ABS_IS_INTENSITY = True, otherwise all the dB values will be doubled.
Band order. The printout shows the band descriptions. The code expects HH, HV, VH, VV for 4 bands and HH, HV, VV for 3. HV and VH are averaged, assuming reciprocity.
Phase units. Degrees vs radians is auto-detected. Only the relative phase between channels matters for HH±VV.
Calibration. If the data is uncalibrated or in digital numbers, the fixed dB limits (−30 to 0) will look wrong. Change vmin/vmax or let them auto-scale.
Speckle. LOOKS = 5 applies a 5×5 boxcar to the power images. Raise it for smoother maps or set it to 1 to see the raw data.
Dependencies. You’ll need rasterio and scipy.
Range requests. As before, if the server doesn’t support HTTP range requests, GDAL will download the whole file for every read. Download each file once to a temp path in that case.

Water should show as dark overall with surface-dominant returns, forest as bright green from volume scattering, and flooded forest or urban areas as red from double-bounce. If you paste the Bands: printout, I can confirm the band mapping.