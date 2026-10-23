"""Generates the Guainía notebooks: python Guainia/make_notebooks.py [00 01 02 03 04] (default: all)."""
import pathlib

import nbformat as nbf

HERE = pathlib.Path(__file__).resolve().parent

SETUP = '''import pathlib, sys
SITE_DIR = pathlib.Path.cwd()                       # the Guainia/ folder (notebooks run from here)
if not (SITE_DIR / "stack_download").is_dir():
    raise FileNotFoundError("open the notebook from the Guainia/ folder (stack_download/ not found)")
ROOT = next(p for p in [SITE_DIR, *SITE_DIR.parents] if (p / "biomass_tomo").is_dir())
sys.path.insert(0, str(ROOT))                       # the biomass_tomo package
'''

PIPELINE = ["0 · Data access", "1 · Stack", "2 · Tomography", "3 · Forest height", "4 · Polarimetric analysis"]


def banner(k):
    return "**Pipeline:** " + " → ".join(f"**{s}**" if i == k else s for i, s in enumerate(PIPELINE))


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (tomosar)", "language": "python"}
    for kind, src in cells:
        book.cells.append(nbf.v4.new_markdown_cell(src.strip()) if kind == "md" else nbf.v4.new_code_cell(src.strip()))
    return book


# =====================================================================================================
NB0 = [
("md", r"""
# 0 · Data access: BIOMASS L1c tomographic stacks from the ESA MAAP

""" + banner(0) + r"""

Every L1c `S1/S2/S3_STA__1S` product is one acquisition of a tomographic stack. All acquisitions of a stack are coregistered onto a common primary and share one `eofeos:stack_id`. This notebook
1. searches the MAAP STAC catalogue for stack products over an area of interest;
2. groups them into stacks (same swath, major cycle, track, frame and version) and **visualises** them: footprints on a map, acquisition timeline, quick-looks;
3. completes the chosen stack, because a wide search can return only part of a stack;
4. **downloads** it into `stack_download/`, in the folder layout that notebook 1 reads.

The search is anonymous. Only the download needs credentials (section 1).
"""),
("md", r"""
## 1. Configuration

**Area and time.** `AOI_BBOX` is (W, S, E, N) in degrees. The default is a 28 km box in the centre of the Guainía frame, so the stack used in notebooks 1–4 is found again. Keep the AOI smaller than a frame: frames are tilted and about 60 km wide, so a box around a whole frame is never fully covered by it.

**Stack choice.** `STACK_KEY` picks a stack by its key, e.g. `S1_M03_T029_F152_v02` (swath, major cycle, track, frame, version). `None` takes the stack with the most acquisitions that fully covers the AOI.

**Download.**
- `DOWNLOAD = False` only searches and visualises. Set it to `True` to download.
- `DOWNLOAD_MODE = "zip"` fetches each whole product (about 0.5 GB) and unzips it to `stack_download/<product>_product/`.
- `DOWNLOAD_MODE = "assets"` fetches only the files the processing needs (SLC tiffs, VRT, annotations, LUT, orbit/attitude), skipping quick-looks and schemas.
- Products already on disk are skipped, and an interrupted download resumes when the cell is re-run.

**Credentials** (download only) live in two files in `TomoSAR_Scripts/`, which are never printed:
- `token.txt`: your MAAP offline token;
- `maap_credentials.txt`: `CLIENT_ID=offline-token` and `CLIENT_SECRET=...`, the public value from the ESA course notebook. Copy `maap_credentials.example.txt` to create it.
"""),
("code", SETUP + r'''
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import box, shape, mapping

from biomass_tomo import maap

AOI_BBOX = (-70.10, 3.30, -69.85, 3.55)     # (W, S, E, N) lon/lat; central Guainía
DATETIME = "2026-04-01/2026-12-31"          # STAC interval "start/end"; None = all
SEARCH_MARGIN_DEG = 0.5                      # search box = AOI plus this margin, to see neighbouring stacks
MIN_ACQUISITIONS = 3
STACK_KEY = "S1_M03_T029_F152_v02"           # None = the stack with most acquisitions fully covering the AOI

DOWNLOAD = False
DOWNLOAD_MODE = "zip"                        # "zip" (whole product, unzipped) or "assets" (needed files only)
DOWNLOAD_DIR = SITE_DIR / "stack_download"   # notebook 1 reads its products from here
TOKEN_FILE = ROOT / "token.txt"
CREDENTIALS_FILE = ROOT / "maap_credentials.txt"

aoi = box(*AOI_BBOX)
'''),
("md", r"""
## 2. Catalogue search and stacks
All L1c stack products intersecting the search box are grouped into stacks. For each stack the table gives:
- the number of acquisitions **found** (`n_acquisitions`) and their repeat cycles;
- `aoi_coverage`: the fraction of the AOI inside the footprint common to all found acquisitions. Only stacks with coverage 1 image the whole AOI in every acquisition.
"""),
("code", r'''
items = maap.search_l1c(bbox=aoi.buffer(SEARCH_MARGIN_DEG).bounds, datetime=DATETIME)
stacks = maap.group_stacks(items, aoi=aoi)
table = pd.DataFrame([{"key": s["key"], "n_acquisitions": s["n_acquisitions"], "cycles": s["cycles"],
                       "first": pd.to_datetime(min(s["dates"])), "last": pd.to_datetime(max(s["dates"])),
                       "aoi_coverage": round(s["aoi_coverage"], 3), "orbit": s["items"][0].properties.get("sat:orbit_state"),
                       "mission_phase": s["mission_phase"]} for s in stacks])
print(f"{len(items)} products in {len(stacks)} stacks")
table.sort_values(["aoi_coverage", "n_acquisitions"], ascending=False, ignore_index=True)
'''),
("md", r"""
### Stack footprints
Each stack's footprint (the first acquisition's) is drawn on OpenStreetMap: blue for stacks, red for the selected one, black dashed for the AOI. Hover for the key, the number of acquisitions and the coverage. Stacks of different swaths (S1–S3) overlap across track. Stacks of different frames follow each other along track.
"""),
("code", r'''
import folium
candidates = [s for s in stacks if s["n_acquisitions"] >= MIN_ACQUISITIONS and (s["aoi_coverage"] or 0) > 0.99]
if STACK_KEY:
    chosen = next((s for s in stacks if s["key"] == STACK_KEY), None)
    if chosen is None:
        raise KeyError(f"{STACK_KEY} not among the stacks found; available: {[s['key'] for s in stacks]}")
else:
    chosen = max(candidates, key=lambda s: (s["aoi_coverage"], s["n_acquisitions"]), default=None)
print("selected:", chosen["key"] if chosen else None)

m = folium.Map(location=[aoi.centroid.y, aoi.centroid.x], zoom_start=7, tiles="OpenStreetMap")
for s in sorted(stacks, key=lambda s: chosen is not None and s["key"] == chosen["key"]):
    sel = chosen is not None and s["key"] == chosen["key"]
    folium.GeoJson(mapping(shape(s["items"][0].geometry)),
                   tooltip=f'{s["key"]}: {s["n_acquisitions"]} acq, AOI coverage {s["aoi_coverage"]:.2f}',
                   style_function=lambda _, c=("red" if sel else "#3388ff"), w=(3 if sel else 1):
                       {"color": c, "weight": w, "fillOpacity": 0.05}).add_to(m)
folium.GeoJson(mapping(aoi), tooltip="AOI", style_function=lambda _: {"color": "black", "weight": 2, "dashArray": "5,5"}).add_to(m)
m.fit_bounds([[aoi.bounds[1] - SEARCH_MARGIN_DEG, aoi.bounds[0] - SEARCH_MARGIN_DEG],
              [aoi.bounds[3] + SEARCH_MARGIN_DEG, aoi.bounds[2] + SEARCH_MARGIN_DEG]])
m
'''),
("md", r"""
### Acquisition timeline
One row per stack and one marker per acquisition. The selected stack is red and its coregistration primary is ringed. In the tomographic phase the revisit is 3 days, so a full 7-acquisition stack spans 18 days.
"""),
("code", r'''
order = sorted(stacks, key=lambda s: s["key"])
fig, ax = plt.subplots(figsize=(10, 0.35 * len(order) + 1.2), constrained_layout=True)
for y, s in enumerate(order):
    sel = chosen is not None and s["key"] == chosen["key"]
    t = pd.to_datetime(s["dates"])
    ax.plot(t, [y] * len(t), "o-", color="red" if sel else "tab:blue", ms=5, lw=0.8)
    prim = [pd.to_datetime(maap.parse_id(s["primary"])["start"])] if s["primary"] else []
    ax.plot(prim, [y] * len(prim), "o", mfc="none", mec="k", ms=10)
ax.set_yticks(range(len(order)), [f'{s["key"]} ({s["n_acquisitions"]})' for s in order], fontsize=7)
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
ax.set_xlabel("acquisition date"); ax.grid(alpha=0.3, axis="x")
plt.show()
'''),
("md", r"""
## 3. Selected stack
A wide search can stop at `max_items`, and a stack may straddle the search box, so the selected stack is **completed**. A second query runs over the centre of its footprint and the time window encoded in its `stack_id`, keeping only products with that `stack_id`. The table lists every acquisition, with the coregistration primary and whether it is already on disk.
"""),
("code", r'''
if chosen is None:
    raise RuntimeError("no stack selected: set STACK_KEY or change the AOI / MIN_ACQUISITIONS")
n_before = chosen["n_acquisitions"]
chosen = maap.complete_stack(chosen)
print(f"{chosen['key']}: {n_before} acquisitions found in the search, {chosen['n_acquisitions']} after completion")
print("stack_id:", chosen["stack_id"])
acq_table = pd.DataFrame([{
    "cycle": int(maap.parse_id(it.id)["cycle"]),
    "datetime": it.datetime,
    "primary": bool(it.properties.get("eofeos:is_coregistration_primary")),
    "absolute_orbit": it.properties.get("sat:absolute_orbit"),
    "size_GB": round(sum(a.extra_fields.get("file:size", 0) or 0 for a in it.assets.values()) / 1e9, 2),
    "on_disk": maap.product_on_disk(DOWNLOAD_DIR, it.id) is not None,
    "product_id": it.id} for it in chosen["items"]])
print(f"download size ≈ {acq_table.size_GB.sum():.1f} GB; already on disk: {acq_table.on_disk.sum()}/{len(acq_table)}")
acq_table
'''),
("md", "### Quick-looks\nThe catalogue quick-look of every acquisition, displayed straight from the MAAP (nothing is downloaded). Similar scenes across the stack mean similar conditions. Rain or flooding on one date shows up as a change in brightness."),
("code", r'''
from IPython.display import HTML, display
cells = []
for it in chosen["items"]:
    tag = " (primary)" if it.properties.get("eofeos:is_coregistration_primary") else ""
    url = it.assets["quicklook_jpg"].href if "quicklook_jpg" in it.assets else None
    cycle = maap.parse_id(it.id)["cycle"]
    cells.append(f'<div style="display:inline-block;margin:4px;text-align:center;font-size:11px">'
                 f'C{cycle} · {it.datetime:%Y-%m-%d}{tag}<br>'
                 + (f'<img src="{url}" width="150">' if url else "no quick-look") + "</div>")
display(HTML("".join(cells)))
'''),
("md", r"""
## 4. Download
Products already on disk are skipped. A fresh access token is requested whenever the current one is about to expire, so a long download does not fail half-way.
- **`zip`**: downloads the `product` asset, checks the archive and extracts it to `stack_download/<product>_product/`. The zip is then deleted.
- **`assets`**: downloads the individual files into `stack_download/<product>/` and checks each file's size against the catalogue. Unfinished files keep a `.part` suffix until they are complete.
"""),
("code", r'''
import time
todo = [it for it in chosen["items"] if maap.product_on_disk(DOWNLOAD_DIR, it.id) is None]
print(f"{len(todo)} of {chosen['n_acquisitions']} products to download into {DOWNLOAD_DIR}")
if DOWNLOAD and todo:
    session = maap.MaapSession(TOKEN_FILE, CREDENTIALS_FILE)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(todo, 1):
        t0 = time.time()
        print(f"[{i}/{len(todo)}] {it.id} ...", end=" ", flush=True)
        if DOWNLOAD_MODE == "zip":
            maap.download_zip(it, DOWNLOAD_DIR, session)
        else:
            maap.download_item(it, DOWNLOAD_DIR, session)
        print(f"{time.time() - t0:.0f} s")
elif todo:
    print("DOWNLOAD = False: nothing downloaded")
'''),
("md", r"""
## 5. Check the downloaded stack
`l1c.find_product_dirs` is the reader notebook 1 uses. All acquisitions must be found, in cycle order, each with its SLC, annotation and LUT files. The primary's product is inspected as a final check.
"""),
("code", r'''
from biomass_tomo import l1c
wanted = {it.id for it in chosen["items"]}
try:
    found = [p for p in l1c.find_product_dirs(DOWNLOAD_DIR) if p.name in wanted]
except FileNotFoundError:
    found = []
print(f"{len(found)} of {len(wanted)} products of {chosen['key']} in {DOWNLOAD_DIR}")
for p in found:
    files = {sd: len(list((p / sd).glob("*"))) for sd in ("measurement", "annotation_coregistered", "annotation_primary")}
    print(f"  C{maap.parse_id(p.name)['cycle']}  {files}  primary={l1c.is_primary(p)}")
if len(found) == len(wanted):
    print("complete: run 01_Stack_Build with STACK_DIR =", DOWNLOAD_DIR)
    l1c.inspect_product(next(p for p in found if l1c.is_primary(p)))
else:
    missing = sorted(wanted - {p.name for p in found})
    print("missing:", *missing, sep="\n  ")
'''),
]

# =====================================================================================================
NB1 = [
("md", r"""
# 1 · Guainía: BIOMASS L1c tomographic stack → netCDF

""" + banner(1) + r"""

This notebook turns the seven BIOMASS Level-1c stack products in `stack_download/` into **one analysis-ready netCDF file**. Notebooks 2–4 read only that file and `Guainia_outputs/run_manifest.json`.

| | |
|---|---|
| Mission phase | Tomographic (TOM), major cycle 3 |
| Swath / track / frame | S1 / T029 / F152, descending |
| Acquisitions | 7 repeat cycles (C01–C07), 2026-04-24 to 2026-05-12, 3-day revisit |
| Location | Guainía, Colombia (≈ 2.85–4.14° N, 69.5–70.35° W) |
| Product | `BIO_S1_STA__1S`: quad-pol single-look complex (β⁰), coregistered onto one primary |

**Steps**
1. list the products and identify the coregistration primary;
2. check one product: files, annotation pixel grid, LUT layers and their axes;
3. write all acquisitions and LUT layers to one netCDF on the common SLC grid;
4. verify the stack: layout, no-data, LUT layers;
5. check the **baseline requirement** for tomography (vertical resolution, height of ambiguity);
6. quick-looks;
7. write the run manifest.
"""),
("md", r"""
## 1. Configuration
`CROP = None` writes the full frame (20 765 azimuth × 1 315 range samples, ≈ 6 GB of SLC plus compressed LUT layers, a few minutes). A crop is given in product pixel indices, e.g. `{"azimuth": (8000, 12000), "range": (0, 1315)}`.
"""),
("code", SETUP + r'''
import re, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from biomass_tomo import l1c
from biomass_tomo import stack as st

SITE = "guainia"
STACK_DIR = SITE_DIR / "stack_download"
POLARISATIONS = ("HH", "HV", "VH", "VV")
CROP = None                      # None = full frame; or {"azimuth": (a0, a1), "range": (r0, r1)}
MIN_ACQUISITIONS = 3             # baseline requirement: minimum number of acquisitions
Z_RANGE = (-30, 80)              # height interval to image [m]; must fit inside the height of ambiguity
OVERWRITE = False                # rebuild the netCDF even if it exists

# optional reference rasters (any CRS) for notebooks 3-4; none exist for Guainía yet
REFERENCE_CHM = None             # e.g. ("chm.tif", "band name or index")
REFERENCE_AGBD = None            # e.g. ("agbd.tif", 1)

OUT_DIR = SITE_DIR / "Guainia_outputs"
OUT_DIR.mkdir(exist_ok=True)
'''),
("md", r"""
## 2. Products
Each product folder holds
- `measurement/`: `*_i_abs.tiff` and `*_i_phase.tiff` (4 bands, HH HV VH VV), so $s = |s|\,e^{j\varphi}$;
- `annotation_coregistered/`: the annotation XML and the **LUT file** `*_lut.nc`, both on the primary's grid;
- `annotation_primary/`: the acquisition's own annotation before coregistration;
- `preview/`: quick-look PNG/KML.

The coregistration primary is the acquisition with zero normal baseline. Its LUT has no `/interferometry` group, because its coherence with itself is trivially 1.
"""),
("code", r'''
product_dirs = l1c.find_product_dirs(STACK_DIR)
rows = []
for p in product_dirs:
    ann = l1c.product_file(p, "annotation", "*_annot.xml")
    get = lambda tag: l1c.annotation_value(ann, (tag,))
    rows.append({"cycle": int(re.search(r"_C(\d\d)_", p.name).group(1)),
                 "start": pd.to_datetime(re.search(r"_1S_(\d{8}T\d{6})_", p.name).group(1)),
                 "primary": l1c.is_primary(p),
                 "normal_baseline_m": float(get("normalBaseline")),
                 "baseline_order": int(get("baselineOrderingIndex")),
                 "coregistration": get("coregistrationMethod"),
                 "product": p.name})
products = pd.DataFrame(rows)
PRIMARY = product_dirs[int(products.index[products.primary][0])]
STACK_KEY = re.search(r"BIO_(S\d)_.*_(M\d\d)_C\d\d_(T\d{3})_(F\d{3})_(\d\d)_", PRIMARY.name).groups()
STACK_KEY = "_".join(STACK_KEY[:4]) + f"_v{STACK_KEY[4]}"
print("stack:", STACK_KEY, "| primary:", PRIMARY.name)
products
'''),
("md", r"""
## 3. Product check
`inspect_product` lists the files and reads the pixel grid from the annotation (`firstSampleSlantRangeTime`, `firstLineAzimuthTime`, `rangeTimeInterval`, `azimuthTimeInterval`, pixel spacings). It also lists every variable in the LUT file.

All LUT layers are on a coarse grid (1731 × 658) whose axes are **relative azimuth time** (s since the first line) and **two-way slant-range time**. `biomass_tomo.l1c` interpolates them bilinearly onto the SLC grid. The layers written to the stack:

| stack variable | LUT path | per acquisition | used for |
|---|---|---|---|
| `dem_phases` | `/coregistration/flatteningPhaseScreen` | ✓ | removes the DEM (topographic) phase |
| `calibration_phases` | `/skpPhaseCalibration/skpCalibrationPhaseScreen` | ✓ | SKP ground-phase calibration: ground → $z=0$ |
| `insar_wavenumbers` | `/coregistration/waveNumbers` | ✓ | vertical wavenumber $k_z$: tomographic steering |
| `calibration_quality`, `coregistration_quality` | `…Quality` | ✓ | quality flags |
| `lut_coherence`, `lut_interferogram` | `/interferometry/…` | ✓ | L1c coherence with the primary (1 / 0 for the primary) |
| `noise_HH` … `noise_VV` | `/denoising/denoising{pol}` | ✓ | thermal noise power (β⁰), not removed in L1c |
| `incidence_angle`, `range_slope`, `azimuth_slope` | `/geometry/…` | | ground-range windows, terrain |
| `dem`, `latitude`, `longitude` | `/geometry/…` | | geolocation |
| `sigma_nought_factor`, `gamma_nought_factor` | `/radiometry/…` | | β⁰ → σ⁰, γ⁰ (power factors) |
"""),
("code", r'''
l1c.inspect_product(PRIMARY)
'''),
("md", "### Frame footprint\nThe frame outline from the LUT geolocation, with the processor's Pauli quick-look of the primary."),
("code", r'''
import folium
az, rg, lat_lut, *_ = l1c.lut_axes(l1c.product_file(PRIMARY, "annotation", "*_lut.nc"), "/geometry/latitude")
_, _, lon_lut, *_ = l1c.lut_axes(l1c.product_file(PRIMARY, "annotation", "*_lut.nc"), "/geometry/longitude")
edge = lambda a: np.r_[a[0, :], a[:, -1], a[-1, ::-1], a[::-1, 0]]
outline = list(zip(edge(lat_lut), edge(lon_lut)))
FRAME_BBOX = (float(np.nanmin(lon_lut)), float(np.nanmin(lat_lut)), float(np.nanmax(lon_lut)), float(np.nanmax(lat_lut)))
print("frame bbox (W, S, E, N):", tuple(round(v, 4) for v in FRAME_BBOX))

m = folium.Map(location=[np.nanmean(lat_lut), np.nanmean(lon_lut)], zoom_start=8)
folium.Polygon(outline, color="red", weight=2, fill=False, tooltip=STACK_KEY).add_to(m)
folium.Marker([lat_lut[0, 0], lon_lut[0, 0]], tooltip="first line, near range").add_to(m)
m
'''),
("code", r'''
ql = sorted((PRIMARY / "preview").glob("*_l1c_ql_pauli.png"))
if ql:
    img = plt.imread(ql[0])
    fig, ax = plt.subplots(figsize=(4, 9), constrained_layout=True)
    ax.imshow(img, aspect="auto"); ax.set_title("Pauli quick-look (processor)", fontsize=9); ax.axis("off")
    plt.show()
'''),
("md", r"""
## 4. Build the stack netCDF
Every acquisition is read on the same pixel grid, because the products are already coregistered. Layout (`biomass_tomo.l1c.build_stack`, identical to the course stack):

| variable | dimensions | content |
|---|---|---|
| `slc_real`, `slc_imag` | (polarisation, acquisition, azimuth, range) | β⁰ SLC, $s = $ `slc_real + 1j*slc_imag`, no-data → 0 |
| per-acquisition LUT layers | (acquisition, azimuth, range) | see the table above |
| single LUT layers | (azimuth, range) | geometry, radiometry |
| `azimuth`, `range` | | along-track position, slant range [m] |
| `azimuth_line`, `range_sample` | | pixel indices in the original frame |
| `temporal_baseline`, `acquisition_time`, `product_id` | (acquisition) | |

LUT values are stored **as delivered**. Their sign convention is determined from the data in notebook 2.
"""),
("code", r'''
STACK_PATH = OUT_DIR / f"{SITE}_{STACK_KEY}_stack.nc"
if STACK_PATH.exists() and not OVERWRITE:
    print("exists, not rebuilt:", STACK_PATH)
else:
    t0 = time.time()
    last = [0.0]
    def progress(msg):
        if msg.startswith("  SLC") and time.time() - last[0] < 20:
            return
        last[0] = time.time()
        print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)
    l1c.build_stack(STACK_DIR, STACK_PATH, crop=CROP, polarisations=POLARISATIONS, site=SITE, progress=progress)
    print(f"done in {time.time() - t0:.0f} s")
print(f"{STACK_PATH.name}: {STACK_PATH.stat().st_size / 1e9:.2f} GB")
'''),
("md", "## 5. Verification\nDimensions, global attributes, every variable with its source LUT and units, and the no-data count per channel."),
("code", r'''
ds = st.open_stack(STACK_PATH)
print(dict(ds.sizes))
for k in ("title", "processor", "swath", "polarisation_sequence", "crop_azimuth", "crop_range", "history"):
    print(f"  {k:22s} {ds.attrs.get(k)}")
pd.DataFrame([{"variable": n, "dims": ", ".join(v.dims), "dtype": str(v.dtype), "units": v.attrs.get("units", ""),
               "source": v.attrs.get("source", "")} for n, v in ds.data_vars.items()]).set_index("variable")
'''),
("code", r'''
nodata = pd.DataFrame(ds.nodata_pixels.values / (ds.sizes["azimuth"] * ds.sizes["range"]) * 100,
                      index=ds.polarisation.values, columns=[f"C{c:02d}" for c in ds.acquisition.values])
print("no-data pixels [% of frame] (unfocused samples at the frame edges, see the black border in the quick-looks)")
nodata.round(2)
'''),
("md", r"""
### LUT layers on the SLC grid
The phase layers are shown for the primary and for the acquisition with the largest $|k_z|$. For the primary, the flattening phase and $k_z$ must be zero. The flattening phase grows with baseline and follows topography. The SKP screen is the ground phase estimated per pixel from the Sum-of-Kronecker-Products decomposition. It combines the slowly varying residual errors (orbit, ionosphere, troposphere) with fine structure that follows land cover and DEM errors, which is why it is not smooth.
"""),
("code", r'''
REF = st.reference_index(ds)
kz_scene = ds.insar_wavenumbers.isel(azimuth=slice(None, None, 200), range=slice(None, None, 50)).mean(("azimuth", "range")).values
FAR = int(np.argmax(np.abs(kz_scene)))
dec = dict(azimuth=slice(None, None, 20), range=slice(None, None, 2))
ext = [0, ds.sizes["azimuth"], 0, ds.sizes["range"]]
layers = [("dem_phases", REF, "twilight"), ("dem_phases", FAR, "twilight"), ("calibration_phases", FAR, "twilight"),
          ("insar_wavenumbers", FAR, "viridis"), ("incidence_angle", None, "viridis"), ("dem", None, "terrain"),
          ("sigma_nought_factor", None, "viridis"), ("noise_HV", REF, "magma")]
fig, axes = plt.subplots(len(layers), 1, figsize=(11, 1.55 * len(layers)), sharex=True, constrained_layout=True)
for ax, (name, ia, cmap) in zip(axes, layers):
    da = ds[name].isel(**dec) if ia is None else ds[name].isel(acquisition=ia, **dec)
    v = da.values.T
    if name in ("dem_phases", "calibration_phases"):
        v = np.angle(np.exp(1j * v)); kw = dict(vmin=-np.pi, vmax=np.pi)
    elif name == "noise_HV":
        v = 10 * np.log10(v); kw = {}
    else:
        kw = dict(vmin=np.nanpercentile(v, 1), vmax=np.nanpercentile(v, 99))
    im = ax.imshow(v, aspect="auto", origin="lower", cmap=cmap, extent=ext, interpolation="none", **kw)
    lab = name + ("" if ia is None else f"  C{int(ds.acquisition[ia]):02d}") + (" (wrapped)" if "phases" in name else "")
    ax.set_title(lab + (" [dB]" if name == "noise_HV" else f" [{ds[name].attrs.get('units', '')}]"), fontsize=8, loc="left")
    ax.set_ylabel("range", fontsize=8); fig.colorbar(im, ax=ax, pad=0.01, aspect=8)
axes[-1].set_xlabel("azimuth line")
plt.show()
'''),
("md", r"""
## 6. Baseline requirement
Tomography needs several acquisitions with different vertical wavenumbers $k_z^i$, where $k_z^i$ converts height into interferometric phase with the primary ($\phi = k_z z$).
- Rayleigh **vertical resolution**: $\delta z = 2\pi/\Delta k_z$, with $\Delta k_z$ the total span of the $k_z$ values.
- **Height of ambiguity**: $z_{amb} = 2\pi/\min_i |k_z^{i+1}-k_z^i|$. Heights further apart than $z_{amb}$ fold onto each other, so `Z_RANGE` must fit inside it.

$k_z$ varies with incidence angle, so both are also shown across range.
"""),
("code", r'''
report = st.baseline_report(ds, min_acquisitions=MIN_ACQUISITIONS, z_extent=Z_RANGE)
for k in ("n_acquisitions", "cycles", "kz_mean_rad_per_m", "kz_span", "vertical_resolution_m",
          "vertical_resolution_m_p5_p95", "height_of_ambiguity_m", "reference_acquisition_index", "tomography_ok", "problems"):
    v = report[k]
    print(f"{k:30s} {np.round(v, 4).tolist() if isinstance(v, list) and v and isinstance(v[0], float) else v}")

kz_rg = ds.insar_wavenumbers.isel(azimuth=slice(None, None, 100)).mean("azimuth").values      # (acq, range)
inc_rg = ds.incidence_angle.isel(azimuth=slice(None, None, 100)).mean("azimuth").values
kz_sorted = np.sort(kz_rg, axis=0)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
axes[0].plot(report["temporal_baseline_days"], report["kz_mean_rad_per_m"], "o-")
axes[0].plot(report["temporal_baseline_days"][REF], report["kz_mean_rad_per_m"][REF], "ro", mfc="none", ms=12, label="primary")
axes[0].set_xlabel("temporal baseline [days]"); axes[0].set_ylabel("mean $k_z$ [rad/m]"); axes[0].legend()
for i in range(kz_rg.shape[0]):
    axes[1].plot(inc_rg, kz_rg[i], label=f"C{int(ds.acquisition[i]):02d}")
axes[1].set_xlabel("incidence angle [deg]"); axes[1].set_ylabel("$k_z$ [rad/m]"); axes[1].legend(fontsize=7, ncol=2)
axes[2].plot(inc_rg, 2 * np.pi / (kz_sorted[-1] - kz_sorted[0]), label=r"resolution $\delta z$")
axes[2].plot(inc_rg, 2 * np.pi / np.min(np.diff(kz_sorted, axis=0), axis=0), label=r"ambiguity $z_{amb}$")
axes[2].axhspan(0, Z_RANGE[1] - Z_RANGE[0], color="0.9", label="Z_RANGE extent")
axes[2].set_xlabel("incidence angle [deg]"); axes[2].set_ylabel("[m]"); axes[2].legend(fontsize=8)
for ax in axes: ax.grid(alpha=0.3)
plt.show()
if not report["tomography_ok"]:
    raise RuntimeError("Stack fails the baseline requirement: " + "; ".join(report["problems"]))
'''),
("md", r"""
## 7. Quick-looks
β⁰ intensity of the primary in HH, HV and VV, and a Pauli RGB (R $|HH-VV|^2$, G $|HV+VH|^2$, B $|HH+VV|^2$). All are averaged over 20 × 2 samples (≈ 130 m × 95 m on the ground) to suppress speckle.
"""),
("code", r'''
def block_mean(a, f=(2, 20)):                     # a: (range, azimuth)
    nr, nx = (a.shape[0] // f[0]) * f[0], (a.shape[1] // f[1]) * f[1]
    return a[:nr, :nx].reshape(nr // f[0], f[0], nx // f[1], f[1]).mean(axis=(1, 3))

load = lambda pol: st.course_layout(ds.slc_real.sel(polarisation=pol).isel(acquisition=REF)) + \
                   1j * st.course_layout(ds.slc_imag.sel(polarisation=pol).isel(acquisition=REF))
S = {p: load(p) for p in POLARISATIONS}
I = {"HH": block_mean(np.abs(S["HH"]) ** 2), "HV": block_mean(np.abs(S["HV"]) ** 2), "VV": block_mean(np.abs(S["VV"]) ** 2)}
pauli = np.stack([block_mean(np.abs(S["HH"] - S["VV"]) ** 2 / 2), block_mean(np.abs(S["HV"] + S["VH"]) ** 2 / 2),
                  block_mean(np.abs(S["HH"] + S["VV"]) ** 2 / 2)], axis=-1)
del S
ext = [0, ds.sizes["azimuth"], 0, ds.sizes["range"]]
fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True)
for ax, (p, v) in zip(axes, I.items()):
    with np.errstate(divide="ignore"):
        db = 10 * np.log10(v)
    im = ax.imshow(db, aspect="auto", origin="lower", cmap="gray", extent=ext,
                   vmin=np.nanpercentile(db[np.isfinite(db)], 2), vmax=np.nanpercentile(db[np.isfinite(db)], 98))
    ax.set_title(f"β⁰ {p} [dB], C{int(ds.acquisition[REF]):02d}", fontsize=8, loc="left"); ax.set_ylabel("range")
    fig.colorbar(im, ax=ax, pad=0.01, aspect=8)
a = np.sqrt(pauli); a = np.clip(a / np.nanpercentile(a, 98, axis=(0, 1)), 0, 1)
axes[3].imshow(a, aspect="auto", origin="lower", extent=ext)
axes[3].set_title("Pauli RGB (R: HH−VV, G: HV+VH, B: HH+VV)", fontsize=8, loc="left"); axes[3].set_ylabel("range")
axes[3].set_xlabel("azimuth line")
plt.show()
'''),
("md", r"""
## 8. Run manifest
`Guainia_outputs/run_manifest.json` holds everything notebooks 2–4 need: the stack path, the analysis crop (default: the whole stack, in stack pixel indices `[az0, az1, rg0, rg1]`), the height range, the baseline report and the optional reference rasters. Each later notebook adds its own results to it.
"""),
("code", r'''
MANIFEST = OUT_DIR / "run_manifest.json"
st.write_manifest(
    MANIFEST, site=SITE, stack_key=STACK_KEY, stack_path=str(STACK_PATH), stack_dir=str(STACK_DIR),
    product_ids=[str(p) for p in ds.product_id.values], primary=PRIMARY.name, frame_bbox=FRAME_BBOX,
    crop=[0, int(ds.sizes["azimuth"]), 0, int(ds.sizes["range"])], z_range=list(Z_RANGE), baseline=report,
    out_dir=str(OUT_DIR), reference={"chm": REFERENCE_CHM, "agbd": REFERENCE_AGBD})
ds.close()
print("written", MANIFEST)
'''),
]

# =====================================================================================================
NB2 = [
("md", r"""
# 2 · Guainía: tomography with the L1c LUTs

""" + banner(2) + r"""

Input: the stack and `Guainia_outputs/run_manifest.json` from notebook 1.

Tomography focuses the stack in height. For every multilook cell, the $N$ acquisitions form a vector $\mathbf y$ whose phases depend on the heights of the scatterers through the vertical wavenumbers $k_z^i$. Before that relation holds, two phase terms supplied in the LUT must be removed:

| LUT layer (stack variable) | what it removes | after removal |
|---|---|---|
| flattening phase screen (`dem_phases`) | phase of the DEM surface relative to the primary (topography + flat earth) | phase relative to the DEM |
| SKP calibration phase screen (`calibration_phases`) | residual phase errors common to all heights (orbit, ionosphere, troposphere, DEM error), estimated from the ground term of a Sum-of-Kronecker-Products decomposition | **ground at $z = 0$** |
| vertical wavenumbers (`insar_wavenumbers`) | | steering vector $a_i(z)=e^{-jk_z^i z}$ |

$$s^i_{corr} = s^i\,e^{j(\mathrm{SIGN\_DEM}\,\phi^i_{flat} + \mathrm{SIGN\_CAL}\,\phi^i_{SKP})}$$

**Steps**
1. load the crop; reference = coregistration primary;
2. show the interferometric phase before and after each LUT correction;
3. determine the LUT phase signs from the data;
4. multibaseline coherence matrices per polarimetric channel;
5. $k_z$ sign check, beamformer and Capon tomograms;
6. vertical profiles, transects and horizontal slices;
7. save tomograms and coherences; record the phase conventions in the manifest.
"""),
("md", r"""
## 1. Configuration
`WINDOW_M` is the multilook window **on the ground** in both directions. Slant-range spacing is converted with the incidence angle (19.8 m slant ≈ 47 m ground at 25°). Coherence and tomograms are computed on a grid decimated by half the window.
"""),
("code", SETUP + r'''
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
import warnings
warnings.filterwarnings("ignore", message="(All-NaN slice|Mean of empty slice|invalid value|divide by zero)", category=RuntimeWarning)

from biomass_tomo import stack as st, tomo

MANIFEST = SITE_DIR / "Guainia_outputs" / "run_manifest.json"
CROP = None                                  # None = crop in the manifest (default: whole stack); or (az0, az1, rg0, rg1)
WINDOW_M = 200.0                             # multilook window on the ground [m]
Z_VECTOR = np.arange(-30, 81, 2.0)           # tomographic height axis above ground [m]
D_LOAD = 0.0                                 # Capon diagonal loading
CHANNELS = {"HH": ["HH"], "VV": ["VV"], "X": ["HV", "VH"]}   # X = (HV + VH)/2, symmetrised cross-pol
SIGN_BLOCK = 3000                            # azimuth lines (crop centre) used to determine the LUT signs

cfg = st.read_manifest(MANIFEST)
CROP = list(CROP or cfg["crop"])
OUT = pathlib.Path(cfg["out_dir"]) / "tomography"
OUT.mkdir(parents=True, exist_ok=True)
print(cfg["site"], cfg["stack_key"], "| crop", CROP)
'''),
("md", "## 2. Data\nThe crop is read into the course layout (range, azimuth, acquisition). The full frame needs about 12 GB of memory."),
("code", r'''
t0 = time.time()
ds = st.open_stack(cfg["stack_path"])
REF = st.reference_index(ds)
data = st.load_crop(ds, CROP)
slc, aux = data["slc"], data["aux"]
r_ax, x_ax, acq = data["r_ax"], data["x_ax"], data["acquisition"]
dr, dx = r_ax[1] - r_ax[0], x_ax[1] - x_ax[0]
Nr, Nx, Nb = aux["insar_wavenumbers"].shape
THETA = np.deg2rad(np.nanmean(aux["incidence_angle"]))
window_r, window_x = tomo.window_samples(WINDOW_M, dr / np.sin(THETA), dx)
print(f"loaded in {time.time() - t0:.0f} s: {Nr} range x {Nx} azimuth, {Nb} acquisitions; "
      f"reference index {REF} (C{int(acq[REF]):02d})")
print(f"slant spacing {dr:.2f} m (ground {dr / np.sin(THETA):.1f} m), azimuth {dx:.2f} m, mean incidence {np.rad2deg(THETA):.1f} deg")
print(f"multilook window {window_r} x {window_x} samples (range x azimuth) = {window_r * window_x} looks")
'''),
("md", r"""
## 3. What the LUT corrections do
HH interferometric phase between the primary and the acquisition with the largest baseline, multilooked over a central block, at three stages:
1. raw: topography plus residual phase errors;
2. after the flattening screen: residual phase errors plus the height of the scattering phase centre;
3. after the SKP screen: only $k_z\,z_{pc}$ remains, where $z_{pc}$ is the phase-centre height above the ground. It is nearly uniform and its coherence is highest.

The panel uses the signs found in section 4 (the cell can be re-run after that section). Until then it uses +1.
"""),
("code", r'''
kz_scene = np.nanmean(aux["insar_wavenumbers"][::20, ::200], axis=(0, 1))
FAR = int(np.argmax(np.abs(kz_scene)))
c0 = Nx // 2
blk = slice(max(c0 - SIGN_BLOCK // 2, 0), min(c0 + SIGN_BLOCK // 2, Nx))
sd_, sc_ = globals().get("SIGN_DEM", 1), globals().get("SIGN_CAL", 1)

def multilooked_ifg(a, b):
    p = a * np.conj(b)
    w = (window_r, window_x)
    c = uniform_filter(p.real, w) + 1j * uniform_filter(p.imag, w)
    n = np.sqrt(uniform_filter(np.abs(a) ** 2, w) * uniform_filter(np.abs(b) ** 2, w))
    return c / n

s_ref, s_far = slc["HH"][:, blk, REF], slc["HH"][:, blk, FAR]
ph = {i: (sd_ * aux["dem_phases"][:, blk, i], sc_ * aux["calibration_phases"][:, blk, i]) for i in (REF, FAR)}
stages = {"raw": (s_ref, s_far),
          "+ flattening": (s_ref * np.exp(1j * ph[REF][0]), s_far * np.exp(1j * ph[FAR][0])),
          "+ flattening + SKP": (s_ref * np.exp(1j * (ph[REF][0] + ph[REF][1])), s_far * np.exp(1j * (ph[FAR][0] + ph[FAR][1])))}
ext = [x_ax[blk][0] / 1e3, x_ax[blk][-1] / 1e3, r_ax[0] / 1e3, r_ax[-1] / 1e3]
fig, axes = plt.subplots(len(stages), 2, figsize=(13, 2.3 * len(stages)), sharex=True, constrained_layout=True)
for row, (name, (a, b)) in enumerate(stages.items()):
    g = multilooked_ifg(a, b)
    im0 = axes[row, 0].imshow(np.angle(g), aspect="auto", origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi, extent=ext)
    im1 = axes[row, 1].imshow(np.abs(g), aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=1, extent=ext)
    R = np.abs(np.nanmean(np.exp(1j * np.angle(g))))
    axes[row, 0].set_title(f"phase, {name}  (uniformity R = {R:.2f})", fontsize=8, loc="left")
    axes[row, 1].set_title(f"|coherence|, {name}  (mean {np.nanmean(np.abs(g)):.2f})", fontsize=8, loc="left")
    axes[row, 0].set_ylabel("slant range [km]")
fig.colorbar(im0, ax=axes[:, 0], shrink=0.6, label="rad"); fig.colorbar(im1, ax=axes[:, 1], shrink=0.6)
for ax in axes[-1]: ax.set_xlabel("azimuth [km]")
fig.suptitle(f"HH, C{int(acq[REF]):02d} × C{int(acq[FAR]):02d}, mean $k_z$ = {kz_scene[FAR]:.3f} rad/m", fontsize=10)
plt.show()
del s_ref, s_far, stages
'''),
("md", r"""
## 4. LUT phase signs
The sign convention of the two screens is not documented, so both are found from the data on the temporal-neighbour pair (the shortest baseline, with little volume decorrelation), over the central `SIGN_BLOCK` lines.
- **Flattening sign:** the sign that maximises mean coherence. The wrong sign leaves twice the topographic phase and decorrelates the multilook.
- **SKP sign:** the sign that maximises the interferogram uniformity $R = |\langle e^{j\arg\gamma}\rangle|$, since after calibration only $k_z z_{pc}$ remains.
"""),
("code", r'''
SEC = REF - 1 if REF > 0 else REF + 1
SIGN_DEM, SIGN_CAL, scores, margins = tomo.phase_signs(slc["HH"][:, blk], aux["dem_phases"][:, blk], aux["calibration_phases"][:, blk],
                                                       REF, SEC, window_r, window_x)
for (sd, sc), s in sorted(scores.items(), key=lambda kv: -kv[1]["coherence"]):
    print(f"SIGN_DEM {sd:+d} SIGN_CAL {sc:+d}: mean|coh| {s['coherence']:.3f}  uniformity {s['uniformity']:.3f}")
print(f"adopted SIGN_DEM {SIGN_DEM:+d}, SIGN_CAL {SIGN_CAL:+d}; margins {margins}")
if min(margins.values()) < 0.05:
    print("WARNING: small margin, the sign is poorly determined on this block")
'''),
("code", r'''
corr = tomo.phase_correction(aux["dem_phases"], aux["calibration_phases"], SIGN_DEM, SIGN_CAL)
for p in slc:
    slc[p] *= corr
del corr
print("SLCs phase-corrected (flattening + SKP)")
'''),
("md", r"""
## 5. Multibaseline coherence
For each channel, the $N\times N$ coherence matrix
$$\hat\Gamma_{mn} = \frac{\langle y_m y_n^*\rangle}{\sqrt{\langle|y_m|^2\rangle\langle|y_n|^2\rangle}}$$
is estimated over the `WINDOW_M` boxcar and decimated by half the window. Cells with less than 90 % valid samples are NaN. The result is cached, so re-running is fast.
"""),
("code", r'''
tag = f"az{CROP[0]}-{CROP[1]}_rg{CROP[2]}-{CROP[3]}_w{int(WINDOW_M)}_s{SIGN_DEM:+d}{SIGN_CAL:+d}"
coherence, intensity, coh_files = {}, {}, {}
for ch, pols in CHANNELS.items():
    cache = OUT / f"coherence_{ch}_{tag}.npz"
    coh_files[ch] = str(cache)
    if cache.exists():
        z = np.load(cache)
        coherence[ch], intensity[ch], ds_r, ds_x = z["coherence"], z["intensity"], int(z["ds_r"]), int(z["ds_x"])
    else:
        t0 = time.time()
        channel = slc[pols[0]] if len(pols) == 1 else sum(slc[p] for p in pols) / len(pols)
        coherence[ch], intensity[ch], ds_r, ds_x = tomo.coherence_matrix(channel, window_r, window_x)
        del channel
        np.savez(cache, coherence=coherence[ch], intensity=intensity[ch], ds_r=ds_r, ds_x=ds_x)
        print(f"{ch}: computed in {time.time() - t0:.0f} s")
    off = ~np.eye(Nb, dtype=bool)
    print(f"{ch}: coherence grid {coherence[ch].shape[:2]}, mean |gamma| off-diagonal {np.nanmean(np.abs(coherence[ch])[:, :, off]):.2f}")
del slc
Nr_c, Nx_c = coherence["X"].shape[:2]
sub_ds = lambda a: a[::ds_r, ::ds_x][:Nr_c, :Nx_c]
r_ds, x_ds = r_ax[::ds_r][:Nr_c], x_ax[::ds_x][:Nx_c]
kz_raw = sub_ds(aux["insar_wavenumbers"])
lat_ds, lon_ds, inc_ds, dem_ds = (sub_ds(aux[k]) for k in ("latitude", "longitude", "incidence_angle", "dem"))
'''),
("md", r"""
Mean |coherence| matrices: coherence falls with the $k_z$ separation (volume decorrelation) and, less, with time. The right panel plots every pair against $|\Delta k_z|$: the decay rate reflects the vertical extent of the scatterers in each channel.
"""),
("code", r'''
fig, axes = plt.subplots(1, len(CHANNELS) + 1, figsize=(4 * (len(CHANNELS) + 1), 3.6), constrained_layout=True)
labels = [f"C{int(a):02d}" for a in acq]
for ax, ch in zip(axes, CHANNELS):
    im = ax.imshow(np.nanmean(np.abs(coherence[ch]), axis=(0, 1)), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(Nb), labels, rotation=90, fontsize=7); ax.set_yticks(range(Nb), labels, fontsize=7); ax.set_title(f"{ch}: mean |γ|", fontsize=9)
fig.colorbar(im, ax=axes[:-1], shrink=0.8)
kzm = np.nanmean(kz_raw, axis=(0, 1))
for ch in CHANNELS:
    m = np.nanmean(np.abs(coherence[ch]), axis=(0, 1))
    pairs = [(i, j) for i in range(Nb) for j in range(i + 1, Nb)]
    axes[-1].plot([abs(kzm[i] - kzm[j]) for i, j in pairs], [m[i, j] for i, j in pairs], "o", label=ch, alpha=0.7)
axes[-1].set_xlabel(r"$|\Delta k_z|$ [rad/m]"); axes[-1].set_ylabel("mean |γ|"); axes[-1].legend(); axes[-1].grid(alpha=0.3)
plt.show()
'''),
("md", r"""
## 6. Tomograms and the $k_z$ sign
- Beamformer: $P_{BF}(z) = \mathbf a^H \hat{\mathbf\Gamma}\,\mathbf a$
- Capon: $P_{CP}(z) = 1/(\mathbf a^H\hat{\mathbf\Gamma}^{-1}\mathbf a)$

Here $\{\mathbf a\}_i = e^{-jk_z^i z}$. Using normalised coherence (not covariance) equalises the acquisitions. After the SKP calibration the ground is at $z = 0$ and the forest lies above it. If the mean profiles put more power below the ground than above, the $k_z$ sign is flipped.
"""),
("code", r'''
t0 = time.time()
KZ_SIGN = +1
for attempt in range(2):
    kz = KZ_SIGN * kz_raw
    tomos = {ch: tomo.tomograms(coherence[ch], kz, Z_VECTOR, D_LOAD) for ch in CHANNELS}
    profiles = {ch: np.nanmean(tomos[ch]["Beamformer"], axis=(0, 1)) for ch in CHANNELS}
    ratios, ok = tomo.kz_sign_check(profiles, Z_VECTOR)
    print(f"KZ_SIGN {KZ_SIGN:+d}: beamformer power above / below ground {ratios}")
    if ok:
        break
    KZ_SIGN = -KZ_SIGN
print(f"adopted KZ_SIGN {KZ_SIGN:+d} ({time.time() - t0:.0f} s)")
summary = tomo.kz_summary(kz)
print(f"vertical resolution {summary['resolution_m']:.1f} m, height of ambiguity {summary['ambiguity_m']:.1f} m")
'''),
("md", "### Mean vertical profiles\nMedian and interquartile range over all valid cells, each profile normalised to its maximum."),
("code", r'''
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, constrained_layout=True)
for ax, method in zip(axes, ("Beamformer", "Capon")):
    for ch in CHANNELS:
        t = tomos[ch][method].reshape(-1, Z_VECTOR.size)
        t = t / np.nanmax(t, axis=1, keepdims=True)
        q = np.nanpercentile(t, [25, 50, 75], axis=0)
        line, = ax.plot(q[1], Z_VECTOR, label=ch)
        ax.fill_betweenx(Z_VECTOR, q[0], q[2], color=line.get_color(), alpha=0.15)
    ax.axhline(0, color="k", lw=0.8); ax.set_title(f"{method} profile (normalised)"); ax.set_xlabel("normalised power"); ax.grid(alpha=0.3)
axes[0].set_ylabel("height above ground [m]"); axes[1].legend()
plt.show()
'''),
("md", "### Vertical transects\nCapon tomograms along azimuth at mid range (top three panels) and along range at mid azimuth (bottom panel). Each column is normalised to its maximum. The white dashed line is the SKP ground ($z = 0$)."),
("code", r'''
R_SLICE, X_SLICE = Nr_c // 2, Nx_c // 2
dz = Z_VECTOR[1] - Z_VECTOR[0]
fig, axes = plt.subplots(len(CHANNELS) + 1, 1, figsize=(12, 2.3 * (len(CHANNELS) + 1)), constrained_layout=True)
for ax, ch in zip(axes, CHANNELS):
    t = tomos[ch]["Capon"][R_SLICE].T
    im = ax.imshow(t / np.nanmax(t, axis=0, keepdims=True), aspect="auto", origin="lower", cmap="jet", vmin=0, vmax=1,
                   extent=[x_ds[0] / 1e3, x_ds[-1] / 1e3, Z_VECTOR[0] - dz / 2, Z_VECTOR[-1] + dz / 2], interpolation="none")
    ax.axhline(0, color="w", ls="--", lw=0.8); ax.set_ylabel("height [m]")
    ax.set_title(f"Capon {ch}, azimuth transect at slant range {r_ds[R_SLICE] / 1e3:.1f} km", fontsize=8, loc="left")
axes[len(CHANNELS) - 1].set_xlabel("azimuth [km]")
t = tomos["X"]["Capon"][:, X_SLICE].T
axes[-1].imshow(t / np.nanmax(t, axis=0, keepdims=True), aspect="auto", origin="lower", cmap="jet", vmin=0, vmax=1,
                extent=[r_ds[0] / 1e3, r_ds[-1] / 1e3, Z_VECTOR[0] - dz / 2, Z_VECTOR[-1] + dz / 2], interpolation="none")
axes[-1].axhline(0, color="w", ls="--", lw=0.8); axes[-1].set_ylabel("height [m]"); axes[-1].set_xlabel("slant range [km]")
axes[-1].set_title(f"Capon X, range transect at azimuth {x_ds[X_SLICE] / 1e3:.1f} km", fontsize=8, loc="left")
fig.colorbar(im, ax=axes, shrink=0.5, label="column-normalised power")
plt.show()
'''),
("md", r"""
### Horizontal slices
Fraction of the cross-pol Capon power at selected heights, each pixel normalised by its total power over 0–60 m. Ground-dominated areas (bare soil, water, savanna) are bright at 0 m, tall forest at 20–30 m.
"""),
("code", r'''
SLICES = [0, 10, 20, 30, 40]
t = tomos["X"]["Capon"]
total = np.nansum(t[..., (Z_VECTOR >= 0) & (Z_VECTOR <= 60)], axis=-1)
fig, axes = plt.subplots(len(SLICES), 1, figsize=(12, 1.6 * len(SLICES)), sharex=True, constrained_layout=True)
for ax, h in zip(axes, SLICES):
    k = int(np.argmin(np.abs(Z_VECTOR - h)))
    im = ax.imshow(t[..., k] / total, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=np.nanpercentile(t[..., k] / total, 98),
                   extent=[x_ds[0] / 1e3, x_ds[-1] / 1e3, r_ds[0] / 1e3, r_ds[-1] / 1e3], interpolation="none")
    ax.set_title(f"z = {Z_VECTOR[k]:.0f} m", fontsize=8, loc="left"); ax.set_ylabel("range [km]")
    fig.colorbar(im, ax=ax, pad=0.01, aspect=8)
axes[-1].set_xlabel("azimuth [km]")
plt.show()
'''),
("md", r"""
## 7. Saving
- `tomography/tomograms_<tag>.nc`: Capon and beamformer tomograms per channel; multilooked intensity; $k_z$, incidence angle, DEM, latitude and longitude on the coherence grid.
- `tomography/coherence_<ch>_<tag>.npz`: coherence matrices (used by notebook 3 for the phase-centre and RVoG heights).
- The manifest gains `phase` (signs, reference) and `tomography` (file paths, window), so notebooks 3 and 4 use exactly the same conventions.
"""),
("code", r'''
grid = ("range", "azimuth")
tomo_ds = xr.Dataset(
    {**{f"capon_{ch}": (grid + ("height",), tomos[ch]["Capon"]) for ch in CHANNELS},
     **{f"beamformer_{ch}": (grid + ("height",), tomos[ch]["Beamformer"]) for ch in CHANNELS},
     **{f"intensity_{ch}": (grid, intensity[ch].astype(np.float32)[:Nr_c, :Nx_c]) for ch in CHANNELS},
     "kz": (grid + ("acquisition",), kz.astype(np.float32)),
     "incidence_angle": (grid, inc_ds), "dem": (grid, dem_ds), "latitude": (grid, lat_ds), "longitude": (grid, lon_ds)},
    coords={"range": r_ds, "azimuth": x_ds, "height": Z_VECTOR, "acquisition": acq},
    attrs={"stack": cfg["stack_path"], "crop": str(CROP), "window_m": WINDOW_M, "window_samples": f"{window_r} x {window_x}",
           "decimation": f"{ds_r} x {ds_x}", "sign_dem": SIGN_DEM, "sign_cal": SIGN_CAL, "kz_sign": KZ_SIGN,
           "reference_index": REF, "height_reference": "ground (after SKP calibration)",
           "kz_definition": "steering a_i(z) = exp(-j kz_i z), kz already multiplied by kz_sign"})
tomo_ds["range"].attrs["units"] = tomo_ds["azimuth"].attrs["units"] = tomo_ds["height"].attrs["units"] = "m"
tomo_path = OUT / f"tomograms_{tag}.nc"
st.save_netcdf(tomo_ds, tomo_path)

cfg.update(crop=CROP,
           phase={"sign_dem": SIGN_DEM, "sign_cal": SIGN_CAL, "kz_sign": KZ_SIGN, "reference_index": REF, "sign_margins": margins},
           tomography={"tomograms": str(tomo_path), "coherence": coh_files, "window_m": WINDOW_M, "tag": tag,
                       "resolution_m": float(summary["resolution_m"]), "ambiguity_m": float(summary["ambiguity_m"])})
st.write_manifest(MANIFEST, **cfg)
print(f"saved {tomo_path} ({tomo_path.stat().st_size / 1e9:.2f} GB) and updated {MANIFEST}")
'''),
]

# =====================================================================================================
NB3 = [
("md", r"""
# 3 · Guainía: forest height from the tomograms

""" + banner(3) + r"""

Input: tomograms and coherence matrices from notebook 2, through the manifest. Heights are relative to the SKP-calibrated ground ($z=0$).

| estimator | ground | top | height |
|---|---|---|---|
| energy envelope | $z=0$ | height below which 95 % of the profile energy lies | $H_{95}$ |
| percentile envelope | 5 % energy height | 95 % energy height | $H_{95} - H_5$ |
| min–max edge | profile peak | highest height above the peak where the min–max-scaled profile ≥ `EDGE_LEVEL` | $z_{edge} - z_{peak}$ |
| canopy top | $z=0$ | highest height contiguous with the peak within `TOP_DB` of the maximum | $h_{top}$ |
| phase centre | $z=0$ | – | $z_{pc}$: median over baselines of $-\arg\gamma/\Delta k_z$ (a centroid, below the top) |
| RVoG | $z=0$ | volume height of the Random-Volume-over-Ground model fitted to all pairs | $h_v$ |

With a vertical resolution of about 20 m (notebook 1), the Capon estimators are reliable for forests taller than roughly half the resolution. Below that, ground and canopy are not separated.

**Reference.** `Forest_Height_Guainia.tif` is a 30 m canopy-height map (EPSG:4326, uint8 metres) using the GLAD coding: 0 = no trees, 101 = water, 102 = snow/ice, 103 = no data. It is pooled onto the coherence grid over about the multilook footprint and used to assess every estimator. The best estimator also gets a bias correction.

**Steps:** load → reference on the radar grid → quality mask → heights → maps and distributions → example profiles → assessment against the reference → bias correction → geographic maps and GeoTIFF → optional biomass → save.

Maps in radar geometry are drawn at true ground scale. x is ground range and y is azimuth, with the first line at the top. For this descending pass that is roughly north-up and east-right.
"""),
("md", r"""
## 1. Configuration
`REFERENCE_POOL` is the pooling size in coherence cells. The coherence grid is about 95 m, so 3 × 3 cells ≈ 280 m, slightly more than the 200 m multilook window plus geolocation error. The assessment is restricted to **forest** cells: no water in the footprint and less than `FOREST_MAX_NONFOREST` of its 30 m pixels without trees.
"""),
("code", SETUP + r'''
import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="(All-NaN slice|Mean of empty slice|invalid value|divide by zero|Degrees of freedom)", category=RuntimeWarning)

from biomass_tomo import stack as st, tomo, reference as ref, display as dp

MANIFEST = SITE_DIR / "Guainia_outputs" / "run_manifest.json"
HEIGHT_CHANNEL = "X"           # channel used for the height maps: "X" (cross-pol), "HH" or "VV"
METHOD = "Capon"               # tomogram used by the profile estimators: "Capon" or "Beamformer"
COH_MIN = 0.1                  # quality mask: mean off-diagonal |gamma|
CONTRAST_MIN_DB = 1.0          # quality mask: profile max / mean [dB]
EDGE_LEVEL = 0.5               # min-max edge level
TOP_DB = -1.0                  # canopy-top threshold below the peak [dB]
H_MAX = 60.0                   # highest plausible canopy [m]
RUN_RVOG = True
RVOG_STEP = 2                  # RVoG on every RVOG_STEP-th cell in both directions (it is the slowest step)
WRITE_GEOTIFF = True

# reference forest height (None = no reference)
REFERENCE_HEIGHT = SITE_DIR / "Forest_Height_Guainia.tif"
REFERENCE_BAND = 1
REFERENCE_INVALID = (101, 102, 103)  # GLAD codes: water, snow/ice, no data
REFERENCE_NONFOREST = 0              # GLAD code: no trees
REFERENCE_POOL = (3, 3)              # pooling in coherence cells (≈ multilook footprint)
FOREST_MAX_NONFOREST = 0.2           # assessment mask: max fraction of tree-less 30 m pixels in the footprint
CORRECT_ESTIMATOR = None             # estimator to bias-correct; None = lowest RMSE against the reference
CORRECTION = "bias"                  # "bias" (subtract the mean bias) or "linear" (regress reference on estimate)
SPLIT_BLOCK_KM = 10.0                # spatial blocks (checkerboard) for the train / test split of the correction

cfg = st.read_manifest(MANIFEST)
if "tomography" not in cfg:
    raise RuntimeError("run notebook 2 first")
OUT = pathlib.Path(cfg["out_dir"]) / "forest_height"
OUT.mkdir(parents=True, exist_ok=True)
ref.fix_proj_data()
'''),
("md", "## 2. Data\nTomogram of `HEIGHT_CHANNEL`, $k_z$, coherence matrices and geolocation on the coherence grid. Maps are shown as (azimuth, range), with axes in ground kilometres."),
("code", r'''
T = xr.open_dataset(cfg["tomography"]["tomograms"])
Z = T.height.values
vol = T[f"{METHOD.lower()}_{HEIGHT_CHANNEL}"].values
kz = T.kz.values
coh = np.load(cfg["tomography"]["coherence"][HEIGHT_CHANNEL])["coherence"]
REF = cfg["phase"]["reference_index"]
lat, lon, inc = T.latitude.values, T.longitude.values, T.incidence_angle.values
az_km = (T.azimuth.values - T.azimuth.values[0]) / 1e3
gr_km = dp.ground_range_km(T.range.values, inc)
mapview = lambda ax, v, **kw: dp.show(ax, np.asarray(v).T, az_km, gr_km, **kw)     # (range, azimuth) -> map
print(f"{METHOD} {HEIGHT_CHANNEL}: {vol.shape} (range, azimuth, height); window {T.attrs['window_m']} m; "
      f"resolution {cfg['tomography']['resolution_m']:.1f} m; scene {gr_km[-1]:.0f} km x {az_km[-1]:.0f} km")
'''),
("md", r"""
## 3. Reference forest height on the radar grid
Every valid 30 m pixel is assigned to its nearest coherence cell (by the LUT geolocation). Sums are pooled over `REFERENCE_POOL` cells, which gives per cell:
- `ref_mean`: mean height of the 30 m pixels, with trees counted at their height and tree-less pixels as 0;
- `ref_max`: the highest 30 m pixel;
- `ref_std`: the spread of the 30 m pixels;
- the fractions of water/invalid and tree-less pixels.

The panels show the reference as delivered (lon/lat), on the radar grid, its within-footprint spread, and the forest mask used for the assessment.
"""),
("code", r'''
refmap, forest = {}, np.ones(lat.shape, bool)
if REFERENCE_HEIGHT is not None and pathlib.Path(REFERENCE_HEIGHT).exists():
    import rasterio
    t0 = time.time()
    refmap = ref.map_to_grid(REFERENCE_HEIGHT, lat, lon, band=REFERENCE_BAND, invalid=REFERENCE_INVALID,
                             nonforest=REFERENCE_NONFOREST, pool=REFERENCE_POOL)
    forest = (refmap["ref_invalid_frac"] == 0) & (refmap["ref_nonforest_frac"] <= FOREST_MAX_NONFOREST) & np.isfinite(refmap["ref_mean"])
    print(f"reference pooled in {time.time() - t0:.0f} s; median {np.nanmedian(refmap['ref_count']):.0f} pixels per footprint; "
          f"forest cells {forest.mean() * 100:.1f} %")
    print(f"ref_mean over forest: median {np.nanmedian(refmap['ref_mean'][forest]):.1f} m, "
          f"5-95 %: {np.nanpercentile(refmap['ref_mean'][forest], 5):.1f}-{np.nanpercentile(refmap['ref_mean'][forest], 95):.1f} m")

    with rasterio.open(REFERENCE_HEIGHT) as src:
        dec = 4
        native = src.read(REFERENCE_BAND, out_shape=(src.height // dec, src.width // dec)).astype(np.float32)
        b = src.bounds
    native[np.isin(native, REFERENCE_INVALID)] = np.nan
    hlo, hhi = dp.limits([native, refmap["ref_mean"][forest]], (1, 99))
    fig, axes = plt.subplots(1, 4, figsize=(17, 7.5), constrained_layout=True)
    im = axes[0].imshow(native, extent=[b.left, b.right, b.bottom, b.top], cmap="viridis", vmin=hlo, vmax=hhi, interpolation="none")
    axes[0].set_aspect(1 / np.cos(np.deg2rad(np.nanmean(lat)))); axes[0].set_xlabel("longitude"); axes[0].set_ylabel("latitude")
    axes[0].set_title("reference (as delivered, lon/lat)", fontsize=9); fig.colorbar(im, ax=axes[0], shrink=0.6, label="m")
    im = mapview(axes[1], refmap["ref_mean"], cmap="viridis", vmin=hlo, vmax=hhi)
    axes[1].set_title("ref_mean on the radar grid", fontsize=9); axes[1].set_ylabel("azimuth [km]"); fig.colorbar(im, ax=axes[1], shrink=0.6, label="m")
    im = mapview(axes[2], refmap["ref_std"], cmap="magma", vmin=0, vmax=np.nanpercentile(refmap["ref_std"], 98))
    axes[2].set_title("ref_std within footprint", fontsize=9); fig.colorbar(im, ax=axes[2], shrink=0.6, label="m")
    mapview(axes[3], forest, cmap="gray", vmin=0, vmax=1); axes[3].set_title("forest mask (assessment)", fontsize=9)
    plt.show()
else:
    print("no reference height map: the assessment sections are skipped")
'''),
("md", r"""
## 4. Quality mask
A cell has height information only if the stack is coherent (mean off-diagonal $|\gamma| \geq$ `COH_MIN`) and the profile has a peak (Capon max/mean ≥ `CONTRAST_MIN_DB`). Masked cells are typically water, frame edges (no-data) or strongly temporally decorrelated areas.
"""),
("code", r'''
nb = coh.shape[2]
mean_coh = np.nanmean(np.abs(coh)[:, :, ~np.eye(nb, dtype=bool)], axis=2)
capon_x = T[f"capon_{HEIGHT_CHANNEL}"].values
contrast = 10 * np.log10(np.nanmax(capon_x, axis=2) / np.nanmean(capon_x, axis=2))
reliable = tomo.quality_mask(coh, capon_x, COH_MIN, CONTRAST_MIN_DB)
print(f"reliable cells: {reliable.mean() * 100:.1f} %")
fig, axes = plt.subplots(1, 3, figsize=(13, 7.5), sharey=True, constrained_layout=True)
for ax, (v, title, kw) in zip(axes, [(mean_coh, "mean off-diagonal |γ|", dict(vmin=0, vmax=dp.limits(mean_coh)[1])),
                                     (contrast, "Capon contrast [dB]", dict(vmin=0, vmax=dp.limits(contrast)[1])),
                                     (reliable, "reliable", dict(vmin=0, vmax=1, cmap="gray"))]):
    im = mapview(ax, v, **{"cmap": "viridis", **kw})
    ax.set_title(title, fontsize=9); fig.colorbar(im, ax=ax, shrink=0.6)
axes[0].set_ylabel("azimuth [km]")
plt.show()
'''),
("md", "## 5. Height estimators"),
("code", r'''
t0 = time.time()
z_low, z_high = tomo.percentile_height(vol, Z, [5, 95], z_range=(Z[0], H_MAX + 10))
z_peak, z_edge = tomo.minmax_edge(vol, Z, level=EDGE_LEVEL, z_range=(Z[0], H_MAX + 10), z_top_max=H_MAX)
heights = {
    "H_envelope": z_high,
    "H_percentile": z_high - z_low,
    "H_minmax": z_edge - z_peak,
    "h_canopy_top": tomo.canopy_top(vol, Z, TOP_DB),
    "z_phase_centre": tomo.phase_centre(coh, kz, REF),
}
if RUN_RVOG:
    sl = (slice(None, None, RVOG_STEP), slice(None, None, RVOG_STEP))
    rv = tomo.rvog_inversion(coh[sl], kz[sl], inc[sl], valid=reliable[sl], hv_grid=np.arange(0, H_MAX + 1, 1.0))
    h_rv = np.full(reliable.shape, np.nan, np.float32)
    h_rv[sl] = np.where(rv["at_grid_limit"], np.nan, rv["h_v"])
    heights["h_rvog"] = h_rv
heights = {k: np.where(reliable & (v >= -5) & (v <= H_MAX), v, np.nan).astype(np.float32) for k, v in heights.items()}
print(f"computed in {time.time() - t0:.0f} s")
for k, v in heights.items():
    print(f"{k:16s} median {np.nanmedian(v):5.1f} m   5-95 %: {np.nanpercentile(v, 5):5.1f} - {np.nanpercentile(v, 95):5.1f} m   "
          f"valid {np.isfinite(v).mean() * 100:5.1f} %")
'''),
("md", "## 6. Maps and distributions\nAll estimators and the reference on one colour scale (2–98 % of all values). The histograms are over forest cells, so they compare directly with the reference."),
("code", r'''
panels = dict(heights, **({"reference (ref_mean)": refmap["ref_mean"]} if refmap else {}))
vmin, vmax = dp.limits(list(panels.values()), (2, 98))
fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 7.2), sharey=True, constrained_layout=True)
for ax, (k, v) in zip(axes, panels.items()):
    im = mapview(ax, v, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(k, fontsize=9)
axes[0].set_ylabel("azimuth [km]")
fig.colorbar(im, ax=axes, label="height [m]", shrink=0.6)
plt.show()

fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
bins = np.arange(-5, H_MAX + 1, 1)
for k, v in heights.items():
    ax.hist(v[forest & np.isfinite(v)], bins=bins, histtype="step", density=True, lw=1.4, label=k)
if refmap:
    ax.hist(refmap["ref_mean"][forest], bins=bins, color="k", alpha=0.25, density=True, label="reference ref_mean")
ax.set_xlabel("height [m]"); ax.set_ylabel("density"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.show()
'''),
("md", "## 7. Example profiles\nRandom reliable forest cells: the Capon profile of `HEIGHT_CHANNEL` (dB, relative to its peak), with the estimated heights and the reference (`ref_mean` green, `ref_max` dotted green)."),
("code", r'''
rng = np.random.default_rng(0)
cand = np.argwhere(reliable & forest & np.isfinite(heights["h_canopy_top"]))
pick = cand[rng.choice(len(cand), size=min(8, len(cand)), replace=False)]
fig, axes = plt.subplots(1, len(pick), figsize=(2.1 * len(pick), 4.4), sharey=True, constrained_layout=True)
marks = {"h_canopy_top": ("r", "-"), "H_envelope": ("orange", "--"), "z_phase_centre": ("m", ":"), "h_rvog": ("c", "-.")}
for ax, (i, j) in zip(np.atleast_1d(axes), pick):
    p = vol[i, j]
    ax.plot(10 * np.log10(p / np.nanmax(p)), Z, "k")
    for k, (c, ls) in marks.items():
        if k in heights and np.isfinite(heights[k][i, j]):
            ax.axhline(heights[k][i, j], color=c, ls=ls, lw=1.2, label=k)
    if refmap:
        ax.axhline(refmap["ref_mean"][i, j], color="g", lw=2, alpha=0.7, label="reference mean")
        ax.axhline(refmap["ref_max"][i, j], color="g", ls=":", lw=1.5, label="reference max")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlim(-15, 0.5); ax.set_title(f"({i}, {j})", fontsize=8); ax.set_xlabel("dB")
np.atleast_1d(axes)[0].set_ylabel("height [m]"); np.atleast_1d(axes)[-1].legend(fontsize=6, loc="upper right")
plt.show()
'''),
("md", r"""
## 8. Assessment against the reference
Every estimator is compared with `ref_mean` over reliable forest cells: number of cells, correlation $r$, bias (estimate − reference), RMSE, MAE and the slope of estimate on reference. Note that over closed forest the reference spans only a few metres (5–95 %), so $r$ is limited by the reference itself. Bias and RMSE are the more informative numbers.

The figure shows
- **top row:** the joint distribution per estimator, with the 1:1 line (red dashed) and the binned median ± IQR (orange);
- **bottom row:** the residual against incidence angle. A trend here points to geometry ($k_z$, resolution) rather than forest.
"""),
("code", r'''
if refmap:
    R = refmap["ref_mean"]
    stats = {}
    for k, v in heights.items():
        s = ref.compare(R[forest], v[forest])
        if s:
            stats[k] = s
    table = pd.DataFrame({k: {m: s[m] for m in ("n", "r", "bias", "rmse", "mae", "slope")} for k, s in stats.items()}).T
    table = table.astype({"n": int}).sort_values("rmse")
    display(table.round(2))

    lo, hi = dp.limits(R[forest], (0.5, 99.5))
    rbins = np.arange(np.floor(lo), np.ceil(hi) + 1, 1.0)
    keys = list(table.index)
    fig, axes = plt.subplots(2, len(keys), figsize=(3.2 * len(keys), 6.6), constrained_layout=True)
    for col, k in enumerate(keys):
        ax = axes[0, col]
        x, y = R[forest], heights[k][forest]
        ok = np.isfinite(x) & np.isfinite(y)
        ax.hist2d(x[ok], y[ok], bins=[rbins, np.arange(-5, H_MAX + 1, 1)], cmap="Greys", cmin=1)
        cen = 0.5 * (rbins[1:] + rbins[:-1])
        q = np.array([np.nanpercentile(y[ok][(x[ok] >= a) & (x[ok] < b)], [25, 50, 75]) if ((x[ok] >= a) & (x[ok] < b)).sum() > 20
                      else [np.nan] * 3 for a, b in zip(rbins[:-1], rbins[1:])])
        ax.errorbar(cen, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], fmt="o", color="tab:orange", ms=3, lw=1)
        ax.plot([0, H_MAX], [0, H_MAX], "r--", lw=0.8)
        ax.set_xlim(rbins[0], rbins[-1]); ax.set_ylim(-5, H_MAX)
        ax.set_xlabel("reference ref_mean [m]"); ax.set_ylabel(f"{k} [m]")
        s = stats[k]
        ax.set_title(f"r={s['r']:.2f} bias={s['bias']:+.1f} RMSE={s['rmse']:.1f}", fontsize=8)
        ax = axes[1, col]
        res = (heights[k] - R)[forest]
        ib = np.linspace(np.nanmin(inc), np.nanmax(inc), 16)
        cen_i = 0.5 * (ib[1:] + ib[:-1])
        q = np.array([np.nanpercentile(res[(inc[forest] >= a) & (inc[forest] < b)], [25, 50, 75]) for a, b in zip(ib[:-1], ib[1:])])
        ax.fill_between(cen_i, q[:, 0], q[:, 2], alpha=0.25); ax.plot(cen_i, q[:, 1], "o-", ms=3)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("incidence [deg]"); ax.set_ylabel("estimate − reference [m]"); ax.grid(alpha=0.3)
    plt.show()
else:
    table = None
    print("no reference: skipped")
'''),
("md", "### Difference map\nThe best estimator (lowest RMSE), the reference and their difference. A spatially coherent difference points to forest structure or terrain effects; salt-and-pepper differences are estimation noise."),
("code", r'''
if refmap:
    BEST = CORRECT_ESTIMATOR or table.index[0]
    d = np.where(forest, heights[BEST] - refmap["ref_mean"], np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(13, 7.5), sharey=True, constrained_layout=True)
    for ax, (v, title) in zip(axes[:2], ((heights[BEST], BEST), (refmap["ref_mean"], "reference ref_mean"))):
        im = mapview(ax, v, cmap="viridis", vmin=vmin, vmax=vmax); ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=axes[:2], shrink=0.6, label="height [m]")
    lim = np.nanpercentile(np.abs(d - np.nanmedian(d)), 95)
    im = mapview(axes[2], d, cmap="RdBu_r", vmin=np.nanmedian(d) - lim, vmax=np.nanmedian(d) + lim)
    axes[2].set_title(f"{BEST} − reference (forest)", fontsize=9); fig.colorbar(im, ax=axes[2], shrink=0.6, label="m")
    axes[0].set_ylabel("azimuth [km]")
    plt.show()
'''),
("md", r"""
## 9. Bias correction
The best estimator still carries a systematic offset relative to the reference. With a 20 m vertical resolution, the −1 dB canopy-top threshold sits below the true top. The correction is fitted on half of the scene and tested on the other half. The halves are a checkerboard of `SPLIT_BLOCK_KM` blocks, so that neighbouring, correlated cells do not end up on both sides of the split.
- `CORRECTION = "bias"`: $h_{corr} = h - \overline{(h - h_{ref})}$. This keeps the spatial pattern of the radar estimate and is the default.
- `CORRECTION = "linear"`: $h_{corr} = a + b\,h$, the regression of the reference on the estimate. With a low $r$, this shrinks the map towards the reference mean ($b \ll 1$). It lowers the RMSE, but the variation then comes mostly from the reference.
"""),
("code", r'''
corrected = {}
if refmap:
    h, R = heights[BEST], refmap["ref_mean"]
    AZ, GR = np.meshgrid(az_km, gr_km)                                # (range, azimuth) like the maps
    train = ((np.floor(AZ / SPLIT_BLOCK_KM) + np.floor(GR / SPLIT_BLOCK_KM)) % 2 == 0)
    ok = forest & np.isfinite(h) & np.isfinite(R)
    tr, te = ok & train, ok & ~train
    if CORRECTION == "linear":
        b, a = np.polyfit(h[tr] - h[tr].mean(), R[tr], 1); a -= b * h[tr].mean()
        h_corr = a + b * h
        params = {"mode": "linear", "a": float(a), "b": float(b)}
    else:
        off = float(np.mean(h[tr] - R[tr]))
        h_corr = h - off
        params = {"mode": "bias", "offset_m": off}
    corrected = {f"{BEST}_corrected": h_corr.astype(np.float32)}
    rows = {"raw, test half": ref.compare(R[te], h[te]), "corrected, test half": ref.compare(R[te], h_corr[te])}
    display(pd.DataFrame({k: {m: s[m] for m in ("n", "r", "bias", "rmse", "mae")} for k, s in rows.items()}).T.round(2))
    print("correction", params, f"| train cells {tr.sum()}, test cells {te.sum()}")
    params.update(estimator=BEST, test_rmse_m=rows["corrected, test half"]["rmse"], test_bias_m=rows["corrected, test half"]["bias"])
else:
    params = None
'''),
("md", r"""
## 10. Geographic maps and GeoTIFF
The height maps are placed on a regular lon/lat grid (EPSG:4326) by nearest neighbour from the LUT geolocation, at the coherence-grid spacing (≈ 95 m). Grid cells far from any radar cell are no-data. The GeoTIFF has one band per estimator, plus the corrected estimate and the pooled reference, with band descriptions.
"""),
("code", r'''
step = max(1, int(np.ceil(np.sqrt(lat.size / 400_000))))
warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh", category=UserWarning)
geo = {f"{BEST if refmap else 'h_canopy_top'}": heights[BEST if refmap else "h_canopy_top"], **corrected,
       **({"reference ref_mean": refmap["ref_mean"]} if refmap else {})}
fig, axes = plt.subplots(1, len(geo), figsize=(5.2 * len(geo), 8), sharey=True, constrained_layout=True)
for ax, (k, v) in zip(np.atleast_1d(axes), geo.items()):
    pc = ax.pcolormesh(lon[::step, ::step], lat[::step, ::step], v[::step, ::step], cmap="viridis", vmin=vmin, vmax=vmax, shading="nearest")
    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(lat)))); ax.set_xlabel("longitude [deg]"); ax.set_title(k, fontsize=10)
np.atleast_1d(axes)[0].set_ylabel("latitude [deg]")
fig.colorbar(pc, ax=axes, label="height [m]", shrink=0.6)
plt.show()

if WRITE_GEOTIFF:
    tif = OUT / f"forest_height_{METHOD}_{HEIGHT_CHANNEL}_{cfg['tomography']['tag']}.tif"
    bands = {**heights, **corrected, **({"reference_ref_mean": refmap["ref_mean"]} if refmap else {})}
    _, res = ref.to_geotiff(bands, lat, lon, tif)
    print(f"GeoTIFF {tif} ({len(bands)} bands: {', '.join(bands)}; {res:.5f} deg ≈ {res * 111e3:.0f} m)")
'''),
("md", "## 11. Biomass (optional)\nIf the manifest names an AGBD map (`reference.agbd`, set in notebook 1), it is sampled at the cells and related to height."),
("code", r'''
agbd = None
if cfg["reference"].get("agbd"):
    path, band = cfg["reference"]["agbd"]
    agbd = ref.sample_raster(path, lat, lon, band=band); agbd[agbd <= 0] = np.nan
    hb = heights[BEST if refmap else "h_canopy_top"]
    okm = np.isfinite(agbd) & np.isfinite(hb)
    if okm.sum() > 3:
        fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
        ax.hist2d(hb[okm], agbd[okm], bins=50, cmap="Greys", cmin=1)
        ax.set_xlabel("height [m]"); ax.set_ylabel("AGBD [Mg/ha]")
        ax.set_title(f"r = {np.corrcoef(hb[okm], agbd[okm])[0, 1]:.2f}, n = {okm.sum()}", fontsize=9)
        plt.show()
else:
    print("no AGBD map in the manifest: skipped")
'''),
("md", "## 12. Saving\nHeights, the corrected estimate, the pooled reference layers, the quality and forest masks and the geolocation go to netCDF on the coherence grid. The manifest gains `forest_height`, with the assessment table and the correction."),
("code", r'''
grid = ("range", "azimuth")
H = xr.Dataset({**{k: (grid, v) for k, v in heights.items()}, **{k: (grid, v) for k, v in corrected.items()},
                **{k: (grid, v) for k, v in refmap.items()},
                "forest": (grid, forest.astype(np.int8)),
                "z_peak": (grid, z_peak), "mean_coherence": (grid, mean_coh.astype(np.float32)),
                "capon_contrast_db": (grid, contrast.astype(np.float32)), "reliable": (grid, reliable.astype(np.int8)),
                **({"agbd": (grid, agbd)} if agbd is not None else {}),
                "latitude": (grid, lat), "longitude": (grid, lon), "incidence_angle": (grid, inc)},
               coords={"range": T.range.values, "azimuth": T.azimuth.values},
               attrs={**T.attrs, "height_channel": HEIGHT_CHANNEL, "method": METHOD, "edge_level": EDGE_LEVEL,
                      "top_db": TOP_DB, "coh_min": COH_MIN, "contrast_min_db": CONTRAST_MIN_DB, "rvog_step": RVOG_STEP if RUN_RVOG else 0,
                      "reference": str(REFERENCE_HEIGHT) if refmap else "", "reference_pool_cells": str(REFERENCE_POOL),
                      "units": "m above the SKP-calibrated ground"})
path = OUT / f"forest_height_{METHOD}_{HEIGHT_CHANNEL}_{cfg['tomography']['tag']}.nc"
st.save_netcdf(H, path)
cfg["reference"]["height"] = [str(REFERENCE_HEIGHT), REFERENCE_BAND] if refmap else None
cfg["forest_height"] = {"heights": str(path), "geotiff": str(tif) if WRITE_GEOTIFF else None,
                        "channel": HEIGHT_CHANNEL, "method": METHOD,
                        "median_m": {k: float(np.nanmedian(v)) for k, v in heights.items()},
                        "assessment": None if table is None else table.round(3).to_dict(orient="index"),
                        "correction": params}
st.write_manifest(MANIFEST, **cfg)
print("saved", path)
'''),
]

# =====================================================================================================
NB4 = [
("md", r"""
# 4 · Guainía: polarimetric analysis

""" + banner(4) + r"""

Two complementary analyses of the quad-pol stack:

**Part A, single-acquisition polarimetry (2-D).** The reference acquisition is calibrated to σ⁰ with the `/radiometry` LUT, and the thermal noise from the `/denoising` LUT is removed from the coherency matrix $\mathbf T_3$. It is then decomposed with PolSARpro:
backscatter and noise level → Pauli RGB → Cloude–Pottier $H/A/\alpha$ and the $H$–$\alpha$ plane → Freeman–Durden → Wishart $H/A/\alpha$ classification → temporal stability over the 7 acquisitions.

**Part B, polarimetric tomography (3-D).** The four phase-corrected channels are focused **coherently** at each height with one common filter, so that $\mathbf S(z) = [S_{HH}, S_{HV}, S_{VH}, S_{VV}](z)$ is a valid scattering matrix:
$$S_{pq}(z) = \mathbf w^H(z)\,\mathbf y_{pq},\qquad \mathbf w = \mathbf a(z)/N\ \text{(beamforming)}\quad\text{or}\quad \frac{\mathbf R^{-1}\mathbf a}{\mathbf a^H\mathbf R^{-1}\mathbf a}\ \text{(Capon)}.$$
Every height slice is decomposed (Pauli, $H/A/\alpha$, Freeman–Durden). The phase conventions come from notebook 2 and the forest height from notebook 3.

PolSARpro layout throughout: `y` = azimuth, `x` = range. Maps are drawn at true ground scale. x is ground range and y is azimuth, with the first line at the top, which for this descending pass is roughly north-up. Colours are stretched between the 2nd and 98th percentiles in dB, so that the small contrasts inside closed forest stay visible.

**GeoTIFFs** (like notebook 3, section 10): the Part A Pauli RGB is exported as an 8-bit RGB GeoTIFF for GIS display, plus a 3-band dB GeoTIFF with the values. Part B exports a Pauli RGB GeoTIFF for each height in `GEOTIFF_HEIGHTS`.
"""),
("md", r"""
## 1. Configuration
Polarimetric processing runs at full resolution, so it uses a block of the analysis crop (`POL_CROP`, stack pixel indices). The default is 4000 azimuth lines (≈ 27 km) at the crop centre, over the full range.
`BOX` is the multilook window (azimuth × range samples). (21, 3) ≈ 140 m × 140 m on the ground, giving 63 looks.
"""),
("code", SETUP + r'''
import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from biomass_tomo import stack as st, polsar, poltomo, reference as ref, display as dp

MANIFEST = SITE_DIR / "Guainia_outputs" / "run_manifest.json"
POL_CROP = None                       # (az0, az1, rg0, rg1) in stack indices; None = 4000 central lines of the analysis crop,
                                      # range limited to samples with data (no-data frame edges bias the Wishart classes)
POL_LINES = 4000
BOX = (21, 3)                         # multilook (azimuth, range) samples
STEP = (BOX[0] // 3, 1)               # output decimation for Part A maps
CALIBRATION = "sigma0"                # "sigma0", "gamma0" or "beta0"
NOISE_REMOVAL = True
NOISE_TOL_DB = 1.5                    # noise-LUT check: tolerance around the across-stack median

# Part B
HEIGHTS = np.arange(-10, 50.1, 2.5)   # focusing heights above ground [m]
FOCUSING = "beamforming"              # "beamforming" or "capon"
CAPON_WINDOW_M, CAPON_LOAD = 150.0, 0.01
MAP_HEIGHTS = [0, 10, 20, 30, 40]
STRETCH = (2, 98)                     # percentile colour stretch (dB for powers)
GEOTIFF_HEIGHTS = [0, 20]             # Part B: heights exported as Pauli RGB GeoTIFFs ([] = none)

cfg = st.read_manifest(MANIFEST)
if "phase" not in cfg:
    raise RuntimeError("run notebook 2 first: the phase conventions come from it")
ph = cfg["phase"]
crop = cfg["crop"]
if POL_CROP is None:
    c = (crop[0] + crop[1]) // 2
    ds = st.open_stack(cfg["stack_path"])
    line = ds.slc_real.sel(polarisation="HH").isel(acquisition=ph["reference_index"], azimuth=slice(c - 50, c + 50),
                                                   range=slice(crop[2], crop[3])).values
    has = np.flatnonzero((line != 0).all(axis=0))
    POL_CROP = [max(crop[0], c - POL_LINES // 2), min(crop[1], c + POL_LINES // 2), crop[2] + int(has[0]), crop[2] + int(has[-1]) + 1]
OUT = pathlib.Path(cfg["out_dir"]) / "polarimetry"
OUT.mkdir(parents=True, exist_ok=True)
blk = f"az{POL_CROP[0]}-{POL_CROP[1]}_rg{POL_CROP[2]}-{POL_CROP[3]}"
print("polarimetry block (az0, az1, rg0, rg1):", POL_CROP, "| phase conventions:", ph)
'''),
("md", "## 2. Data\nThe block is loaded once for all acquisitions and reordered to (azimuth, range[, acquisition])."),
("code", r'''
t0 = time.time()
ds = st.open_stack(cfg["stack_path"])
aux_names = ("dem_phases", "calibration_phases", "insar_wavenumbers", "incidence_angle", "latitude", "longitude",
             "sigma_nought_factor", "gamma_nought_factor", "noise_HH", "noise_HV", "noise_VH", "noise_VV")
data = st.load_crop(ds, POL_CROP, aux_names=aux_names)
to_yx = lambda a: np.ascontiguousarray(np.swapaxes(a, 0, 1))
aux = {k: to_yx(v) for k, v in data["aux"].items()}
slc = {p: to_yx(v) for p, v in data["slc"].items()}
acq = data["acquisition"]
d_az = float(ds.azimuth[1] - ds.azimuth[0]); d_rg = float(ds.range[1] - ds.range[0])
theta = np.deg2rad(np.nanmean(aux["incidence_angle"]))
REF = ph["reference_index"]
Ny, Nx, Nb = aux["insar_wavenumbers"].shape
del data
az_km = (POL_CROP[0] + np.arange(Ny)) * d_az / 1e3                      # along-track position [km]
gr_km = dp.ground_range_km(ds.range.values[POL_CROP[2]:POL_CROP[3]], aux["incidence_angle"].T)
print(f"{Ny} azimuth x {Nx} range samples, {Nb} acquisitions, loaded in {time.time() - t0:.0f} s; "
      f"block {gr_km[-1]:.0f} km (ground range) x {az_km[-1] - az_km[0]:.0f} km (azimuth)")
print(f"multilook {BOX[0]} x {BOX[1]} = {BOX[0] * d_az:.0f} m x {BOX[1] * d_rg / np.sin(theta):.0f} m ground, {BOX[0] * BOX[1]} looks")
'''),
("md", r"""
# Part A: single-acquisition polarimetry

## A1. Calibration and noise
The L1c SLC is β⁰. Power is converted with the `/radiometry` LUT factors, σ⁰ = β⁰·`sigma_nought_factor` (≈ sin θ) and γ⁰ = β⁰·`gamma_nought_factor` (≈ tan θ). The `/denoising` LUT gives the thermal-noise power per channel (β⁰); it is **not** removed in L1c. At P band the cross-pol is only a few dB above the noise, so noise inflates HV and pushes entropy towards 1. Noise is subtracted from the $\mathbf T_3$ diagonal and from $\mathrm{Re}\,T_{12}$:
$$T_{11},T_{22} \mathrel{-}= \tfrac12(N_{HH}+N_{VV}),\quad \mathrm{Re}\,T_{12} \mathrel{-}= \tfrac12(N_{HH}-N_{VV}),\quad T_{33} \mathrel{-}= \tfrac12(N_{HV}+N_{VH}).$$

**Noise-LUT check.** The noise LUT is not reliable in every product. In this stack, C02 reports HH/HV noise at about −4 dB, above the measured signal, although its backscatter matches the other dates. An acquisition whose median noise departs from the across-stack median by more than `NOISE_TOL_DB` in any channel is flagged. Its noise is replaced by the per-pixel median over the unflagged acquisitions.
"""),
("code", r'''
factor = {"sigma0": aux["sigma_nought_factor"], "gamma0": aux["gamma_nought_factor"], "beta0": None}[CALIBRATION]
levels = {p: 10 * np.log10(np.nanmedian(aux[f"noise_{p}"][::20, ::10], axis=(0, 1))) for p in polsar.POLS}
print(pd.DataFrame(levels, index=[f"C{int(a):02d}" for a in acq]).round(2).T.to_string())
BAD_NOISE = polsar.noise_check(levels, NOISE_TOL_DB)
good = [i for i in range(Nb) if i not in BAD_NOISE]
for p in polsar.POLS:
    if BAD_NOISE:
        aux[f"noise_{p}"][:, :, BAD_NOISE] = np.median(aux[f"noise_{p}"][:, :, good], axis=2, keepdims=True)
print("noise LUT flagged and replaced for:", [f"C{int(acq[i]):02d}" for i in BAD_NOISE] or "none")
noise = {p: aux[f"noise_{p}"][:, :, REF] for p in polsar.POLS}
ref_slc = {p: slc[p][:, :, REF] for p in polsar.POLS}
t0 = time.time()
T3_raw = polsar.coherency_T3(ref_slc, factor=factor, noise=None, box=BOX)
T3 = polsar.coherency_T3(ref_slc, factor=factor, noise=noise if NOISE_REMOVAL else None, box=BOX)
print(f"T3 in {time.time() - t0:.0f} s")
dec = (slice(STEP[0] // 2, None, STEP[0]), slice(STEP[1] // 2, None, STEP[1]))
T3d, T3d_raw = T3.isel(y=dec[0], x=dec[1]).persist(), T3_raw.isel(y=dec[0], x=dec[1]).persist()
lat_d, lon_d, inc_d = aux["latitude"][dec], aux["longitude"][dec], aux["incidence_angle"][dec]
valid_d = (np.abs(ref_slc["HH"]) > 0)[dec]
az_d, gr_d = az_km[dec[0]], gr_km[dec[1]]
show = lambda ax, img, **kw: dp.show(ax, img, az_d, gr_d, **kw)          # Part A maps (y, x)
bs_raw, bs = polsar.backscatter(T3d_raw), polsar.backscatter(T3d)
nesz = {p: (noise[p] * (factor if factor is not None else 1))[dec] for p in polsar.POLS}
'''),
("code", r'''
db = dp.to_db                         # dB; powers <= -100 dB (noise-clipped, no-data) -> NaN
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), constrained_layout=True)
inc_bins = np.linspace(np.nanmin(inc_d), np.nanmax(inc_d), 25); centres = 0.5 * (inc_bins[1:] + inc_bins[:-1])
for p, c in (("HH", "tab:blue"), ("VV", "tab:green"), ("XX", "tab:red")):
    v = bs_raw[p]
    axes[0].plot(centres, [np.nanmedian(db(v)[valid_d & (inc_d >= a) & (inc_d < b)]) for a, b in zip(inc_bins[:-1], inc_bins[1:])], color=c, label=f"{p} (with noise)")
axes[0].plot(centres, [np.nanmedian(db(nesz["HV"])[(inc_d >= a) & (inc_d < b)]) for a, b in zip(inc_bins[:-1], inc_bins[1:])], "k--", label="noise HV (NESZ)")
axes[0].set_xlabel("incidence [deg]"); axes[0].set_ylabel(f"{CALIBRATION} [dB]"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
axes[0].set_title("median backscatter and noise level vs incidence", fontsize=9)
for p, c in (("HH", "tab:blue"), ("VV", "tab:green"), ("XX", "tab:red")):
    axes[1].hist(db(bs[p])[valid_d].ravel(), bins=120, range=(-30, 5), histtype="step", color=c, density=True, label=f"{p}, noise removed")
    axes[1].hist(db(bs_raw[p])[valid_d].ravel(), bins=120, range=(-30, 5), histtype="step", color=c, ls=":", density=True)
axes[1].set_xlabel(f"{CALIBRATION} [dB]"); axes[1].legend(fontsize=8); axes[1].set_title("distributions (dotted: with noise)", fontsize=9)
plt.show()
snr = np.nanmedian(db(bs_raw["XX"])[valid_d] - db(nesz["HV"])[valid_d])
print(f"median cross-pol SNR {snr:.1f} dB")
'''),
("md", "## A2. Backscatter maps and Pauli RGB\nσ⁰ in HH, cross-pol (XX = (HV+VH)/2) and VV, each stretched between its own 2nd and 98th percentiles. The Pauli RGB uses R $T_{22}$ (double bounce, $|HH-VV|^2$), G $T_{33}$ (volume, $|HV+VH|^2$) and B $T_{11}$ (surface, $|HH+VV|^2$), each channel stretched in dB. Forest is green, ground–trunk interaction (flooded forest, river banks) red/magenta, and smooth surfaces and water blue/dark."),
("code", r'''
T11, T22, T33 = (np.where(valid_d, T3d[m].values, np.nan) for m in ("m11", "m22", "m33"))
pauli_rgb = dp.rgb_db(T22, T33, T11, STRETCH)
fig, axes = plt.subplots(2, 2, figsize=(15, 7.8), sharex=True, sharey=True, constrained_layout=True)
for ax, p in zip(axes.flat, ("HH", "XX", "VV")):
    v = np.where(valid_d, db(bs[p]), np.nan)
    lo, hi = dp.limits(v, STRETCH)
    im = show(ax, v, cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(f"{CALIBRATION} {p} [dB]", fontsize=9); fig.colorbar(im, ax=ax, shrink=0.8)
show(axes[1, 1], pauli_rgb); axes[1, 1].set_title("Pauli RGB (R: T22, G: T33, B: T11)", fontsize=9)
for ax in axes[:, 0]: ax.set_ylabel("azimuth [km]")
plt.show()
'''),
("md", r"""
## A2b. Pauli RGB GeoTIFF
As in notebook 3 (section 10), the Pauli image is placed on a regular lon/lat grid (EPSG:4326) by nearest neighbour from the LUT geolocation. Two files are written:
- `pauli_rgb_<block>.tif`: 8-bit RGB with the same dB stretch as above (0 = no-data), for direct display in QGIS or Google Earth;
- `pauli_db_<block>.tif`: three float bands $T_{22}$, $T_{33}$, $T_{11}$ in dB (σ⁰ units), for analysis or a different stretch.

The preview below is read back from the RGB GeoTIFF.
"""),
("code", r'''
import rasterio
pauli_tif = OUT / f"pauli_rgb_{blk}.tif"
rgb8 = dp.rgb_uint8(pauli_rgb, valid_d)
_, res = ref.to_geotiff({"R_T22": rgb8[..., 0], "G_T33": rgb8[..., 1], "B_T11": rgb8[..., 2]}, lat_d, lon_d, pauli_tif, rgb=True)
pauli_db_tif = OUT / f"pauli_db_{blk}.tif"
ref.to_geotiff({"T22_dB": db(T22), "T33_dB": db(T33), "T11_dB": db(T11)}, lat_d, lon_d, pauli_db_tif)
print(f"{pauli_tif.name} and {pauli_db_tif.name}: {res:.5f} deg ≈ {res * 111e3:.0f} m")
with rasterio.open(pauli_tif) as src:
    img, b = src.read().transpose(1, 2, 0), src.bounds
fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
ax.imshow(np.where(img.any(axis=-1, keepdims=True), img, 255), extent=[b.left, b.right, b.bottom, b.top], interpolation="none")
ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(lat_d)))); ax.set_xlabel("longitude [deg]"); ax.set_ylabel("latitude [deg]")
ax.set_title(f"Pauli RGB GeoTIFF (EPSG:4326), C{int(acq[REF]):02d}", fontsize=10)
plt.show()
'''),
("md", r"""
## A3. Cloude–Pottier $H/A/\alpha$
The eigen-decomposition of $\mathbf T_3$ gives the **entropy** $H$ (0 = one mechanism, 1 = random), the **anisotropy** $A$ (relative weight of the 2nd and 3rd mechanisms) and the mean **α** angle (0° surface, 45° dipole/volume, 90° double bounce). Tropical forest at P band is expected in zone Z5/Z2 (high entropy, α ≈ 40–55°). The plane is shown with and without noise removal.
"""),
("code", r'''
from polsarpro.decompositions import h_a_alpha, freeman
t0 = time.time()
haa = h_a_alpha(T3d, boxcar_size=[1, 1], flags=("entropy", "anisotropy", "alpha")).compute()
haa_raw = h_a_alpha(T3d_raw, boxcar_size=[1, 1], flags=("entropy", "anisotropy", "alpha")).compute()
print(f"H/A/alpha in {time.time() - t0:.0f} s")
fig, axes = plt.subplots(2, 2, figsize=(15, 7.8), sharex=True, sharey=True, constrained_layout=True)
for ax, (n, cmap) in zip(axes.flat, (("entropy", "viridis"), ("anisotropy", "cividis"), ("alpha", "plasma"))):
    v = np.where(valid_d, haa[n].values, np.nan)
    lo, hi = dp.limits(v, STRETCH)
    im = show(ax, v, cmap=cmap, vmin=lo, vmax=hi)
    ax.set_title(f"{n} (stretch {lo:.2f}–{hi:.2f})", fontsize=9); fig.colorbar(im, ax=ax, shrink=0.8)
v = np.where(valid_d, haa_raw.entropy.values - haa.entropy.values, np.nan)
im = show(axes[1, 1], v, cmap="magma", vmin=0, vmax=dp.limits(v, STRETCH)[1])
axes[1, 1].set_title("entropy increase caused by noise", fontsize=9); fig.colorbar(im, ax=axes[1, 1], shrink=0.8)
for ax in axes[:, 0]: ax.set_ylabel("azimuth [km]")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1]})
H, Al = haa.entropy.values[valid_d], haa.alpha.values[valid_d]
axes[0].hist2d(H, Al, bins=[200, 180], range=[[0, 1], [0, 90]], cmap="magma_r", cmin=1, norm="log")
polsar.h_alpha_plane(axes[0]); axes[0].set_title(f"H–α plane (log density), noise {'removed' if NOISE_REMOVAL else 'kept'}", fontsize=9)
axes[1].hist(haa_raw.entropy.values[valid_d], bins=200, range=(0, 1), histtype="step", label="with noise")
axes[1].hist(H, bins=200, range=(0, 1), histtype="step", label="noise removed")
axes[1].set_xlim(*dp.limits([H, haa_raw.entropy.values[valid_d]], (0.5, 99.9))); axes[1].set_xlabel("entropy H"); axes[1].legend(fontsize=8)
plt.show()
'''),
("md", "## A4. Freeman–Durden\nThree-component model-based decomposition into surface ($P_s$), double-bounce ($P_d$) and volume ($P_v$) power. Shown as an RGB and as fractions of the total."),
("code", r'''
fd = freeman(T3d, boxcar_size=[1, 1]).compute()
Ps, Pd, Pv = (np.clip(fd[k].values, 0, None) for k in ("odd", "double", "volume"))
tot = Ps + Pd + Pv
m = lambda v: np.where(valid_d & (tot > 0), v, np.nan)
fig, axes = plt.subplots(2, 2, figsize=(15, 7.8), sharex=True, sharey=True, constrained_layout=True)
show(axes[0, 0], dp.rgb_db(m(Pd), m(Pv), m(Ps), STRETCH)); axes[0, 0].set_title("Freeman–Durden RGB (R: Pd, G: Pv, B: Ps)", fontsize=9)
for ax, (n, v, cmap) in zip([axes[0, 1], axes[1, 0], axes[1, 1]],
                            (("Pv / total", Pv, "Greens"), ("Ps / total", Ps, "Blues"), ("Pd / total", Pd, "Reds"))):
    f_ = m(v / tot)
    lo, hi = dp.limits(f_, STRETCH)
    im = show(ax, f_, cmap=cmap, vmin=lo, vmax=hi); ax.set_title(f"{n} (stretch {lo:.2f}–{hi:.2f})", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
for ax in axes[:, 0]: ax.set_ylabel("azimuth [km]")
plt.show()
fr = pd.DataFrame({k: np.nanpercentile((v / tot)[valid_d], [10, 50, 90]) for k, v in (("Ps", Ps), ("Pd", Pd), ("Pv", Pv))},
                  index=["p10", "median", "p90"])
fr.round(3)
'''),
("md", r"""
## A5. Wishart $H/A/\alpha$ classification
Unsupervised: an initial partition by the $H$–$\alpha$ zones, refined by iterations of the complex Wishart distance to the class centres, then split into 16 classes by anisotropy. Classes are data-driven; check their mean $H$, α and Pauli powers before interpreting them.
"""),
("code", r'''
from polsarpro.classification import wishart_h_a_alpha
t0 = time.time()
cls = wishart_h_a_alpha(T3d, boxcar_size=[1, 1], max_iter=10, verbose=False).compute()
print(f"Wishart in {time.time() - t0:.0f} s")
labels = cls["wishart_h_a_alpha_class"].values.astype(float)
labels[~valid_d] = np.nan
fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
im = show(ax, labels, cmap="tab20", vmin=0.5, vmax=16.5)
fig.colorbar(im, ax=ax, shrink=0.8, label="class", ticks=range(1, 17)); ax.set_ylabel("azimuth [km]")
ax.set_title("Wishart H/A/α classes", fontsize=9)
plt.show()
summary = pd.DataFrame([{"class": int(c), "fraction_%": (labels == c).mean() * 100,
                         "H": np.nanmean(haa.entropy.values[labels == c]), "A": np.nanmean(haa.anisotropy.values[labels == c]),
                         "alpha": np.nanmean(haa.alpha.values[labels == c]),
                         "Pv_frac": np.nanmean((Pv / tot)[labels == c]), "XX_dB": np.nanmedian(db(bs["XX"])[labels == c])}
                        for c in np.unique(labels[np.isfinite(labels)])]).set_index("class")
summary.round(2)
'''),
("md", r"""
## A6. Temporal stability over the 7 acquisitions
Median σ⁰ per channel (noise removed, with the checked noise of section A1) and the cross-pol ratio for every acquisition, over the block. Over 18 days in the dry-to-wet transition, changes of more than ~0.5 dB point to rain or soil-moisture events. Those also show up as temporal decorrelation in the tomography.
"""),
("code", r'''
rows = []
f = factor if factor is not None else 1.0
for i in range(Nb):
    med = {}
    for p in polsar.POLS:
        I = np.abs(slc[p][:, :, i]) ** 2
        ok = I > 0
        med[p] = 10 * np.log10(np.median(((I - (aux[f"noise_{p}"][:, :, i] if NOISE_REMOVAL else 0)) * f)[ok][::7]))
    rows.append({"cycle": f"C{int(acq[i]):02d}", "date": str(ds.product_id.values[i])[15:23], **{f"{p}_dB": med[p] for p in polsar.POLS},
                 "HV/HH_dB": med["HV"] - med["HH"]})
temporal = pd.DataFrame(rows).set_index("cycle")
fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
for p in polsar.POLS:
    ax.plot(temporal.index, temporal[f"{p}_dB"], "o-", label=p)
ax.set_ylabel(f"median {CALIBRATION} [dB]"); ax.legend(ncol=4, fontsize=8); ax.grid(alpha=0.3)
plt.show()
temporal.round(2)
'''),
("md", r"""
# Part B: polarimetric tomography

## B1. Phase correction and vertical focusing
The same LUT corrections and signs as in notebook 2 are applied to every acquisition. The four channels are then focused at each height in `HEIGHTS`. The boxcar for the per-height decompositions is `BOX`, with one output per window. Power-ratio parameters ($H$, α, fractions) are unaffected by the per-pixel calibration factor. Noise is not removed here.
"""),
("code", r'''
corr = np.exp(1j * (ph["sign_dem"] * aux["dem_phases"] + ph["sign_cal"] * aux["calibration_phases"])).astype(np.complex64)
for p in slc:
    slc[p] *= corr
del corr
kz = (ph["kz_sign"] * aux["insar_wavenumbers"]).astype(np.float32)
t0 = time.time()
cube, (sy, sx) = poltomo.polarimetric_cube(
    slc, kz, HEIGHTS, boxcar=BOX, step=BOX, focusing=FOCUSING,
    capon_kw=dict(window_m=CAPON_WINDOW_M, d_az=d_az, d_rg=d_rg, theta=theta, d_load=CAPON_LOAD),
    progress=lambda k, z: print(f"\rz = {int(z):+3d} m ({k + 1}/{len(HEIGHTS)})", end="", flush=True))
print(f"\nfocused and decomposed in {time.time() - t0:.0f} s; cube {cube['T11'].shape} (height, y, x)")
lat_s, lon_s = aux["latitude"][sy, sx], aux["longitude"][sy, sx]
valid_s = (np.abs(slc["HH"][:, :, REF]) > 0)[sy, sx]
az_s, gr_s = az_km[sy][:cube["T11"].shape[1]], gr_km[sx][:cube["T11"].shape[2]]
'''),
("md", r"""
## B2. Vertical profiles
Median and interquartile range of each parameter against height. At P band one expects ground and ground–trunk double bounce near $z = 0$ (higher $T_{22}$/span, $P_d$, larger α), random volume in the canopy ($T_{33}$, $P_v$, $H \to 1$, α ≈ 45°), and only side-lobe leakage above the canopy top.
"""),
("code", r'''
panels = {"Pauli fraction": {"T11/span": "T11_frac", "T22/span": "T22_frac", "T33/span": "T33_frac"},
          "Freeman fraction": {"Ps": "Ps_frac", "Pd": "Pd_frac", "Pv": "Pv_frac"},
          "entropy / anisotropy": {"H": "entropy", "A": "anisotropy"},
          "alpha [deg]": {"alpha": "alpha"},
          "span [dB]": {"span": "span"}}
colours = ["tab:blue", "tab:red", "tab:green"]
fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 4.5), sharey=True, constrained_layout=True)
for ax, (title, items) in zip(axes, panels.items()):
    for c, (lab, name) in zip(colours, items.items()):
        v = cube[name][:, valid_s]
        if name == "span":
            v = 10 * np.log10(v)
        q = np.nanpercentile(v, [25, 50, 75], axis=1)
        ax.plot(q[1], HEIGHTS, color=c, label=lab); ax.fill_betweenx(HEIGHTS, q[0], q[2], color=c, alpha=0.15)
    ax.axhline(0, color="k", lw=0.8); ax.set_title(title, fontsize=9); ax.legend(fontsize=8); ax.grid(alpha=0.3)
axes[0].set_ylabel("height above ground [m]")
plt.show()
'''),
("md", "## B3. Height slices\nPauli RGB, Freeman RGB, entropy and α at selected heights. Each RGB channel has one scaling shared by all heights, so differences between slices reflect vertical structure."),
("code", r'''
idx = [int(np.argmin(np.abs(HEIGHTS - h))) for h in MAP_HEIGHTS]
vm = lambda n: np.where(valid_s[None], cube[n], np.nan)
rgbs = {"Pauli RGB": dp.rgb_db(vm("T22"), vm("T33"), vm("T11"), STRETCH),
        "Freeman RGB": dp.rgb_db(vm("Pd"), vm("Pv"), vm("Ps"), STRETCH),
        "entropy H": vm("entropy"), "alpha": vm("alpha")}
fig, axes = plt.subplots(len(rgbs), len(idx), figsize=(3.7 * len(idx), 2.15 * len(rgbs) + 0.6),
                         sharex=True, sharey=True, constrained_layout=True)
for row, (title, vol) in enumerate(rgbs.items()):
    if vol.ndim == 3:
        lo, hi = dp.limits([vol[k] for k in idx], STRETCH)          # one scale for all heights
    for col, k in enumerate(idx):
        ax = axes[row, col]
        if vol.ndim == 4:
            dp.show(ax, vol[k], az_s, gr_s)
        else:
            im = dp.show(ax, vol[k], az_s, gr_s, cmap="viridis" if title.startswith("entropy") else "plasma", vmin=lo, vmax=hi)
        ax.set_xlabel("")
        if row == 0: ax.set_title(f"z = {HEIGHTS[k]} m", fontsize=10)
        if col == 0: ax.set_ylabel(f"{title}\nazimuth [km]", fontsize=9)
    if vol.ndim == 3:
        fig.colorbar(im, ax=axes[row, :], shrink=0.8, pad=0.01, label=f"{lo:.2f}–{hi:.2f}")
for ax in axes[-1]: ax.set_xlabel("ground range [km]")
plt.show()

slice_tifs = []
for h in GEOTIFF_HEIGHTS:
    k = int(np.argmin(np.abs(HEIGHTS - h)))
    rgb8 = dp.rgb_uint8(rgbs["Pauli RGB"][k], valid_s)
    t = OUT / f"pauli_rgb_z{int(HEIGHTS[k]):+03d}m_{FOCUSING}_{blk}.tif"
    ref.to_geotiff({"R_T22": rgb8[..., 0], "G_T33": rgb8[..., 1], "B_T11": rgb8[..., 2]}, lat_s, lon_s, t, rgb=True)
    slice_tifs.append(str(t))
print("Pauli RGB GeoTIFFs of height slices:", *[pathlib.Path(t).name for t in slice_tifs], sep="\n  ")
'''),
("md", "## B4. Vertical section with forest height\nPauli RGB along azimuth at the middle range of the block, averaged over 5 range cells and stretched on the section itself (2.5 m height steps, bilinear display), with the heights from notebook 3 matched by geolocation: canopy top (white), bias-corrected canopy top (yellow), the reference `ref_mean` (cyan, dashed) and the phase centre (white, dotted)."),
("code", r'''
xc = cube["T11"].shape[2] // 2
sec = {n: np.nanmean(vm(n)[:, :, max(xc - 2, 0):xc + 3], axis=2) for n in ("T22", "T33", "T11")}
section = dp.rgb_db(sec["T22"], sec["T33"], sec["T11"], STRETCH)  # stretched on the section itself
sec_km = az_s
dz = HEIGHTS[1] - HEIGHTS[0]
fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
ax.imshow(section, aspect="auto", origin="lower", interpolation="bilinear",
          extent=[sec_km[0], sec_km[-1], HEIGHTS[0] - dz / 2, HEIGHTS[-1] + dz / 2])
if "forest_height" in cfg and pathlib.Path(cfg["forest_height"]["heights"]).exists():
    from scipy.spatial import cKDTree
    with xr.open_dataset(cfg["forest_height"]["heights"]) as f_:
        Hh = f_.load()                                       # load and release the file (no lock left open)
    ok = np.isfinite(Hh.latitude.values.ravel())
    tree = cKDTree(np.column_stack([Hh.latitude.values.ravel()[ok], Hh.longitude.values.ravel()[ok]]))
    _, nn = tree.query(np.column_stack([lat_s[:, xc], lon_s[:, xc]]))
    for name, style in (("h_canopy_top", "w-"), ("h_canopy_top_corrected", "y-"), ("ref_mean", "c--"), ("z_phase_centre", "w:")):
        if name in Hh:
            ax.plot(sec_km, Hh[name].values.ravel()[ok][nn], style, lw=1.4, label=name)
    ax.legend(fontsize=8, loc="upper right")
ax.axhline(0, color="w", ls="--", lw=0.6)
ax.set_xlabel("azimuth [km]"); ax.set_ylabel("height [m]"); ax.set_title("Pauli RGB section (R: T22, G: T33, B: T11)", fontsize=9)
plt.show()
'''),
("md", "## B5. Polarimetry vs forest height\nGround-level double-bounce fraction and canopy-level volume fraction and entropy against the canopy-top height of notebook 3. This tests whether the vertical polarimetric signature follows forest structure."),
("code", r'''
if "forest_height" in cfg:
    h_top = Hh.h_canopy_top.values.ravel()[ok][tree.query(np.column_stack([lat_s.ravel(), lon_s.ravel()]))[1]].reshape(lat_s.shape)
    k0, k20 = int(np.argmin(np.abs(HEIGHTS - 0))), int(np.argmin(np.abs(HEIGHTS - 20)))
    feats = {"T22/span at z=0": cube["T22_frac"][k0], "Pv fraction at z=20 m": cube["Pv_frac"][k20],
             "entropy at z=20 m": cube["entropy"][k20], "alpha at z=0": cube["alpha"][k0]}
    fig, axes = plt.subplots(1, len(feats), figsize=(3.4 * len(feats), 3.4), constrained_layout=True)
    for ax, (lab, v) in zip(axes, feats.items()):
        m = valid_s & np.isfinite(v) & np.isfinite(h_top)
        ax.hist2d(h_top[m], v[m], bins=40, cmap="Greys", cmin=1)
        r = np.corrcoef(h_top[m], v[m])[0, 1] if m.sum() > 3 else np.nan
        ax.set_xlabel("h_canopy_top [m]"); ax.set_ylabel(lab, fontsize=9); ax.set_title(f"r = {r:.2f}, n = {m.sum()}", fontsize=9)
    plt.show()
else:
    print("run notebook 3 for the forest height")
'''),
("md", r"""
## Saving
- `polarimetry/polsar_<block>.nc`: Part A maps (σ⁰ per channel, $\mathbf T_3$ diagonal, $H/A/\alpha$, Freeman–Durden, Wishart class, geolocation) on the decimated grid.
- `polarimetry/poltomo_<focusing>_<block>.nc`: the Part B cube (height, y, x).
- GeoTIFFs written above: `pauli_rgb_<block>.tif`, `pauli_db_<block>.tif` and `pauli_rgb_z<height>m_<focusing>_<block>.tif`.

Both are recorded in the manifest under `polarimetry`.
"""),
("code", r'''
yx = ("y", "x")
A = xr.Dataset({**{f"{CALIBRATION}_{p}": (yx, bs[p].astype(np.float32)) for p in bs},
                **{f"T{k}{k}": (yx, T3d[f"m{k}{k}"].values.astype(np.float32)) for k in (1, 2, 3)},
                **{n: (yx, haa[n].values.astype(np.float32)) for n in ("entropy", "anisotropy", "alpha")},
                "Ps": (yx, Ps.astype(np.float32)), "Pd": (yx, Pd.astype(np.float32)), "Pv": (yx, Pv.astype(np.float32)),
                "wishart_class": (yx, labels.astype(np.float32)), "valid": (yx, valid_d.astype(np.int8)),
                "latitude": (yx, lat_d.astype(np.float32)), "longitude": (yx, lon_d.astype(np.float32))},
               coords={"y": POL_CROP[0] + np.arange(T3d.sizes["y"]) * STEP[0] + STEP[0] // 2,
                       "x": POL_CROP[2] + np.arange(T3d.sizes["x"]) * STEP[1] + STEP[1] // 2},
               attrs={"stack": cfg["stack_path"], "acquisition": f"C{int(acq[REF]):02d}", "calibration": CALIBRATION,
                      "noise_removed": int(NOISE_REMOVAL), "boxcar": str(BOX), "step": str(STEP),
                      "layout": "y = azimuth line, x = range sample (stack indices)"})
pathA = OUT / f"polsar_{blk}.nc"
st.save_netcdf(A, pathA)

B = xr.Dataset({k: (("height", "y", "x"), v.astype(np.float32)) for k, v in cube.items()},
               coords={"height": HEIGHTS, "y": POL_CROP[0] + np.arange(cube["T11"].shape[1]) * BOX[0] + BOX[0] // 2,
                       "x": POL_CROP[2] + np.arange(cube["T11"].shape[2]) * BOX[1] + BOX[1] // 2})
B["latitude"] = (("y", "x"), lat_s.astype(np.float32)); B["longitude"] = (("y", "x"), lon_s.astype(np.float32))
B.attrs = {"stack": cfg["stack_path"], "focusing": FOCUSING, "boxcar": str(BOX), **{k: str(v) for k, v in ph.items()},
           "layout": "y = azimuth line, x = range sample (stack indices)"}
pathB = OUT / f"poltomo_{FOCUSING}_{blk}.nc"
st.save_netcdf(B, pathB)

cfg["polarimetry"] = {"block": POL_CROP, "polsar": str(pathA), "poltomo": str(pathB), "calibration": CALIBRATION,
                      "noise_removed": NOISE_REMOVAL, "focusing": FOCUSING,
                      "pauli_geotiff": str(pauli_tif), "pauli_db_geotiff": str(pauli_db_tif), "pauli_slice_geotiffs": slice_tifs}
st.write_manifest(MANIFEST, **cfg)
print("saved", pathA, "and", pathB)
'''),
]

if __name__ == "__main__":
    import sys
    wanted = sys.argv[1:]                       # e.g. "02 04" writes only those; default all
    for name, cells in [("00_Data_Access.ipynb", NB0), ("01_Stack_Build.ipynb", NB1), ("02_Tomography.ipynb", NB2),
                        ("03_Forest_Height.ipynb", NB3), ("04_Polarimetric_Analysis.ipynb", NB4)]:
        if wanted and name[:2] not in wanted:
            continue
        nbf.write(nb(cells), HERE / name)
        print("wrote", HERE / name)
