"""
Helpers for the BIOMASS PolSAR notebooks: Pauli (step 3), Cloude–Pottier H/A/alpha/beta (step 4),
AOI maps (step 5), the circular polarisation basis (step 6) and interferometry (step 7).

The BIOMASS products are opened with ESA's BiomassProduct.py, which must be in the same folder.
"""

import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import zipfile
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from IPython.display import Image as IPythonImage, display
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Polygon as MplPolygon, Rectangle
from matplotlib.path import Path as MplPath
from rasterio.control import GroundControlPoint
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform as warp_transform
from scipy.ndimage import map_coordinates, uniform_filter

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # find BiomassProduct.py next to this file, whatever the notebook folder is


def _import_biomass_product():
    """ESA's BiomassProductSTA from BiomassProduct.py next to this file (also e.g. 'BiomassProduct (1).py')."""
    try:
        from BiomassProduct import BiomassProductSTA
        return BiomassProductSTA
    except ModuleNotFoundError as err:
        candidates = sorted(HERE.glob("BiomassProduct*.py"))
        if err.name != "BiomassProduct" or not candidates:
            raise
        spec = importlib.util.spec_from_file_location("BiomassProduct", candidates[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules["BiomassProduct"] = module
        spec.loader.exec_module(module)
        return module.BiomassProductSTA


try:
    # hide the harmless SyntaxWarning from the Windows paths in ESA's test code at the end of the file
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        BiomassProductSTA = _import_biomass_product()
except ImportError as err:
    py_files = ", ".join(sorted(f.name for f in HERE.glob("*.py"))) or "none"
    raise ImportError(
        f"Could not import BiomassProduct.py: {err}\n"
        f"polsar_utils.py is in {HERE}, which contains these .py files: {py_files}\n"
        "Put BiomassProduct.py in that folder.\n"
        "If a package is missing, run the 'First time only' cell (or: %pip install --user <package name>) "
        "and restart the kernel."
    ) from err

POLS = ("HH", "HV", "VH", "VV")
C = 299792458.0
DATE_RE = re.compile(r"(\d{8})[tT](\d{6})")
DATA_DIR_ENV = "BIOMASS_DATA_DIR"


# =============================================================================
# Finding and selecting scenes
# =============================================================================

@dataclass
class Scene:
    """One BIOMASS STA product (one acquisition of a stack)."""
    path: Path
    data_dir: Path
    stack: str       # short stack name, normally the stack folder (e.g. stack_01)
    stack_id: str    # bio:stackID from the product XML ("" if missing)
    date: datetime
    is_primary: bool = False
    _product: object = field(default=None, repr=False)

    @property
    def name(self):
        return self.path.name

    @property
    def stack_label(self):
        return f"{self.stack} ({self.stack_id})" if self.stack_id and self.stack_id != self.stack else self.stack

    @property
    def label(self):
        primary = "  ★ primary" if self.is_primary else ""
        return f"{self.stack} | {self.date:%Y-%m-%d %H:%M}{primary} | {self.name}"

    @property
    def product(self):
        """ESA BiomassProductSTA object, opened on first use."""
        if self._product is None:
            with contextlib.redirect_stdout(io.StringIO()):
                self._product = BiomassProductSTA(str(self.path))
        return self._product


@lru_cache(maxsize=None)
def _xml_values(xml_path):
    """First text value of every tag in an XML file (namespaces ignored)."""
    values = {}
    for el in ET.parse(xml_path).iter():
        tag = el.tag.split("}")[-1]
        if el.text and el.text.strip() and tag not in values:
            values[tag] = el.text.strip()
    return values


def _first_file(folder, pattern):
    files = sorted(f for f in Path(folder).glob(pattern) if ".ipynb_checkpoints" not in f.parts)
    return files[0] if files else None


def _product_info(product_dir):
    """(isCoregistrationPrimary, stackID) from the main product XML, the same file ESA's class reads."""
    mph = _first_file(product_dir, "bio*sta*.xml")
    if mph is None:
        return None, ""
    values = _xml_values(str(mph))
    flag = values.get("isCoregistrationPrimary")
    return (None if flag is None else flag.lower() == "true"), values.get("stackID", "")


def _search_data_dirs(roots, max_depth=6, max_folders=20000):
    """Folders that contain BIOMASS STA products, found by a limited search below the given roots."""
    found, visited = [], 0
    for root in roots:
        queue = [(Path(root).expanduser(), 0)]
        while queue and visited < max_folders:
            folder, depth = queue.pop(0)
            visited += 1
            try:
                entries = [e for e in os.scandir(folder) if e.is_dir() and not e.name.startswith(".")]
            except OSError:
                continue
            products = [e for e in entries if e.name.startswith("BIO_") and "STA" in e.name]
            if products:
                # <data_dir>/<stack>/[L1C/]BIO_...: the data folder is the one above the stack folder
                stack_folder = folder.parent if folder.name.upper() == "L1C" else folder
                found.append(stack_folder.parent)
                continue
            if depth < max_depth:
                queue += [(Path(e.path), depth + 1) for e in entries]
    return list(dict.fromkeys(found))


def find_data_dir(data_dir=None):
    """
    The folder with the stacks, in this order:
    1. data_dir (DATA_DIR in the notebook): a local folder or a mounted cloud bucket
    2. the environment variable BIOMASS_DATA_DIR
    3. a search for BIOMASS products below the notebook folder and the home folder
    """
    if data_dir is None and os.environ.get(DATA_DIR_ENV):
        data_dir = os.environ[DATA_DIR_ENV]
        print(f"Using {DATA_DIR_ENV} = {data_dir}")
    if data_dir is not None:
        data_dir = Path(data_dir).expanduser()
        if not data_dir.is_dir():
            raise FileNotFoundError(f"The data folder does not exist: {data_dir}")
        return data_dir

    print("DATA_DIR not set, searching for BIOMASS products below this folder and your home folder...")
    candidates = _search_data_dirs([Path.cwd(), Path.home()])
    if len(candidates) == 1:
        print(f"Found: {candidates[0]}")
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "No BIOMASS products found. Set DATA_DIR to the folder that contains your stacks, e.g.\n"
            '    DATA_DIR = "/path/to/biomass_inputs"')
    raise FileNotFoundError(
        "Several folders with BIOMASS products were found. Copy the one you want into DATA_DIR:\n"
        + "\n".join(f'    DATA_DIR = "{c}"' for c in candidates))


def find_primary(scenes):
    """
    The primary (coregistration reference) of a stack.
    Uses isCoregistrationPrimary from the product XML; if that is missing,
    the primaryImage named in the coregistration annotation.
    """
    flagged = [s for s in scenes if s.is_primary]
    if flagged:
        return flagged[0]
    for scene in scenes:
        annot = _first_file(scene.path / "annotation_coregistered", "*_annot.xml")
        primary_image = _xml_values(str(annot)).get("primaryImage") if annot else None
        m = DATE_RE.search(primary_image or "")
        if m:
            date = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            match = next((s for s in scenes if s.date == date), None)
            if match:
                match.is_primary = True
                return match
    return None


def find_stacks(data_dir=None):
    """
    All stacks in the data folder as {stack name: [scenes sorted by date]}.
    A stack is a stack folder (e.g. stack_01); its real ID (bio:stackID) is read from the product XML.
    Products that lie directly in the data folder are grouped by their stack ID.
    """
    data_dir = find_data_dir(data_dir)
    paths = set()
    for pattern in ("*/L1C/BIO_*STA*", "*/BIO_*STA*", "BIO_*STA*"):
        # a product folder contains a measurement folder (skips wrapper folders, e.g. from unzipping)
        paths.update(p for p in data_dir.glob(pattern) if (p / "measurement").is_dir())
    entries = []
    for path in sorted(paths):
        m = DATE_RE.search(path.name)
        if not m:
            continue
        is_primary, stack_id = _product_info(path)
        parts = path.relative_to(data_dir).parts
        # the stack folder is the first folder below data_dir, unless the product lies loose or in a
        # wrapper folder named like a product; then the product is grouped by its stack ID
        folder = parts[0] if len(parts) > 1 and not parts[0].startswith("BIO_") else ""
        date = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        entries.append((path, folder, stack_id, date, bool(is_primary)))

    ids_per_folder = {}
    for _, folder, stack_id, _, _ in entries:
        ids_per_folder.setdefault(folder, set()).add(stack_id)
    stacks = {}
    for path, folder, stack_id, date, is_primary in entries:
        if folder and len(ids_per_folder[folder]) == 1:
            name = folder
        elif folder:  # several stacks in one folder: add the end of the ID
            name = f"{folder}_{stack_id.split('_')[-1]}" if stack_id else folder
        else:
            name = stack_id or "no_stack_id"
        stacks.setdefault(name, []).append(Scene(path, data_dir, name, stack_id, date, is_primary))
    for scenes in stacks.values():
        scenes.sort(key=lambda s: s.date)
        find_primary(scenes)
    if not stacks:
        raise FileNotFoundError(f"No BIOMASS STA products found in {data_dir}")
    return dict(sorted(stacks.items()))


def print_overview(stacks):
    """Numbered list of all stacks and scenes, with the stack ID and the primary marked ★."""
    print(f"{len(stacks)} stack(s), {sum(len(s) for s in stacks.values())} scene(s)\n")
    for i, (name, scenes) in enumerate(stacks.items()):
        print(f"[{i}] {name}  ({len(scenes)} scenes)")
        if scenes[0].stack_id and scenes[0].stack_id != name:
            print(f"     ID: {scenes[0].stack_id}")
        for j, s in enumerate(scenes):
            primary = "★ primary" if s.is_primary else "         "
            print(f"      [{j}] {s.date:%Y-%m-%d %H:%M}  {primary}  {s.name}")
    print("\n★ = primary (coregistration reference) of the stack")


def _as_list(value):
    """Accepts a list, a single value, or text such as "[0, 2]", "0" or "stack_01, stack_20"."""
    items = list(value) if isinstance(value, (list, tuple)) else [value]
    result = []
    for item in items:
        if isinstance(item, str):
            for part in item.strip("[]() ").split(","):
                part = part.strip().strip("'\"")
                if part:
                    result.append(int(part) if part.isdigit() and len(part) < 5 else part)  # long digits = date
        else:
            result.append(item)
    return result


def _is_all(value):
    return isinstance(value, str) and value.strip().lower() == "all"


def _match_stack(stacks, item):
    """Stack name for a number, a stack name, a full stack ID, or part of a stack ID (e.g. "DZ1NQJ")."""
    names = list(stacks)
    if isinstance(item, int):
        if not 0 <= item < len(names):
            raise ValueError(f"Stack number {item} does not exist. Choose between 0 and {len(names) - 1}.")
        return names[item]
    text = str(item).strip()
    if text in stacks:
        return text
    exact = [n for n in names if stacks[n][0].stack_id == text]
    if exact:
        return exact[0]
    partial = [n for n in names if text.lower() in n.lower() or text.lower() in stacks[n][0].stack_id.lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise ValueError(f"'{text}' matches several stacks: {', '.join(partial)}. Be more specific.")
    raise ValueError(f"Stack '{text}' not found. Available: {', '.join(names)} (or their stack IDs)")


def _choose_stacks(stacks, which_stacks):
    if _is_all(which_stacks):
        return list(stacks)
    return list(dict.fromkeys(_match_stack(stacks, item) for item in _as_list(which_stacks)))


def print_primaries(stacks, which_stacks="all"):
    """Prints the primary of each chosen stack and returns them as {stack: scene}."""
    primaries = {}
    for name in _choose_stacks(stacks, which_stacks):
        primary = find_primary(stacks[name])
        primaries[name] = primary
        print(f"{stacks[name][0].stack_label}")
        if primary:
            print(f"    ★ {primary.date:%Y-%m-%d %H:%M}  {primary.name}\n      {primary.path}")
        else:
            print("    no primary found")
    return primaries


def select_scenes(stacks, which_stacks="all", which_scenes="all"):
    """
    The scenes to process.
    which_stacks: "all", or a list of stack names, stack IDs (or the end of an ID) or numbers.
    which_scenes: "all", "primary", or a list of dates ("2026-09-15") / numbers within each stack.
    """
    chosen = _choose_stacks(stacks, which_stacks)
    if isinstance(which_scenes, str) and which_scenes.strip().lower() in ("all", "primary"):
        which_scenes = which_scenes.strip().lower()

    selected = []
    for name in chosen:
        scenes = stacks[name]
        if which_scenes == "all":
            selected += scenes
        elif which_scenes == "primary":
            primary = find_primary(scenes)
            if primary:
                selected.append(primary)
            else:
                print(f"Warning: no primary found in {name}, skipped")
        else:
            for item in _as_list(which_scenes):
                if isinstance(item, int):
                    if 0 <= item < len(scenes):
                        selected.append(scenes[item])
                    else:
                        print(f"Warning: {name} has no scene number {item} (0-{len(scenes) - 1}), skipped")
                else:
                    digits = re.sub(r"\D", "", str(item))
                    selected += [s for s in scenes if f"{s.date:%Y%m%d%H%M%S}".startswith(digits)]

    if not selected:
        raise ValueError("No scenes match STACKS / SCENES. Check the overview list.")
    print(f"Selected {len(selected)} scene(s) in {len(chosen)} stack(s):")
    for s in selected:
        print("  ", s.label)
    return selected


# =============================================================================
# Reading data
# =============================================================================

def _band_pols(src):
    pols = []
    for i, desc in enumerate(src.descriptions):
        meta = " ".join([desc or ""] + [str(v) for v in src.tags(i + 1).values()]).upper()
        m = re.search(r"\b(HH|HV|VH|VV)\b", meta)
        pols.append(m.group(1) if m else None)
    if None in pols:
        if src.count == 4:
            return list(POLS)  # BIOMASS band order
        raise ValueError(f"Cannot determine the polarisations of {src.name}")
    return pols


def _valid_mask(amp, nodata=None):
    """Valid pixels: finite, > 0, not the nodata value, and not identical in all bands (fill value)."""
    valid = np.isfinite(amp).all(0) & (amp > 0).all(0)
    if nodata is not None:
        valid &= (amp != nodata).all(0)
    valid &= ~(amp == amp[:1]).all(0)
    return valid


def _phase_to_radians(pha, valid):
    """
    Phase in radians and the unit it was stored in. Degrees are recognised from the valid pixels only:
    no-data values (e.g. -9999) at the image edges must not trigger a conversion of phases that are in radians.
    """
    values = np.abs(pha[..., valid])[..., ::97]
    if values.size and np.percentile(values, 99.9) > 2 * np.pi + 0.1:
        return np.deg2rad(pha), "degrees"
    return pha, "radians"


def read_scene(scene, window=None):
    """
    Complex SLC per polarisation, validity mask, and the GCPs of the image (for GeoTIFF export).
    The phase is converted to radians if it is stored in degrees.
    """
    product = scene.product
    with rasterio.open(product.measurement_abs_file) as a, rasterio.open(product.measurement_phase_file) as p:
        pols = _band_pols(a)
        amp = a.read(window=window).astype(np.float32)
        pha = p.read(window=window).astype(np.float32)
        gcps, gcp_crs = a.gcps
    valid = _valid_mask(amp, product.noDataValue) & np.isfinite(pha).all(0)
    if product.noDataValue is not None:
        valid &= (pha != product.noDataValue).all(0)
    pha, unit = _phase_to_radians(pha, valid)
    slc = {pol: np.where(valid, amp[i] * np.exp(1j * pha[i]), 0).astype(np.complex64)
           for i, pol in enumerate(pols)}
    offset = (int(window.row_off), int(window.col_off)) if window is not None else (0, 0)
    print(f"  image {valid.shape[0]} x {valid.shape[1]} (azimuth x range), valid pixels {valid.mean():.1%}, "
          f"phase stored in {unit}")
    return slc, valid, {"gcps": gcps, "crs": gcp_crs, "offset": offset}


def pixel_spacing(scene):
    """(range, azimuth) pixel spacing in metres from the coregistration annotation."""
    values = _xml_values(str(scene.product.annotation_coregistered_xml_file))

    def number(tag):
        try:
            return float(values[tag].split()[0])
        except (KeyError, ValueError):
            return None

    dr = number("rangePixelSpacing")
    if dr is None and number("rangeTimeInterval"):
        dr = C / 2 * number("rangeTimeInterval")
    dx = number("azimuthPixelSpacing")
    if dx is None and number("azimuthTimeInterval"):
        dx = 6900.0 * number("azimuthTimeInterval")  # approx. BIOMASS ground speed
    return dr, dx


def multilook_window(scene, window_m, resolution=(25.0, 8.0)):
    """
    Odd (azimuth, range) window in pixels covering about window_m metres, the output step (half the
    window) and the approximate number of independent looks (resolution = slant range, azimuth in m).
    """
    dr, dx = pixel_spacing(scene)
    if not (dr and dx):
        print("  pixel spacing not in the annotation, using a 15 x 5 pixel window")
        size, looks = (15, 5), None
    else:
        size = (max(3, 2 * int(window_m / dx / 2) + 1), max(3, 2 * int(window_m / dr / 2) + 1))
        looks = (size[0] * dx / max(dx, resolution[1])) * (size[1] * dr / max(dr, resolution[0]))
    step = (max(1, size[0] // 2), max(1, size[1] // 2))
    print(f"  multilook window {size[0]} x {size[1]} pixels (~{window_m} m)"
          + (f", ~{looks:.0f} looks" if looks else ""))
    return size, step, looks


def _multilook(power, valid, size, step):
    """Boxcar mean of a real array over valid pixels only, subsampled by step; returns (mean, valid)."""
    sub = (slice(None, None, step[0]), slice(None, None, step[1]))
    weight = uniform_filter(valid.astype(np.float32), size=size)[sub]
    mean = uniform_filter(np.where(valid, power, 0).astype(np.float32), size=size)[sub] / np.maximum(weight, 1e-6)
    return mean, weight > 0.5


# =============================================================================
# Pauli
# =============================================================================

def _stretch_db(power, valid, lo=2, hi=98):
    db = 10 * np.log10(np.maximum(power, 1e-12))
    a, b = np.percentile(db[valid], [lo, hi])
    out = np.clip((db - a) / (b - a), 0, 1)
    out[~valid] = 0
    return out


def pauli_rgb(slc, valid, size, step):
    """Pauli RGB: R = |HH-VV|²/2 (double bounce), G = |HV+VH|²/2 (volume), B = |HH+VV|²/2 (surface)."""
    hh, hv, vh, vv = (slc[p] for p in POLS)
    channels = [np.abs(hh - vv) ** 2 / 2, np.abs(hv + vh) ** 2 / 2, np.abs(hh + vv) ** 2 / 2]
    rgb, ok = [], None
    for power in channels:
        mean, ok = _multilook(power, valid, size, step)
        rgb.append(_stretch_db(mean, ok))
    return np.dstack(rgb), ok


def _title(scene, analysis):
    primary = " (primary)" if scene.is_primary else ""
    return f"{analysis} – {scene.stack_label}\n{scene.date:%Y-%m-%d %H:%M}{primary}  ·  {scene.name}"


def plot_pauli(rgb, scene):
    fig, ax = plt.subplots(figsize=(8, 10), constrained_layout=True)
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    ax.set_title(_title(scene, "Pauli RGB") + "\nR = |HH−VV|   G = |HV+VH|   B = |HH+VV|", fontsize=8)
    ax.set_xlabel("range")
    ax.set_ylabel("azimuth")
    return fig


def _plot_maps(panels, scene, analysis):
    """Side-by-side maps: panels = [(title, image, colormap, vmin, vmax)]."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 9), constrained_layout=True)
    for ax, (title, img, cmap, vmin, vmax) in zip(np.atleast_1d(axes), panels):
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad(MASKED_COLOR)  # masked / no-data pixels in grey, never confused with data
        im = ax.imshow(img, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.02)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("range")
    np.atleast_1d(axes)[0].set_ylabel("azimuth")
    fig.suptitle(_title(scene, analysis), fontsize=9)
    return fig


# =============================================================================
# Cloude–Pottier H / A / alpha / beta
# =============================================================================

BETA_MIN_ALPHA = 15.0  # [deg] beta is undefined for pure surface scattering (alpha ~ 0)
A_MIN_H = 0.3  # anisotropy is only meaningful when the entropy is not low
MASKED_COLOR = "#b8b8b8"


def polarimetric_checks(slc, valid):
    """Prints reciprocity (HV vs VH) and the HH-VV phase difference, and warns if they look wrong."""
    sub = (slice(None, None, 4), slice(None, None, 4))
    v = valid[sub]
    hh, hv, vh, vv = (slc[p][sub][v].astype(np.complex128) for p in POLS)
    power = lambda x: np.mean(np.abs(x) ** 2)
    reciprocity = np.abs(np.mean(hv * np.conj(vh))) / np.sqrt(power(hv) * power(vh))
    ratio_db = 10 * np.log10(power(hv) / power(vh))
    copol = np.mean(hh * np.conj(vv))
    phase = np.degrees(np.angle(copol))
    print(f"  checks: |corr(HV,VH)| {reciprocity:.2f}, HV/VH {ratio_db:+.2f} dB, "
          f"HH-VV phase {phase:+.1f}°, |corr(HH,VV)| {np.abs(copol) / np.sqrt(power(hh) * power(vv)):.2f}")
    if reciprocity < 0.8 or abs(ratio_db) > 1:
        print("  WARNING: HV and VH differ more than expected for calibrated data")
    if abs(phase) > 90:
        print("  WARNING: HH-VV phase offset > 90°, check the calibration / sign convention")


def coherency_t3(slc, valid, size, step):
    """Multilooked 3x3 coherency matrix from the Pauli vector k = [HH+VV, HH-VV, 2HV] / sqrt(2)."""
    hh, vv = slc["HH"], slc["VV"]
    hv = 0.5 * (slc["HV"] + slc["VH"])  # reciprocity
    k = [(hh + vv) / np.sqrt(2), (hh - vv) / np.sqrt(2), np.sqrt(2) * hv]
    t3, ok = None, None
    for i in range(3):
        for j in range(i, 3):
            prod = k[i] * np.conj(k[j])
            re_part, ok = _multilook(prod.real, valid, size, step)
            if t3 is None:
                t3 = np.zeros(re_part.shape + (3, 3), dtype=np.complex64)
            if i == j:
                t3[..., i, i] = re_part
            else:
                im_part, _ = _multilook(prod.imag, valid, size, step)
                t3[..., i, j] = re_part + 1j * im_part
                t3[..., j, i] = np.conj(t3[..., i, j])
    return t3, ok


def h_a_alpha(t3, ok):
    """
    Eigendecomposition of T3 → span, entropy H, anisotropy A, mean alpha and mean beta [deg].
    Eigenvector u = [cos α, sin α cos β e^iδ, sin α sin β e^iγ]: α = arccos|u1|, β = atan(|u3| / |u2|).
    """
    lam, vec = np.linalg.eigh(t3.astype(np.complex128))
    lam = np.clip(lam[..., ::-1], 0, None)  # lambda1 >= lambda2 >= lambda3
    vec = vec[..., ::-1]
    span = lam.sum(-1)
    p = lam / np.maximum(span[..., None], 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(p > 0, p * np.log(p), 0), axis=-1) / np.log(3)
        anisotropy = (lam[..., 1] - lam[..., 2]) / (lam[..., 1] + lam[..., 2])
    alpha_i = np.degrees(np.arccos(np.clip(np.abs(vec[..., 0, :]), 0, 1)))
    beta_i = np.degrees(np.arctan2(np.abs(vec[..., 2, :]), np.abs(vec[..., 1, :])))
    alpha = np.sum(p * alpha_i, axis=-1)
    beta = np.sum(p * beta_i, axis=-1)
    beta[alpha < BETA_MIN_ALPHA] = np.nan
    beta1 = beta_i[..., 0].copy()  # beta of the dominant mechanism only
    beta1[alpha_i[..., 0] < BETA_MIN_ALPHA] = np.nan
    entropy = np.clip(entropy, 0, 1)
    anisotropy[entropy < A_MIN_H] = np.nan  # lambda2 and lambda3 are both tiny and noisy at low entropy
    result = {"span": span, "H": entropy, "A": anisotropy, "alpha": alpha, "beta": beta, "beta1": beta1}
    for key in result:
        result[key] = result[key].astype(np.float32)
        result[key][~ok] = np.nan
    return result


def plot_h_alpha(result, scene, show_beta=False):
    span_db = 10 * np.log10(result["span"])
    panels = [("span [dB]", span_db, "gray", np.nanpercentile(span_db, 2), np.nanpercentile(span_db, 98)),
              ("entropy H", result["H"], "viridis", 0, 1),
              (f"anisotropy A\n(grey: H < {A_MIN_H:g})", result["A"], "viridis", 0, 1),
              ("mean alpha [deg]", result["alpha"], "jet", 0, 90)]
    if show_beta:
        panels += [(f"mean beta [deg]\n(grey: alpha < {BETA_MIN_ALPHA:g}°)", result["beta"], "cividis", 0, 90),
                   (f"beta of dominant mechanism [deg]\n(grey: alpha1 < {BETA_MIN_ALPHA:g}°)", result["beta1"],
                    "cividis", 0, 90)]
    return _plot_maps(panels, scene, "Cloude–Pottier")


# =============================================================================
# Circular polarisation basis
# =============================================================================

PHASE_MIN_RHO = 0.3  # phases are shown only where the corresponding coherence is at least this


def circular_channels(slc):
    """
    Scattering matrix in the circular basis (Lee & Pottier 2009, backscatter alignment):
      S_RR = j S_HV + (S_HH - S_VV) / 2
      S_LL = j S_HV - (S_HH - S_VV) / 2
      S_RL = j (S_HH + S_VV) / 2
    Equivalent to S_circ = Uᵀ S U (not U S U^H, which is wrong for monostatic backscatter).
    """
    hh, vv = slc["HH"], slc["VV"]
    hv = 0.5 * (slc["HV"] + slc["VH"])  # reciprocity
    return {"RR": 1j * hv + 0.5 * (hh - vv), "LL": 1j * hv - 0.5 * (hh - vv), "RL": 0.5j * (hh + vv)}


def _check_circular_convention():
    """A flat surface must give only RL; a dihedral only RR and LL."""
    surface = circular_channels({"HH": np.array([1 + 0j]), "VV": np.array([1 + 0j]),
                                 "HV": np.array([0j]), "VH": np.array([0j])})
    dihedral = circular_channels({"HH": np.array([1 + 0j]), "VV": np.array([-1 + 0j]),
                                  "HV": np.array([0j]), "VH": np.array([0j])})
    assert np.allclose([surface["RR"], surface["LL"]], 0) and np.allclose(np.abs(surface["RL"]), 1)
    assert np.allclose(dihedral["RL"], 0) and np.allclose(np.abs([dihedral["RR"], dihedral["LL"]]), 1)


_check_circular_convention()


def _correlation(a, b, power_a, power_b, valid, size, step):
    """Multilooked complex correlation <a b*>: (coherence magnitude, phase [deg])."""
    product = a * np.conj(b)
    re, _ = _multilook(product.real, valid, size, step)
    im, _ = _multilook(product.imag, valid, size, step)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.hypot(re, im) / np.sqrt(power_a * power_b)
    return np.clip(rho, 0, 1), np.degrees(np.arctan2(im, re))


def circular_parameters(slc, valid, size, step):
    """
    Multilooked circular-basis parameters:
      CPR        = (<|S_RR|²> + <|S_LL|²>) / (2 <|S_RL|²>)     circular polarisation ratio
      rho_RRLL   = |<S_RR S_LL*>| / sqrt(<|S_RR|²> <|S_LL|²>)   RR-LL coherence
      phase_RRLL = arg<S_RR S_LL*>                             RR-LL phase (ice fabric / orientation)
      phase_HHVV = arg<S_HH S_VV*>                             co-polar phase difference (ice fabric)
      orientation = (phase_RRLL + π) / 4, wrapped to ±45°     tilt interpretation (Lee et al. 2000), GeoTIFF only
    Phases are set to NaN where their coherence is below PHASE_MIN_RHO.
    """
    c = circular_channels(slc)
    rr, ok = _multilook(np.abs(c["RR"]) ** 2, valid, size, step)
    ll, _ = _multilook(np.abs(c["LL"]) ** 2, valid, size, step)
    rl, _ = _multilook(np.abs(c["RL"]) ** 2, valid, size, step)
    hh, _ = _multilook(np.abs(slc["HH"]) ** 2, valid, size, step)
    vv, _ = _multilook(np.abs(slc["VV"]) ** 2, valid, size, step)
    with np.errstate(divide="ignore", invalid="ignore"):
        cpr = (rr + ll) / (2 * rl)
    rho_rrll, phase_rrll = _correlation(c["RR"], c["LL"], rr, ll, valid, size, step)
    rho_hhvv, phase_hhvv = _correlation(slc["HH"], slc["VV"], hh, vv, valid, size, step)

    theta = (np.radians(phase_rrll) + np.pi) / 4
    theta = np.degrees(np.where(theta > np.pi / 4, theta - np.pi / 2, theta))
    for phase, rho in ((phase_rrll, rho_rrll), (phase_hhvv, rho_hhvv), (theta, rho_rrll)):
        phase[~(rho >= PHASE_MIN_RHO)] = np.nan  # a phase is noise where the channels are hardly correlated

    result = {"CPR": cpr, "rho_RRLL": rho_rrll, "phase_RRLL": phase_rrll,
              "rho_HHVV": rho_hhvv, "phase_HHVV": phase_hhvv, "orientation": theta}
    for key in result:
        result[key] = result[key].astype(np.float32)
        result[key][~ok] = np.nan
    return result


def _phase_range(phase, setting="auto"):
    """Symmetric colour range ±limit [deg] for a phase map: a number, or "auto" (98th percentile of |phase|)."""
    if setting != "auto":
        return float(setting)
    values = np.abs(phase[np.isfinite(phase)])
    if values.size == 0:
        return 180.0
    target = float(np.percentile(values, 98))
    return next((v for v in (1, 2, 5, 10, 15, 20, 30, 45, 90, 180) if v >= target), 180)


def plot_circular(result, scene, hhvv_range="auto"):
    cpr_max = max(1.0, float(np.nanpercentile(result["CPR"], 98)))
    hhvv = result["phase_HHVV"]
    limit = _phase_range(hhvv, hhvv_range)
    stats = f"median {np.nanmedian(hhvv):+.1f}°, std {np.nanstd(hhvv):.1f}°" if np.isfinite(hhvv).any() else "no data"
    panels = [("circular polarisation ratio CPR", result["CPR"], "magma", 0, cpr_max),
              ("RR–LL coherence |ρ_RRLL|", result["rho_RRLL"], "viridis", 0, 1),
              (f"RR–LL phase arg⟨S_RR·S_LL*⟩ [deg]\n(grey: |ρ_RRLL| < {PHASE_MIN_RHO:g})", result["phase_RRLL"],
               "twilight", -180, 180),
              # diverging scale zoomed to ±limit: 0° (isotropic) is white, small deviations become visible
              (f"HH–VV phase arg⟨S_HH·S_VV*⟩ [deg], scale ±{limit:g}°\n({stats}; grey: |ρ_HHVV| < {PHASE_MIN_RHO:g})",
               hhvv, "RdBu_r", -limit, limit)]
    return _plot_maps(panels, scene, "Circular basis")


# =============================================================================
# Saving
# =============================================================================

def new_run_folder(output_dir, analysis, new_folder=True, label="", settings=None, scenes=None):
    """
    Where the results of this run are written.
    new_folder=True:  <output_dir>/<YYYY-MM-DD_HHMMSS>_<analysis>[_<label>]/, with a run_info.json that records
                      the settings and the scenes, so every run is kept and can be traced.
    new_folder=False: <output_dir>/ itself (results are overwritten or skipped, see SKIP_EXISTING).
    """
    base = Path(output_dir).expanduser()
    if not new_folder:
        base.mkdir(parents=True, exist_ok=True)
        print(f"Results go to {base}")
        return base
    now = datetime.now()
    clean_label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label)).strip("_") if label else ""
    run_dir = base / (f"{now:%Y-%m-%d_%H%M%S}_{analysis}" + (f"_{clean_label}" if clean_label else ""))
    run_dir.mkdir(parents=True, exist_ok=False)
    info = {
        "analysis": analysis,
        "created": now.isoformat(timespec="seconds"),
        "label": label,
        "settings": settings or {},
        "scenes": [{"stack": s.stack, "stack_id": s.stack_id, "date": f"{s.date:%Y-%m-%dT%H:%M:%S}",
                    "is_primary": s.is_primary, "input": str(s.path)} for s in (scenes or [])],
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"New run folder: {run_dir}")
    return run_dir


def output_path(scene, kind, output_dir, ext="png"):
    """
    Output file mirroring the input structure, so every result can be traced back to its product:
    <output_dir>/<stack folder>/L1C/<product name>/<product name>_<kind>.<ext>
    """
    folder = Path(output_dir).expanduser() / scene.path.relative_to(scene.data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{scene.name}_{kind}.{ext}"


def stack_output_folder(scene, output_dir):
    """<output_dir>/<stack folder> (same name as the input stack folder)."""
    parts = scene.path.relative_to(scene.data_dir).parts
    in_stack_folder = len(parts) > 1 and not parts[0].startswith("BIO_")
    folder = Path(output_dir).expanduser() / (parts[0] if in_stack_folder else scene.stack)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _record(scene, output_dir, analysis, files):
    """Adds the processed scene and its output files to <stack folder>/stack_info.json."""
    info_path = stack_output_folder(scene, output_dir) / "stack_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info.update({"stack_name": scene.stack, "stack_id": scene.stack_id, "data_dir": str(scene.data_dir)})
    if scene.is_primary:
        info["primary"] = scene.name
    entry = info.setdefault("scenes", {}).setdefault(scene.name, {})
    entry.update({"date": f"{scene.date:%Y-%m-%dT%H:%M:%S}", "is_primary": scene.is_primary, "input": str(scene.path)})
    entry.setdefault("outputs", {})[analysis] = {
        "files": [str(Path(f).relative_to(Path(output_dir).expanduser())) for f in files],
        "processed": datetime.now().isoformat(timespec="seconds"),
    }
    info_path.write_text(json.dumps(info, indent=2))


def save_figure(fig, path, show=True):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        display(fig)
    plt.close(fig)
    print(f"  saved {path.name}")
    return path


def save_geotiff(path, bands, geo, step, band_names=None, nodata=None):
    """
    Multiband GeoTIFF with the GCPs of the original image, rescaled to the multilooked grid,
    so it can be opened in QGIS/SNAP at the right location.
    """
    bands = np.asarray(bands)
    if bands.ndim == 2:
        bands = bands[None]
    profile = dict(driver="GTiff", height=bands.shape[1], width=bands.shape[2], count=bands.shape[0],
                   dtype=bands.dtype, compress="deflate", nodata=nodata)
    # the file is georeferenced with GCPs right after it is created; hide rasterio's warning in between
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(bands)
            for i, name in enumerate(band_names or [], start=1):
                dst.set_band_description(i, name)
        if geo["gcps"]:
            row0, col0 = geo["offset"]
            gcps = [GroundControlPoint(row=(g.row - row0) / step[0], col=(g.col - col0) / step[1],
                                       x=g.x, y=g.y, z=g.z) for g in geo["gcps"]]
            with rasterio.open(path, "r+") as dst:
                dst.gcps = (gcps, geo["crs"])
    print(f"  saved {path.name}")
    return path


# =============================================================================
# Running on many scenes
# =============================================================================

def _run(scenes, kind, process, output_dir, skip_existing, show=True):
    done, skipped, failed = 0, 0, []
    for i, scene in enumerate(scenes, start=1):
        print(f"\n[{i}/{len(scenes)}] {scene.label}")
        existing = output_path(scene, kind, output_dir)
        if skip_existing and existing.exists():
            print(f"  already done, showing the saved result from {datetime.fromtimestamp(existing.stat().st_mtime):%Y-%m-%d %H:%M}"
                  " (set SKIP_EXISTING = False to recompute)")
            if show:
                display(IPythonImage(filename=str(existing)))
            skipped += 1
            continue
        try:
            files = process(scene)
            _record(scene, output_dir, kind, files)
            done += 1
        except Exception as err:  # keep going with the other scenes
            print(f"  ERROR: {err}")
            failed.append(scene)
    print(f"\nFinished: {done} processed, {skipped} already done, {len(failed)} failed. "
          f"Results in {Path(output_dir).expanduser()}")
    return failed


def run_pauli(scenes, output_dir, window_m=50, geotiff=True, skip_existing=True, show=True):
    """Pauli RGB for every scene: PNG (and optionally GeoTIFF)."""
    def process(scene):
        slc, valid, geo = read_scene(scene)
        size, step, _ = multilook_window(scene, window_m)
        rgb, ok = pauli_rgb(slc, valid, size, step)
        del slc
        files = [save_figure(plot_pauli(rgb, scene), output_path(scene, "pauli", output_dir), show)]
        if geotiff:
            bands = np.moveaxis((rgb * 255).astype(np.uint8), -1, 0)
            files.append(save_geotiff(output_path(scene, "pauli", output_dir, "tif"), bands, geo, step,
                                      ["R |HH-VV|", "G |HV+VH|", "B |HH+VV|"], nodata=0))
        return files

    return _run(scenes, "pauli", process, output_dir, skip_existing, show)


def run_cloude_pottier(scenes, output_dir, window_m=100, geotiff=True, skip_existing=True, show=True,
                       show_beta=False):
    """Span, H, A, alpha (and optionally beta) maps for every scene: PNG (and optionally GeoTIFF with beta)."""
    def process(scene):
        slc, valid, geo = read_scene(scene)
        polarimetric_checks(slc, valid)
        size, step, looks = multilook_window(scene, window_m)
        if looks is not None and looks < 30:
            print("  WARNING: few looks, entropy will be biased low. Increase WINDOW_M.")
        t3, ok = coherency_t3(slc, valid, size, step)
        del slc
        result = h_a_alpha(t3, ok)
        print(f"  mean H {np.nanmean(result['H']):.2f}, A {np.nanmean(result['A']):.2f}, "
              f"alpha {np.nanmean(result['alpha']):.1f}°, beta {np.nanmean(result['beta']):.1f}° "
              f"({np.mean(np.isfinite(result['beta'])):.0%} of pixels with alpha ≥ {BETA_MIN_ALPHA:g}°)")
        files = [save_figure(plot_h_alpha(result, scene, show_beta), output_path(scene, "h_alpha", output_dir), show)]
        if geotiff:
            bands = np.stack([result[k] for k in ("span", "H", "A", "alpha", "beta", "beta1")])
            files.append(save_geotiff(output_path(scene, "h_alpha", output_dir, "tif"), bands, geo, step,
                                      ["span", "entropy H", "anisotropy A", "mean alpha [deg]", "mean beta [deg]",
                                       "beta dominant mechanism [deg]"], nodata=np.nan))
        return files

    return _run(scenes, "h_alpha", process, output_dir, skip_existing, show)


def run_circular(scenes, output_dir, window_m=100, geotiff=True, skip_existing=True, show=True, hhvv_range="auto"):
    """CPR, RR-LL coherence, RR-LL phase and HH-VV phase for every scene: PNG (and optionally GeoTIFF)."""
    def process(scene):
        slc, valid, geo = read_scene(scene)
        polarimetric_checks(slc, valid)
        size, step, _ = multilook_window(scene, window_m)
        result = circular_parameters(slc, valid, size, step)
        del slc
        print(f"  median CPR {np.nanmedian(result['CPR']):.2f}, mean |ρ_RRLL| {np.nanmean(result['rho_RRLL']):.2f}, "
              f"mean |ρ_HHVV| {np.nanmean(result['rho_HHVV']):.2f}; phases shown in "
              f"{np.mean(np.isfinite(result['phase_RRLL'])):.0%} (RR-LL) and "
              f"{np.mean(np.isfinite(result['phase_HHVV'])):.0%} (HH-VV) of the pixels")
        files = [save_figure(plot_circular(result, scene, hhvv_range), output_path(scene, "circular", output_dir),
                             show)]
        if geotiff:
            keys = ("CPR", "rho_RRLL", "phase_RRLL", "rho_HHVV", "phase_HHVV", "orientation")
            files.append(save_geotiff(output_path(scene, "circular", output_dir, "tif"),
                                      np.stack([result[k] for k in keys]), geo, step,
                                      ["CPR", "RR-LL coherence", "RR-LL phase [deg]", "HH-VV coherence",
                                       "HH-VV phase [deg]", "orientation angle [deg]"], nodata=np.nan))
        return files

    return _run(scenes, "circular", process, output_dir, skip_existing, show)


# =============================================================================
# Interferometry (step 7)
# =============================================================================

INSAR_MASKED_COLOR = "#f3d6d6"  # masked pixels in the grey coherence maps (pale red, never confused with data)
BIOMASS_WAVELENGTH = C / 435e6  # [m] P-band, ~0.69 m
STATS_MIN_COHERENCE = 0.3  # phase statistics only where the coherence is at least this


def read_band(scene, pol, window=None):
    """One polarisation as a complex SLC (phase converted from degrees to radians) and its validity mask."""
    product = scene.product
    with rasterio.open(product.measurement_abs_file) as a, rasterio.open(product.measurement_phase_file) as p:
        band = _band_pols(a).index(pol) + 1
        amp = a.read(band, window=window).astype(np.float32)
        pha = p.read(band, window=window).astype(np.float32)
    valid = np.isfinite(amp) & np.isfinite(pha) & (amp > 0)
    if product.noDataValue is not None:
        valid &= (amp != product.noDataValue) & (pha != product.noDataValue)
    pha, _ = _phase_to_radians(pha, valid)
    return np.where(valid, amp * np.exp(1j * pha), 0).astype(np.complex64), valid


def _gcps(scene):
    with rasterio.open(scene.product.measurement_abs_file) as src:
        gcps, crs = src.gcps
    return {"gcps": gcps, "crs": crs, "offset": (0, 0)}


def phase_processing_flags(scene):
    """InSAR processing flags from the coregistration annotation (what has already been applied to the data)."""
    values = _xml_values(str(scene.product.annotation_coregistered_xml_file))
    keys = ("skpPhaseCalibrationFlag", "skpPhaseCorrectionFlag", "skpPhaseCorrectionFlatteningOnlyFlag")
    return {k: values.get(k, "not in annotation") for k in keys}


def _lut_array(scene, keywords):
    """
    A LUT layer on its own coarse grid as (array, name), or (None, None): the first 2D layer whose name
    contains all keywords. Turned to (azimuth, range) like the image if its dimensions say (range, azimuth).
    """
    lut = scene.product.lut_variables
    name = next((k for k in lut if all(w in k.lower() for w in keywords) and "quality" not in k.lower()
                 and np.ndim(lut[k].values) >= 2), None)
    if name is None:
        return None, None
    arr = np.asarray(lut[name].values, dtype=np.float64)
    dims = [str(d).lower() for d in lut[name].dims][-2:]
    arr = arr[0] if arr.ndim == 3 else arr
    if len(dims) == 2 and "range" in dims[0] and "azimuth" in dims[1]:
        arr = arr.T
    return arr, name


def _lut_layer(scene, keywords, shape):
    """A LUT layer (first name containing all keywords) interpolated to the image grid (LUT assumed to span it)."""
    arr, name = _lut_array(scene, keywords)
    if arr is None:
        raise ValueError(f"No LUT layer containing {keywords}. Available: {', '.join(scene.product.lut_variables)}")
    arr = np.where(np.isfinite(arr), arr, np.nanmean(arr))
    rows = np.linspace(0, arr.shape[0] - 1, shape[0])
    cols = np.linspace(0, arr.shape[1] - 1, shape[1])
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return map_coordinates(arr, [rr, cc], order=1, mode="nearest").astype(np.float32), name


def phase_correction(scene, shape, mode):
    """
    Phase screen to apply to a scene (course convention: SLC · exp(j · (φ_flattening − φ_calibration))).
    mode: "none", "calibration", "flattening" or "flattening+calibration". Experimental: compare with "none".
    """
    if mode == "none":
        return None
    if mode not in ("calibration", "flattening", "flattening+calibration"):
        raise ValueError(f"Unknown PHASE_CORRECTION '{mode}'")
    correction = 0
    if "flattening" in mode:
        flat, flat_name = _lut_layer(scene, ("flattening",), shape)
        print(f"    flattening phase from {flat_name}")
        correction = flat
    if "calibration" in mode:
        cal, cal_name = _lut_layer(scene, ("skp", "screen"), shape)
        print(f"    calibration phase from {cal_name}")
        correction = correction - cal
    return correction


def complex_coherence(s1, v1, s2, v2, size, step):
    """
    Multilooked complex coherence γ = <s1 s2*> / sqrt(<|s1|²> <|s2|²>), boxcar of `size` pixels,
    subsampled by `step` (half the window). |γ| = coherence, arg(γ) = interferogram.
    """
    valid = v1 & v2
    product = s1 * np.conj(s2)
    re, ok = _multilook(product.real, valid, size, step)
    im, _ = _multilook(product.imag, valid, size, step)
    p1, _ = _multilook(np.abs(s1) ** 2, valid, size, step)
    p2, _ = _multilook(np.abs(s2) ** 2, valid, size, step)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = (re + 1j * im) / np.sqrt(p1 * p2)
    gamma[~ok] = np.nan
    return gamma.astype(np.complex64)


def phase_to_mm(phase):
    """Interferometric phase [rad] as line-of-sight displacement [mm] (d = φ·λ / 4π)."""
    return phase * BIOMASS_WAVELENGTH / (4 * np.pi) * 1e3


def phase_statistics(gamma, min_coherence=STATS_MIN_COHERENCE):
    """
    Statistics of the interferogram arg(γ) [rad] where |γ| >= min_coherence: circular mean and standard deviation,
    2nd and 98th percentile (around 0), and the fraction of pixels used.
    """
    finite = np.isfinite(gamma)
    ok = finite & (np.abs(gamma) >= min_coherence)
    if not ok.any():
        return {"phase_mean": np.nan, "phase_std": np.nan, "phase_p2": np.nan, "phase_p98": np.nan, "phase_used": 0.0}
    resultant = np.mean(gamma[ok] / np.abs(gamma[ok]))
    p2, p98 = np.percentile(np.angle(gamma[ok]), [2, 98])
    return {"phase_mean": float(np.angle(resultant)),
            "phase_std": float(np.sqrt(-2 * np.log(max(np.abs(resultant), 1e-12)))),
            "phase_p2": float(p2), "phase_p98": float(p98),
            "phase_used": float(ok.sum() / max(finite.sum(), 1))}


def _format_phase_stats(stats):
    if not np.isfinite(stats["phase_mean"]):
        return "phase: no pixels with enough coherence"
    return (f"phase mean {stats['phase_mean']:+.3f} rad ({phase_to_mm(stats['phase_mean']):+.1f} mm), "
            f"std {stats['phase_std']:.3f} rad, 2–98 % {stats['phase_p2']:+.3f}…{stats['phase_p98']:+.3f} rad")


def _interferogram_limit(pairs, setting="full"):
    """Colour range ±limit [rad]: "full" (±π), "auto" (98th percentile of |phase| over all pairs) or a number."""
    if setting in (None, "full"):
        return np.pi
    if setting != "auto":
        return min(abs(float(setting)), np.pi)
    values = np.concatenate([np.abs(np.angle(g[np.isfinite(g)])) for _, g in pairs])
    if values.size == 0:
        return np.pi
    target = float(np.percentile(values, 98))
    return next((v for v in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0) if v >= target), np.pi)


def plot_interferometry(primary, pairs, pol, window_m, phase_range="full"):
    """
    One figure per polarisation: a column per pair, interferogram on top and coherence below.
    phase_range: "full" (±π, cyclic colours), "auto" (zoomed to the data, blue–white–red) or a number in rad.
    """
    n = len(pairs)
    limit = _interferogram_limit(pairs, phase_range)
    full = limit >= np.pi
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n + 1.8, 13), constrained_layout=True, squeeze=False)
    phase_cmap = plt.get_cmap("hsv" if full else "RdBu_r").copy()
    phase_cmap.set_bad(MASKED_COLOR)
    coh_cmap = plt.get_cmap("gray").copy()
    coh_cmap.set_bad(INSAR_MASKED_COLOR)
    for k, (secondary, gamma) in enumerate(pairs):
        dt = (secondary.date - primary.date).days
        im_phase = axes[0, k].imshow(np.angle(gamma), cmap=phase_cmap, vmin=-limit, vmax=limit,
                                     aspect="auto", interpolation="nearest")
        axes[0, k].set_title(f"{secondary.date:%Y-%m-%d}\nΔt = {dt:+d} days", fontsize=9)
        im_coh = axes[1, k].imshow(np.abs(gamma), cmap=coh_cmap, vmin=0, vmax=1, aspect="auto",
                                   interpolation="nearest")
        axes[1, k].set_title(f"mean |γ| = {np.nanmean(np.abs(gamma)):.2f}", fontsize=9)
        for ax in axes[:, k]:
            ax.set_xlabel("range", fontsize=8)
            ax.tick_params(labelsize=7)
            if k:
                ax.set_yticklabels([])
    axes[0, 0].set_ylabel("interferogram\nazimuth")
    axes[1, 0].set_ylabel("coherence\nazimuth")
    if full:
        cbar = fig.colorbar(im_phase, ax=axes[0, :], shrink=0.8, pad=0.01,
                            ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cbar.ax.set_yticklabels(["−π", "−π/2", "0", "π/2", "π"])
        cbar.set_label("interferometric phase [rad]")
    else:
        cbar = fig.colorbar(im_phase, ax=axes[0, :], shrink=0.8, pad=0.01)
        cbar.set_label(f"interferometric phase [rad], zoomed scale ±{limit:g} rad (≈ ±{phase_to_mm(limit):.1f} mm)")
    fig.colorbar(im_coh, ax=axes[1, :], shrink=0.8, pad=0.01).set_label("coherence |γ|")
    fig.suptitle(f"Interferometry {pol} – {primary.stack_label}\n"
                 f"primary {primary.date:%Y-%m-%d %H:%M}  ·  {n} pair(s)  ·  window ~{window_m:g} m",
                 fontsize=10)
    return fig


def _choose_secondaries(scenes, primary, which):
    others = [s for s in scenes if s is not primary]
    if _is_all(which):
        return others
    chosen = []
    for item in _as_list(which):
        if isinstance(item, int):
            if 0 <= item < len(scenes) and scenes[item] is not primary:
                chosen.append(scenes[item])
            else:
                print(f"  Warning: scene number {item} is not a secondary of this stack, skipped")
        else:
            digits = re.sub(r"\D", "", str(item))
            chosen += [s for s in others if f"{s.date:%Y%m%d%H%M%S}".startswith(digits)]
    return list(dict.fromkeys(chosen))


def run_interferometry(stacks, output_dir, which_stacks="all", secondaries="all", pols=POLS, window_m=300,
                       correction="flattening", geotiff=True, show=True, phase_range="full"):
    """
    For every chosen stack: interferogram and coherence of every primary–secondary pair, for every polarisation.
    One figure per polarisation and stack; optionally a 2-band GeoTIFF (coherence, phase) per pair and polarisation.
    Coherence and phase statistics of every pair are printed and saved in a CSV.
    """
    output_dir = Path(output_dir).expanduser()
    for name in _choose_stacks(stacks, which_stacks):
        scenes = stacks[name]
        primary = find_primary(scenes)
        print(f"\n===== {scenes[0].stack_label} =====")
        if primary is None:
            print("  no primary found, skipped")
            continue
        pairs_scenes = sorted(_choose_secondaries(scenes, primary, secondaries), key=lambda s: s.date)
        if not pairs_scenes:
            print("  no secondaries selected, skipped")
            continue
        print(f"  primary: {primary.date:%Y-%m-%d %H:%M}  {primary.name}")
        print(f"  secondaries: {', '.join(f'{s.date:%Y-%m-%d}' for s in pairs_scenes)}")
        print("  processing flags:", ", ".join(f"{k} = {v}" for k, v in phase_processing_flags(primary).items()))
        size, step, _ = multilook_window(primary, window_m)
        folder = stack_output_folder(primary, output_dir) / "interferometry"
        folder.mkdir(parents=True, exist_ok=True)
        geo = _gcps(primary)
        files, rows = [], []
        corrections = {}

        for pol in pols:
            print(f"\n  {pol}")
            s1, v1 = read_band(primary, pol)
            if correction != "none":
                corrections.setdefault(primary.name, phase_correction(primary, s1.shape, correction))
                s1 = s1 * np.exp(1j * corrections[primary.name])
            pairs = []
            for sec in pairs_scenes:
                try:
                    s2, v2 = read_band(sec, pol)
                    if correction != "none":
                        corrections.setdefault(sec.name, phase_correction(sec, s2.shape, correction))
                        s2 = s2 * np.exp(1j * corrections[sec.name])
                    gamma = complex_coherence(s1, v1, s2, v2, size, step)
                    del s2, v2
                except Exception as err:  # keep going with the other pairs
                    print(f"    {sec.date:%Y-%m-%d}: ERROR {err}")
                    continue
                coh = np.abs(gamma)
                dt = (sec.date - primary.date).days
                stats = phase_statistics(gamma)
                print(f"    {primary.date:%Y-%m-%d} × {sec.date:%Y-%m-%d} (Δt {dt:+d} d): "
                      f"mean |γ| {np.nanmean(coh):.2f}, median {np.nanmedian(coh):.2f}\n"
                      f"      {_format_phase_stats(stats)}")
                rows.append({"pol": pol, "primary": primary.name, "secondary": sec.name,
                             "secondary_date": f"{sec.date:%Y-%m-%d}", "dt_days": dt,
                             "mean_coherence": round(float(np.nanmean(coh)), 4),
                             "median_coherence": round(float(np.nanmedian(coh)), 4),
                             "phase_mean_rad": round(stats["phase_mean"], 4),
                             "phase_mean_mm": round(float(phase_to_mm(stats["phase_mean"])), 2),
                             "phase_std_rad": round(stats["phase_std"], 4),
                             "phase_p2_rad": round(stats["phase_p2"], 4),
                             "phase_p98_rad": round(stats["phase_p98"], 4),
                             "phase_pixels_used": round(stats["phase_used"], 3)})
                pairs.append((sec, gamma))
                if geotiff:
                    tif = folder / f"{primary.date:%Y%m%dT%H%M%S}_{sec.date:%Y%m%dT%H%M%S}_{pol}_insar.tif"
                    files.append(save_geotiff(tif, np.stack([coh, np.degrees(np.angle(gamma))]).astype(np.float32),
                                              geo, step, ["coherence", "interferometric phase [deg]"], nodata=np.nan))
            del s1, v1
            if pairs:
                png = folder / f"{name}_{pol}_interferometry.png"
                files.append(save_figure(plot_interferometry(primary, pairs, pol, window_m, phase_range), png, show))

        if rows:
            summary = folder / f"{name}_coherence_summary.csv"
            lines = [",".join(rows[0])] + [",".join(str(r[k]) for k in rows[0]) for r in rows]
            summary.write_text("\n".join(lines) + "\n")
            files.append(summary)
            _record(primary, output_dir, "interferometry", files)
            print(f"\n  summary: {summary}")
    print(f"\nFinished. Results in {output_dir}")


# =============================================================================
# Interferometry diagnostics
# =============================================================================

def processing_flags(scene):
    """Every <...Flag> in the XML files of a product, as {file name: {tag: value}} (all values if they differ)."""
    flags = {}
    for xml in sorted(scene.path.rglob("*.xml")):
        if ".ipynb_checkpoints" in xml.parts:
            continue
        found = {}
        try:
            for el in ET.parse(xml).iter():
                tag = el.tag.split("}")[-1]
                if tag.lower().endswith("flag") and el.text and el.text.strip():
                    found.setdefault(tag, []).append(el.text.strip())
        except ET.ParseError:
            continue
        if found:
            flags[xml.name] = {tag: " / ".join(dict.fromkeys(values)) for tag, values in found.items()}
    return flags


def print_processing_flags(scene, compare=()):
    """Prints all processing flags of a product, and the flags that differ in the products in `compare`."""
    flags = processing_flags(scene)
    print(f"  Processing flags of {scene.name}:")
    if not flags:
        print("    no <...Flag> tags found in the XML files")
    for xml_name, values in flags.items():
        print(f"    {xml_name}")
        for tag, value in values.items():
            print(f"      {tag:45s} {value}")
    simple = {tag: value for values in flags.values() for tag, value in values.items()}
    for other in compare:
        other_flags = {tag: value for values in processing_flags(other).values() for tag, value in values.items()}
        diff = {t: (simple.get(t), other_flags.get(t)) for t in set(simple) | set(other_flags)
                if simple.get(t) != other_flags.get(t)}
        if diff:
            print(f"  Different in {other.date:%Y-%m-%d}: "
                  + ", ".join(f"{t} = {b} (here {a})" for t, (a, b) in sorted(diff.items())))
    if compare:
        print(f"  ({len(compare)} other product(s) compared)")


def print_lut_layers(scene):
    """Lists the 2D layers of the LUT of a product with their size and value range."""
    print(f"  LUT layers of {scene.name}:")
    for name, var in scene.product.lut_variables.items():
        arr = np.asarray(var.values)
        if arr.ndim < 2 or arr.dtype.kind not in "fiu":
            continue
        finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr.ravel()
        value_range = f"{finite.min():+.4g} … {finite.max():+.4g}" if finite.size else "no data"
        print(f"    {name:55s} {str(arr.shape):16s} {value_range}")


def plot_lut_phases(scenes, primary):
    """
    LUT phases of every acquisition on the coarse LUT grid: the flattening phase (wrapped, cyclic colours)
    and the SKP calibration phase screen (common blue–white–red scale).
    """
    n = len(scenes)
    flat = [_lut_array(s, ("flattening",)) for s in scenes]
    skp = [_lut_array(s, ("skp", "screen")) for s in scenes]
    skp_values = [np.abs(a[np.isfinite(a)]) for a, _ in skp if a is not None]
    skp_limit = float(np.percentile(np.concatenate(skp_values), 98)) if skp_values else 1.0
    skp_limit = max(skp_limit, 1e-3)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n + 1.8, 10), constrained_layout=True, squeeze=False)
    cyclic = plt.get_cmap("hsv").copy()
    cyclic.set_bad(MASKED_COLOR)
    diverging = plt.get_cmap("RdBu_r").copy()
    diverging.set_bad(MASKED_COLOR)
    im_flat = im_skp = None
    for k, scene in enumerate(scenes):
        mark = " ★" if scene is primary else ""
        for row, ((arr, name), cmap, limit) in enumerate([(flat[k], cyclic, np.pi), (skp[k], diverging, skp_limit)]):
            ax = axes[row, k]
            ax.tick_params(labelsize=7)
            if arr is None:
                ax.text(0.5, 0.5, "not in LUT", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            shown = np.angle(np.exp(1j * arr)) if row == 0 else arr
            im = ax.imshow(shown, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
            if row == 0:
                im_flat = im
                ax.set_title(f"{scene.date:%Y-%m-%d}{mark}\nrange {np.nanmin(arr):+.0f}…{np.nanmax(arr):+.0f} rad",
                             fontsize=8)
            else:
                im_skp = im
                ax.set_title(f"std {np.nanstd(arr):.3f} rad", fontsize=8)
    axes[0, 0].set_ylabel("flattening phase (wrapped)\nazimuth (LUT grid)")
    axes[1, 0].set_ylabel("SKP calibration screen\nazimuth (LUT grid)")
    if im_flat is not None:
        fig.colorbar(im_flat, ax=axes[0, :], shrink=0.8, pad=0.01).set_label("flattening phase [rad]")
    if im_skp is not None:
        fig.colorbar(im_skp, ax=axes[1, :], shrink=0.8, pad=0.01).set_label(
            f"SKP phase [rad], scale ±{skp_limit:.3g} (≈ ±{phase_to_mm(skp_limit):.1f} mm)")
    names = sorted({name for _, name in flat + skp if name})
    fig.suptitle(f"LUT phases – {primary.stack_label}   (★ = primary)\n" + ", ".join(names), fontsize=9)
    return fig


def _preview_file(scene, kind):
    folder = scene.path / "preview"
    return _first_file(folder, f"*{kind}*ql*.png") if folder.is_dir() else None


def plot_esa_previews(scenes, primary):
    """ESA's quicklooks of the interferogram (*itf*ql.png) and coherence (*coh*ql.png) of every product, or None."""
    found = [(s, _preview_file(s, "itf"), _preview_file(s, "coh")) for s in scenes]
    found = [f for f in found if f[1] or f[2]]
    if not found:
        for s in scenes[:1]:
            folder = s.path / "preview"
            files = sorted(p.name for p in folder.glob("*")) if folder.is_dir() else []
            print(f"  no *itf*ql.png / *coh*ql.png quicklooks; preview folder of {s.date:%Y-%m-%d} contains: "
                  + (", ".join(files) or "nothing"))
        return None
    n = len(found)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n + 1, 10), constrained_layout=True, squeeze=False)
    for k, (scene, itf, coh) in enumerate(found):
        mark = " ★" if scene is primary else ""
        for row, (path, label) in enumerate([(itf, "interferogram"), (coh, "coherence")]):
            ax = axes[row, k]
            ax.set_xticks([])
            ax.set_yticks([])
            if path is None:
                ax.text(0.5, 0.5, "no quicklook", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.imshow(plt.imread(str(path)), aspect="auto", interpolation="nearest")
            if row == 0:
                ax.set_title(f"{scene.date:%Y-%m-%d}{mark}", fontsize=8)
            if k == 0:
                ax.set_ylabel(f"ESA {label} quicklook")
    fig.suptitle(f"ESA quicklooks – {primary.stack_label}   (★ = primary)\n"
                 "orientation and colour scale as delivered by ESA, compare the patterns with step 7", fontsize=9)
    return fig


def run_insar_diagnostics(stacks, output_dir, which_stacks="all", show=True):
    """
    Checks behind a flat interferogram, per stack: all processing flags, the LUT layers, the flattening phase
    and SKP calibration screen of every acquisition, and ESA's own interferogram/coherence quicklooks.
    """
    output_dir = Path(output_dir).expanduser()
    for name in _choose_stacks(stacks, which_stacks):
        scenes = sorted(stacks[name], key=lambda s: s.date)
        primary = find_primary(scenes)
        print(f"\n===== {scenes[0].stack_label} =====")
        if primary is None:
            print("  no primary found, skipped")
            continue
        secondaries = [s for s in scenes if s is not primary]
        folder = stack_output_folder(primary, output_dir) / "insar_diagnostics"
        folder.mkdir(parents=True, exist_ok=True)
        files = []

        print(f"\n1. Processing flags (primary {primary.date:%Y-%m-%d}, compared with the secondaries)")
        print_processing_flags(primary, compare=secondaries)

        print("\n2. LUT layers" + (f" (secondary {secondaries[0].date:%Y-%m-%d})" if secondaries else ""))
        print_lut_layers(secondaries[0] if secondaries else primary)

        print("\n3. Flattening phase and SKP calibration screen from the LUT")
        try:
            files.append(save_figure(plot_lut_phases(scenes, primary), folder / f"{name}_lut_phases.png", show))
        except Exception as err:
            print(f"  ERROR: {err}")

        print("\n4. ESA quicklooks")
        fig = plot_esa_previews(scenes, primary)
        if fig is not None:
            files.append(save_figure(fig, folder / f"{name}_esa_quicklooks.png", show))

        if files:
            _record(primary, output_dir, "insar_diagnostics", files)
    print(f"\nFinished. Results in {output_dir}")


# =============================================================================
# AOI maps
# =============================================================================

NATURAL_EARTH_VECTOR_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/{}.geojson"
NATURAL_EARTH_RELIEF_URL = "https://naciscdn.org/naturalearth/50m/raster/GRAY_50M_SR_OB.zip"
NATURAL_EARTH_DIR = Path(__file__).resolve().parent / "natural_earth"

MAP_STYLE = {
    "ocean": "#d6e3ee",
    "land": "#fbfbf9",
    "ice_shelf": "#e2edf6",
    "coast": dict(color="#4f6073", linewidth=0.35),
    "ice_shelf_edge": dict(color="#8aa4bb", linewidth=0.25),
    "grid": dict(color="#7d8b99", linewidth=0.3, linestyle=(0, (2, 3)), alpha=0.7),
    "text": "#1f2933",
    "muted": "#5f6b77",
    "frame": "#3d4852",
    "highlight": "#c0392b",
    "primary": "#f2b705",
}
STACK_COLORS = ["#d1495b", "#00798c", "#edae49", "#66a182", "#2e4057", "#8d5a97", "#e07a5f", "#3d405b"]


def footprint(scene):
    """Scene footprint as closed (lon, lat) arrays, from the annotation (sarImage/footprint) or the LUT."""
    product = scene.product
    text = _xml_values(str(product.annotation_coregistered_xml_file)).get("footprint")
    if text:
        values = np.array(text.replace(",", " ").split(), dtype=float)
        lat, lon = values[0::2], values[1::2]
        if np.abs(lat).max() > 90:  # stored as lon, lat
            lat, lon = lon, lat
    else:
        lat = np.asarray(product.lut_variables["geometry_latitude"].values)
        lon = np.asarray(product.lut_variables["geometry_longitude"].values)
        corners = [(0, 0), (0, -1), (-1, -1), (-1, 0)]
        lat = np.array([lat[c] for c in corners])
        lon = np.array([lon[c] for c in corners])
    return np.append(lon, lon[0]), np.append(lat, lat[0])


def choose_crs(lons, lats):
    """Map projection for the data: Antarctic / Arctic polar stereographic, otherwise UTM."""
    lat, lon = float(np.mean(lats)), float(np.mean(lons))
    if lat < -60:
        return "EPSG:3031"
    if lat > 60:
        return "EPSG:3413"
    zone = int((lon + 180) // 6) % 60 + 1
    return f"EPSG:{(32700 if lat < 0 else 32600) + zone}"


def crs_name(crs):
    names = {"EPSG:3031": "WGS 84 / Antarctic Polar Stereographic",
             "EPSG:3413": "WGS 84 / NSIDC Sea Ice Polar Stereographic North",
             "EPSG:4326": "WGS 84"}
    if crs in names:
        return f"{names[crs]} ({crs})"
    code = int(crs.split(":")[1])
    return f"WGS 84 / UTM zone {code % 100}{'S' if code > 32700 else 'N'} ({crs})"


def _overview_crs(lons, lats):
    """Polar projection if all scenes are polar, a regional one if they are close, else a world map."""
    if np.max(lats) < -60:
        return "EPSG:3031"
    if np.min(lats) > 60:
        return "EPSG:3413"
    if np.ptp(lats) < 20 and np.ptp(lons) < 20:
        return choose_crs(lons, lats)
    return "EPSG:4326"


def _to_map(lon, lat, crs):
    x, y = warp_transform("EPSG:4326", crs, np.asarray(lon).tolist(), np.asarray(lat).tolist())
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    bad = ~(np.isfinite(x) & np.isfinite(y))
    x[bad], y[bad] = np.nan, np.nan
    return x, y


def _region(crs, lons, lats, margin_deg):
    """Lon/lat box used to skip background polygons far outside the map."""
    if crs == "EPSG:4326":
        return (-180, 180, -90, 90)
    if crs == "EPSG:3031":
        return (-180, 180, -90, min(-50, np.max(lats) + margin_deg))
    if crs == "EPSG:3413":
        return (-180, 180, max(50, np.min(lats) - margin_deg), 90)
    return (np.min(lons) - margin_deg, np.max(lons) + margin_deg, np.min(lats) - margin_deg, np.max(lats) + margin_deg)


def _extent(xy_list, margin=0.4, minimum=3e4):
    """Square map extent around the footprints."""
    xs = np.concatenate([xy[0] for xy in xy_list])
    ys = np.concatenate([xy[1] for xy in xy_list])
    cx, cy = (np.nanmin(xs) + np.nanmax(xs)) / 2, (np.nanmin(ys) + np.nanmax(ys)) / 2
    half = max(np.nanmax(xs) - np.nanmin(xs), np.nanmax(ys) - np.nanmin(ys), minimum) * (0.5 + margin)
    return (cx - half, cx + half, cy - half, cy + half)


def _continent_extent(crs, xy_list):
    if crs == "EPSG:3031":
        return (-3.3e6, 3.3e6, -3.3e6, 3.3e6)
    if crs == "EPSG:3413":
        return (-3.5e6, 3.5e6, -4.0e6, 3.0e6)
    if crs == "EPSG:4326":
        return (-180, 180, -90, 90)
    return _extent(xy_list, margin=2.0, minimum=5e5)


# ---------- basemap data (Natural Earth, public domain, downloaded once) ----------

@lru_cache(maxsize=None)
def _natural_earth(layer):
    """Polygons (list of rings in lon/lat) of a Natural Earth vector layer."""
    path = NATURAL_EARTH_DIR / f"{layer}.geojson"
    if not path.exists():
        try:
            NATURAL_EARTH_DIR.mkdir(exist_ok=True)
            print(f"Downloading map data {layer} (only the first time)...")
            urllib.request.urlretrieve(NATURAL_EARTH_VECTOR_URL.format(layer), path)
        except Exception as err:
            path.unlink(missing_ok=True)
            print(f"  could not download {layer} ({err}); the map is drawn without it")
            return []
    polygons = []
    for feature in json.loads(path.read_text()).get("features", []):
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates", [])
        if geom.get("type") == "Polygon":
            coords = [coords]
        elif geom.get("type") != "MultiPolygon":
            continue
        for poly in coords:
            polygons.append([np.asarray(ring, dtype=float)[:, :2] for ring in poly])
    return polygons


@lru_cache(maxsize=1)
def _relief_source():
    """Natural Earth grey shaded relief with ocean bottom (raster), or None if it cannot be obtained."""
    folder = NATURAL_EARTH_DIR / "GRAY_50M_SR_OB"
    tif = next(folder.rglob("*.tif"), None) if folder.exists() else None
    if tif is None:
        zip_path = NATURAL_EARTH_DIR / "GRAY_50M_SR_OB.zip"
        try:
            NATURAL_EARTH_DIR.mkdir(exist_ok=True)
            print("Downloading shaded relief (~25 MB, only the first time)...")
            urllib.request.urlretrieve(NATURAL_EARTH_RELIEF_URL, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(folder)
            zip_path.unlink()
            tif = next(folder.rglob("*.tif"), None)
        except Exception as err:
            zip_path.unlink(missing_ok=True)
            print(f"  could not download the shaded relief ({err}); using a simple gradient instead")
            return None
    if tif is None:
        return None
    with rasterio.open(tif) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        if transform.is_identity:  # no georeferencing found: the file covers the whole world
            transform = from_bounds(-180, -90, 180, 90, src.width, src.height)
    return data, transform


def _layer_geometry(layer, crs, region):
    """A Natural Earth layer projected to the map: (fill path, outline x, outline y), or None."""
    polygons = [rings for rings in _natural_earth(layer)
                if not (rings[0][:, 0].max() < region[0] or rings[0][:, 0].min() > region[1]
                        or rings[0][:, 1].max() < region[2] or rings[0][:, 1].min() > region[3])]
    if not polygons:
        return None
    rings = [r for poly in polygons for r in poly]
    lon = np.concatenate([r[:, 0] for r in rings])
    lat = np.concatenate([r[:, 1] for r in rings])
    x, y = _to_map(lon, lat, crs)

    lengths = [len(r) for r in rings]
    codes = np.full(len(x), MplPath.LINETO)
    codes[np.cumsum([0] + lengths[:-1])] = MplPath.MOVETO
    path = MplPath(np.column_stack([x, y]), codes)

    # outline without the artificial edges where the data are cut at 180° and at the poles
    cut = ((np.abs(lon[:-1]) > 179.99) & (np.abs(lon[1:]) > 179.99)) | \
          ((np.abs(lat[:-1]) > 89.9) & (np.abs(lat[1:]) > 89.9))
    cut[np.cumsum(lengths)[:-1] - 1] = True  # do not connect consecutive rings
    seg_x = np.column_stack([x[:-1], x[1:], np.full(len(x) - 1, np.nan)])
    seg_y = np.column_stack([y[:-1], y[1:], np.full(len(y) - 1, np.nan)])
    seg_x[cut], seg_y[cut] = np.nan, np.nan
    return path, seg_x.ravel(), seg_y.ravel()


RELIEF_MIN_WIDTH = 1.5e6  # [m] the 50m relief is only sharp enough for continent-scale maps


def _draw_relief(ax, crs, extent, size, clip_patch):
    """Shaded relief as a soft overlay on land only (only the shadows darken the map)."""
    source = _relief_source()
    if source is None or clip_patch is None:
        return False
    data, transform = source
    xmin, xmax, ymin, ymax = extent
    shade = np.full((size, size), np.nan, dtype=np.float32)
    reproject(data, shade, src_transform=transform, src_crs="EPSG:4326",
              dst_transform=from_bounds(xmin, ymin, xmax, ymax, size, size), dst_crs=crs,
              resampling=Resampling.bilinear, dst_nodata=np.nan)
    ok = np.isfinite(shade)
    if ok.sum() < 100:
        return False
    lo, hi = np.percentile(shade[ok], [2, 98])
    shade = np.clip((shade - lo) / max(hi - lo, 1e-6), 0, 1)
    rgba = np.zeros((size, size, 4), dtype=np.float32)
    rgba[..., :3] = (0.12, 0.20, 0.30)
    rgba[..., 3] = np.where(ok, (1 - shade) * 0.25, 0)
    im = ax.imshow(rgba, extent=(xmin, xmax, ymin, ymax), origin="upper", interpolation="bilinear", zorder=1.2)
    im.set_clip_path(clip_patch)
    return True


def _draw_gradient(ax, crs, extent, land_patch, size=400):
    """Fallback without relief: white-to-blue-grey gradient on land, from the interior to the coast."""
    if land_patch is None:
        return
    xmin, xmax, ymin, ymax = extent
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, size), np.linspace(ymax, ymin, size))
    polar_view = crs in ("EPSG:3031", "EPSG:3413") and xmin < 0 < xmax and ymin < 0 < ymax
    cx, cy = (0.0, 0.0) if polar_view else ((xmin + xmax) / 2, (ymin + ymax) / 2)
    d = np.hypot(gx - cx, gy - cy)
    d = (d / d.max())[..., None]
    land = (1 - d) * np.array([1.0, 1.0, 1.0]) + d * np.array([0.88, 0.92, 0.95])
    im = ax.imshow(land, extent=extent, origin="upper", zorder=1.2)
    im.set_clip_path(land_patch)


def _draw_background(ax, crs, region, extent, detail, relief_size):
    ax.set_facecolor(MAP_STYLE["ocean"])
    land = _layer_geometry(f"ne_{detail}_land", crs, region)
    shelf = _layer_geometry(f"ne_{detail}_antarctic_ice_shelves_polys", crs, region)
    land_patch = None
    if land:
        land_patch = PathPatch(land[0], facecolor=MAP_STYLE["land"], edgecolor="none", zorder=1)
        ax.add_patch(land_patch)
    large = crs == "EPSG:4326" or (extent[1] - extent[0]) >= RELIEF_MIN_WIDTH
    if large and not _draw_relief(ax, crs, extent, relief_size, land_patch):
        _draw_gradient(ax, crs, extent, land_patch)
    if shelf:
        ax.add_patch(PathPatch(shelf[0], facecolor=MAP_STYLE["ice_shelf"], edgecolor="none", zorder=1.4))
        ax.plot(shelf[1], shelf[2], zorder=2.5, **MAP_STYLE["ice_shelf_edge"])
    if land:
        ax.plot(land[1], land[2], zorder=2.6, **MAP_STYLE["coast"])


# ---------- map furniture ----------

def _nice_step(span, candidates):
    return next((c for c in candidates if span / c <= 6), candidates[-1])


def _format_degrees(value, positive, negative):
    """71.5 → 71°30'S style label."""
    hemisphere = "" if abs(value) < 1e-9 or abs(abs(value) - 180) < 1e-9 else (negative if value < 0 else positive)
    degrees = int(abs(value))
    minutes = int(round((abs(value) - degrees) * 60))
    if minutes == 60:
        degrees, minutes = degrees + 1, 0
    return f"{degrees}°{minutes:02d}'{hemisphere}" if minutes else f"{degrees}°{hemisphere}"


def _edge_crossings(x, y, extent):
    """Where a projected line crosses the map frame: list of (edge, position)."""
    xmin, xmax, ymin, ymax = extent
    crossings = []
    for edge, coord, other, value, lo, hi in (("left", x, y, xmin, ymin, ymax), ("right", x, y, xmax, ymin, ymax),
                                              ("bottom", y, x, ymin, xmin, xmax), ("top", y, x, ymax, xmin, xmax)):
        d = coord - value
        k = np.where(np.isfinite(d[:-1]) & np.isfinite(d[1:]) & (np.sign(d[:-1]) != np.sign(d[1:])))[0]
        for i in k:
            t = d[i] / (d[i] - d[i + 1])
            pos = other[i] + t * (other[i + 1] - other[i])
            if lo < pos < hi:
                crossings.append((edge, pos))
    return crossings


def _draw_graticule(ax, crs, extent):
    """Thin dashed latitude/longitude lines, labelled outside the frame where they cross it."""
    xmin, xmax, ymin, ymax = extent
    if crs == "EPSG:4326":
        lon0, lon1, lat0, lat1 = xmin, xmax, ymin, ymax
    else:
        gx, gy = np.meshgrid(np.linspace(xmin, xmax, 30), np.linspace(ymin, ymax, 30))
        glon, glat = (np.asarray(v, dtype=float) for v in
                      warp_transform(crs, "EPSG:4326", gx.ravel().tolist(), gy.ravel().tolist()))
        ok = np.isfinite(glon) & np.isfinite(glat)
        lon0, lon1, lat0, lat1 = glon[ok].min(), glon[ok].max(), glat[ok].min(), glat[ok].max()
    polar = crs in ("EPSG:3031", "EPSG:3413")
    if polar and xmin < 0 < xmax and ymin < 0 < ymax:  # pole inside the map
        lon0, lon1 = -180, 180
        lat0, lat1 = (-90, lat1) if crs == "EPSG:3031" else (lat0, 90)
    lat_step = _nice_step(lat1 - lat0, [0.25, 0.5, 1, 2, 5, 10, 15, 30])
    lon_step = _nice_step(lon1 - lon0, [0.5, 1, 2, 5, 10, 15, 30, 45, 60])

    offsets = {"left": ((-4, 0), "right", "center"), "right": ((4, 0), "left", "center"),
               "bottom": ((0, -4), "center", "top"), "top": ((0, 4), "center", "bottom")}
    used = []

    def label_at_edges(x, y, text, preferred):
        crossings = _edge_crossings(x, y, extent)
        chosen = [c for c in crossings if c[0] in preferred] or crossings
        for edge, pos in chosen:
            px, py = {"left": (xmin, pos), "right": (xmax, pos), "bottom": (pos, ymin), "top": (pos, ymax)}[edge]
            if any(e == edge and abs(p - pos) < 0.06 * (xmax - xmin) for e, p in used):
                continue  # avoid overlapping labels on the same edge
            used.append((edge, pos))
            (dx, dy), ha, va = offsets[edge]
            ax.annotate(text, (px, py), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                        fontsize=7, color=MAP_STYLE["muted"], annotation_clip=False)
        return bool(chosen)

    lat_values = np.arange(np.ceil(lat0 / lat_step) * lat_step, lat1 + 1e-9, lat_step)
    for lat in lat_values[np.abs(lat_values) < 90]:
        lons = np.linspace(-180, 180, 1441) if polar else np.linspace(lon0, lon1, 721)
        x, y = _to_map(lons, np.full_like(lons, lat), crs)
        ax.plot(x, y, zorder=3, **MAP_STYLE["grid"])
        text = _format_degrees(lat, "N", "S")
        if not label_at_edges(x, y, text, ("left", "right")):
            # circle completely inside the map (e.g. around the pole): small label on the circle
            inside = np.where((x > xmin) & (x < xmax) & (y > ymin) & (y < ymax))[0]
            if inside.size:
                j = inside[np.nanargmin(y[inside])]
                ax.text(x[j], y[j], text, fontsize=7, color=MAP_STYLE["muted"], ha="center", va="bottom", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc=MAP_STYLE["land"], ec="none", alpha=0.85))

    lon_values = np.arange(np.ceil(lon0 / lon_step) * lon_step, lon1 + 1e-9, lon_step)
    if polar and lon1 - lon0 >= 359:
        lon_values = lon_values[lon_values < 180]  # 180°E and 180°W are the same line
    for lon in lon_values:
        lats = np.linspace(lat0, lat1, 361)
        x, y = _to_map(np.full_like(lats, lon), lats, crs)
        ax.plot(x, y, zorder=3, **MAP_STYLE["grid"])
        label_at_edges(x, y, _format_degrees(lon, "E", "W"), ("bottom", "top"))


def _draw_scale_bar(ax, extent):
    """Simple scale bar: a line with end ticks and the length in km, lower-left corner."""
    xmin, xmax, ymin, ymax = extent
    width, height = xmax - xmin, ymax - ymin
    length = max(v for v in (1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6) if v <= max(width / 4, 1e3))
    x0, y0, tick = xmin + 0.04 * width, ymin + 0.04 * height, 0.008 * height
    style = dict(color=MAP_STYLE["text"], linewidth=1.2, solid_capstyle="butt", zorder=8)
    ax.plot([x0, x0 + length], [y0, y0], **style)
    for xt in (x0, x0 + length):
        ax.plot([xt, xt], [y0 - tick, y0 + tick], **style)
    ax.text(x0 + length / 2, y0 + 1.6 * tick, f"{length / 1e3:g} km", ha="center", va="bottom", fontsize=8,
            color=MAP_STYLE["text"], zorder=8)


def _draw_north_arrow(ax):
    ax.annotate("N", xy=(0.05, 0.95), xytext=(0.05, 0.87), xycoords="axes fraction", ha="center", va="center",
                fontsize=9, weight="bold", color=MAP_STYLE["text"], zorder=8,
                arrowprops=dict(arrowstyle="-|>", color=MAP_STYLE["text"], linewidth=1))


def _setup_map(ax, crs, extent, region, detail, grid=True, scale_bar=True, relief_size=900):
    _draw_background(ax, crs, region, extent, detail, relief_size)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(MAP_STYLE["frame"])
        spine.set_linewidth(0.6)
    if grid:
        _draw_graticule(ax, crs, extent)
    if scale_bar and crs != "EPSG:4326":
        _draw_scale_bar(ax, extent)
    if crs.startswith("EPSG:326") or crs.startswith("EPSG:327"):
        _draw_north_arrow(ax)


def _new_map_figure(extent, title, subtitle, crs):
    """Figure with title, subtitle and a credit line, and a map area with the aspect ratio of the extent."""
    map_w = 9.0
    map_h = map_w * (extent[3] - extent[2]) / (extent[1] - extent[0])
    top, bottom, side = 1.25, 0.5, 0.5
    fig_w, fig_h = map_w + 2 * side, map_h + top + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([side / fig_w, bottom / fig_h, map_w / fig_w, map_h / fig_h])
    fig.text(side / fig_w, 1 - 0.3 / fig_h, title, fontsize=13, weight="bold", color=MAP_STYLE["text"], va="top")
    fig.text(side / fig_w, 1 - 0.62 / fig_h, subtitle, fontsize=9, color=MAP_STYLE["muted"], va="top")
    fig.text(side / fig_w, 0.18 / fig_h,
             f"Projection: {crs_name(crs)}   ·   Basemap: Natural Earth   ·   Data: ESA BIOMASS L1C",
             fontsize=7, color=MAP_STYLE["muted"], va="bottom")
    return fig, ax


def _add_inset(ax, crs, overview_extent, overview_region, zoom_extent):
    """Location map in the top-right corner, flush with the main map frame."""
    inset = ax.inset_axes([0.70, 0.70, 0.30, 0.30])
    inset.set_zorder(10)
    _setup_map(inset, crs, overview_extent, overview_region, "50m", grid=False, scale_bar=False, relief_size=400)
    x0, x1, y0, y1 = zoom_extent
    inset.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=MAP_STYLE["highlight"],
                              linewidth=1.2, zorder=9))
    if (x1 - x0) < 0.03 * (overview_extent[1] - overview_extent[0]):  # too small to see: add a marker
        inset.plot((x0 + x1) / 2, (y0 + y1) / 2, "o", markersize=7, markerfacecolor="none",
                   markeredgecolor=MAP_STYLE["highlight"], markeredgewidth=1.2, zorder=9)
    for spine in inset.spines.values():
        spine.set_linewidth(0.8)


def _plot_footprint(ax, xy, color, linewidth=0.9, fill_alpha=0.22):
    points = np.column_stack(xy)
    ax.add_patch(MplPolygon(points, closed=True, facecolor=color, alpha=fill_alpha, edgecolor="none", zorder=5))
    ax.add_patch(MplPolygon(points, closed=True, fill=False, edgecolor=color, linewidth=linewidth,
                            joinstyle="round", zorder=5.1))


def _footprint_handle(color, label, linewidth=0.9):
    return Patch(facecolor=to_rgba(color, 0.22), edgecolor=color, linewidth=linewidth, label=label)


def _primary_handle():
    return Line2D([], [], marker="*", linestyle="none", markersize=11, markerfacecolor=MAP_STYLE["primary"],
                  markeredgecolor=MAP_STYLE["frame"], markeredgewidth=0.5, label="primary")


def _legend(ax, handles, title, loc="lower right"):
    legend = ax.legend(handles=handles, title=title, loc=loc, fontsize=8, title_fontsize=8.5, frameon=True,
                       fancybox=False, framealpha=0.95, edgecolor="#c5cdd5", borderpad=0.8, labelspacing=0.6,
                       handlelength=1.6, handleheight=1.0)
    legend.get_frame().set_linewidth(0.6)
    legend.get_title().set_fontweight("bold")
    legend._legend_box.align = "left"
    legend.set_zorder(20)


def _mark_primary(ax, xy):
    ax.plot(np.nanmean(xy[0][:-1]), np.nanmean(xy[1][:-1]), marker="*", markersize=13, linestyle="none",
            markerfacecolor=MAP_STYLE["primary"], markeredgecolor=MAP_STYLE["frame"], markeredgewidth=0.5, zorder=7)


def _track_frame(name):
    m = re.search(r"_T(\d{3})_F(\d{3})", name)
    return f"track {m.group(1)}, frame {m.group(2)}" if m else ""


# ---------- the three maps ----------

def plot_overview_map(scenes, footprints):
    """All selected scenes on one map, one colour per stack."""
    lons = np.concatenate([footprints[s.name][0] for s in scenes])
    lats = np.concatenate([footprints[s.name][1] for s in scenes])
    crs = _overview_crs(lons, lats)
    xy = {s.name: _to_map(*footprints[s.name], crs) for s in scenes}
    extent = _continent_extent(crs, list(xy.values()))
    stacks = list(dict.fromkeys(s.stack for s in scenes))
    fig, ax = _new_map_figure(extent, "BIOMASS acquisitions – overview",
                              f"{len(scenes)} scenes in {len(stacks)} stack(s)", crs)
    _setup_map(ax, crs, extent, _region(crs, lons, lats, 15), "50m")
    handles = []
    for i, stack in enumerate(stacks):
        color = STACK_COLORS[i % len(STACK_COLORS)]
        stack_scenes = [s for s in scenes if s.stack == stack]
        for s in stack_scenes:
            _plot_footprint(ax, xy[s.name], color, linewidth=0.8)
        cx = np.nanmean(np.concatenate([xy[s.name][0] for s in stack_scenes]))
        cy = np.nanmean(np.concatenate([xy[s.name][1] for s in stack_scenes]))
        ax.plot(cx, cy, "o", markersize=5, markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.8,
                zorder=6)
        ax.annotate(stack, (cx, cy), xytext=(6, 6), textcoords="offset points", fontsize=8, weight="bold",
                    color=MAP_STYLE["text"], zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        handles.append(_footprint_handle(color, f"{stack}  ·  {len(stack_scenes)} scene(s)"))
    _legend(ax, handles, "Stacks")
    return fig


def plot_stack_map(stack_scenes, footprints):
    """Zoomed map of one stack: every acquisition by date, primary marked; inset shows the location."""
    lons = np.concatenate([footprints[s.name][0] for s in stack_scenes])
    lats = np.concatenate([footprints[s.name][1] for s in stack_scenes])
    crs = choose_crs(lons, lats)
    xy = {s.name: _to_map(*footprints[s.name], crs) for s in stack_scenes}
    extent = _extent(list(xy.values()))
    first, last = stack_scenes[0], stack_scenes[-1]
    subtitle = "  ·  ".join(x for x in (f"{len(stack_scenes)} acquisition(s)",
                                        f"{first.date:%Y-%m-%d} – {last.date:%Y-%m-%d}",
                                        _track_frame(first.name)) if x)
    if first.stack_id:
        subtitle += f"\nStack ID: {first.stack_id}"
    fig, ax = _new_map_figure(extent, f"BIOMASS stack {first.stack}", subtitle, crs)
    _setup_map(ax, crs, extent, _region(crs, lons, lats, 3), "10m")
    colors = plt.cm.YlOrRd(np.linspace(0.45, 0.95, max(len(stack_scenes), 2)))
    handles = []
    for color, s in zip(colors, stack_scenes):
        _plot_footprint(ax, xy[s.name], color, linewidth=1.6 if s.is_primary else 0.8, fill_alpha=0.12)
        handles.append(_footprint_handle(color, f"{s.date:%Y-%m-%d %H:%M}", 1.6 if s.is_primary else 0.8))
        if s.is_primary:
            _mark_primary(ax, xy[s.name])
    if any(s.is_primary for s in stack_scenes):
        handles.append(_primary_handle())
    _legend(ax, handles, "Acquisition (UTC)")
    _add_inset(ax, crs, _continent_extent(crs, list(xy.values())), _region(crs, lons, lats, 15), extent)
    return fig


def plot_scene_map(scene, footprints):
    """Zoomed map of one scene; inset shows the location."""
    lons, lats = footprints[scene.name]
    crs = choose_crs(lons, lats)
    xy = _to_map(lons, lats, crs)
    extent = _extent([xy])
    subtitle = "  ·  ".join(x for x in (f"stack {scene.stack}", _track_frame(scene.name),
                                        "primary" if scene.is_primary else "") if x)
    if scene.stack_id:
        subtitle += f"\nStack ID: {scene.stack_id}"
    fig, ax = _new_map_figure(extent, f"BIOMASS scene {scene.date:%Y-%m-%d %H:%M} UTC", subtitle, crs)
    _setup_map(ax, crs, extent, _region(crs, lons, lats, 3), "10m")
    color = MAP_STYLE["highlight"]
    _plot_footprint(ax, xy, color, linewidth=1.2)
    handles = [_footprint_handle(color, scene.name[:47] + ("…" if len(scene.name) > 47 else ""), 1.2)]
    if scene.is_primary:
        _mark_primary(ax, xy)
        handles.append(_primary_handle())
    _legend(ax, handles, "Scene footprint")
    _add_inset(ax, crs, _continent_extent(crs, [xy]), _region(crs, lons, lats, 15), extent)
    return fig


def run_aoi_maps(scenes, output_dir, overview=True, per_stack=True, per_scene=True, show=True):
    """Overview map, one zoomed map per stack and one per scene, saved in output_dir."""
    output_dir = Path(output_dir).expanduser()
    footprints = {}
    for s in scenes:
        try:
            footprints[s.name] = footprint(s)
        except Exception as err:
            print(f"No footprint for {s.label}: {err}")
    scenes = [s for s in scenes if s.name in footprints]
    if not scenes:
        print("No footprints found, nothing to plot.")
        return
    if overview:
        print("\nOverview map")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_figure(plot_overview_map(scenes, footprints), output_dir / "aoi_overview.png", show)
    if per_stack:
        for stack in dict.fromkeys(s.stack for s in scenes):
            stack_scenes = [s for s in scenes if s.stack == stack]
            print(f"\nStack map {stack_scenes[0].stack_label}")
            name = stack_scenes[0].stack_id or stack
            save_figure(plot_stack_map(stack_scenes, footprints),
                        stack_output_folder(stack_scenes[0], output_dir) / f"{name}_aoi.png", show)
    if per_scene:
        for s in scenes:
            print(f"\nScene map {s.label}")
            path = save_figure(plot_scene_map(s, footprints), output_path(s, "aoi", output_dir), show)
            _record(s, output_dir, "aoi", [path])
