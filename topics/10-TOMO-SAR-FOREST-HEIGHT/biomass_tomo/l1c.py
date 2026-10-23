"""
BIOMASS Level-1c stack products (S1/S2/S3_STA__1S): reading and writing one analysis-ready netCDF stack.

A product folder (unzipped from the MAAP, or written by biomass_tomo.maap) holds
    measurement/                 *_i_abs.tiff, *_i_phase.tiff, *_i.vrt   (4 bands: HH HV VH VV)
    annotation_coregistered/     *_annot.xml, *_lut.nc                   (on the primary's grid)
    annotation_primary/          *_annot.xml                             (the acquisition's own annotation)
The folder may be nested once (<id>_product/<id>/measurement) as in the MAAP zip.

All acquisitions of a stack are coregistered onto the primary grid, so one pixel grid serves the whole
stack. The LUT layers of *_lut.nc are given on a coarse (relative azimuth time, slant-range time) grid
and are interpolated bilinearly onto the SLC grid.

Stack netCDF layout (build_stack):
    slc_real, slc_imag     (polarisation, acquisition, azimuth, range)   s = real + j*imag = |s| exp(j*phase)
    per-acquisition LUTs   (acquisition, azimuth, range)                 dem_phases, calibration_phases, ...
    single LUTs            (azimuth, range)                              incidence_angle, dem, latitude, ...
    coordinates            polarisation, acquisition (repeat cycle), azimuth / range [m],
                           azimuth_line / range_sample (frame indices), product_id, temporal_baseline
Values are stored as delivered: no-data SLC samples are set to 0 and LUT values keep their sign and units.

Command line:
    python -m biomass_tomo.l1c STACK_DIR OUT.nc [--crop AZ0 AZ1 RG0 RG1] [--inspect]
"""
import pathlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
import rasterio as rio
from rasterio.windows import Window

C0 = 299_792_458.0
POLARISATIONS = ("HH", "HV", "VH", "VV")

PRODUCT_PATTERN = re.compile(
    r"BIO_S\d_STA__1S_(?P<start>\d{8}T\d{6})_\d{8}T\d{6}_\w_G\d\d_M\d\d_C(?P<cycle>\d\d)_T\d{3}_F\d{3}_\d\d_\w+$")

# stack variable -> LUT path and storage. per_acquisition: one layer per acquisition (else from the first
# product only). primary_fill: value written for the coregistration primary, whose LUT lacks the layer.
LUT_VARIABLES = {
    # tomography
    "dem_phases":             dict(path="/coregistration/flatteningPhaseScreen", required=True, per_acquisition=True),
    "calibration_phases":     dict(path="/skpPhaseCalibration/skpCalibrationPhaseScreen", required=True, per_acquisition=True),
    "insar_wavenumbers":      dict(path="/coregistration/waveNumbers", required=True, per_acquisition=True),
    # geometry
    "incidence_angle":        dict(path="/geometry/incidenceAngle", required=True, per_acquisition=False),
    "range_slope":            dict(path="/geometry/rangeTerrainSlope", required=True, per_acquisition=False),
    "dem":                    dict(path="/geometry/height", required=True, per_acquisition=False),
    "azimuth_slope":          dict(path="/geometry/azimuthTerrainSlope", required=False, per_acquisition=False),
    "latitude":               dict(path="/geometry/latitude", required=False, per_acquisition=False),
    "longitude":              dict(path="/geometry/longitude", required=False, per_acquisition=False),
    # quality and L1c interferometry
    "calibration_quality":    dict(path="/skpPhaseCalibration/skpCalibrationPhaseScreenQuality", required=False, per_acquisition=True),
    "coregistration_quality": dict(path="/coregistration/coregistrationShiftsQuality", required=False, per_acquisition=True),
    "lut_coherence":          dict(path="/interferometry/coherence", required=False, per_acquisition=True, primary_fill=1.0),
    "lut_interferogram":      dict(path="/interferometry/interferogram", required=False, per_acquisition=True, primary_fill=0.0),
    # radiometry: the SLC is beta-nought; sigma0 = beta0 * sigma_nought_factor, gamma0 = beta0 * gamma_nought_factor
    "sigma_nought_factor":    dict(path="/radiometry/sigmaNought", required=False, per_acquisition=False),
    "gamma_nought_factor":    dict(path="/radiometry/gammaNought", required=False, per_acquisition=False),
    # thermal noise power per polarisation (beta-nought), not removed from the L1c SLC
    **{f"noise_{p}": dict(path=f"/denoising/denoising{p}", required=False, per_acquisition=True) for p in POLARISATIONS},
}
for _spec in LUT_VARIABLES.values():
    _spec.setdefault("primary_fill", None)

# annotation fields of the pixel grid: key -> candidate XML tags (case-insensitive, first found wins)
ANNOT_FIELDS = {
    "first_range_time": ("firstSampleSlantRangeTime", "firstRangeTime", "rangeTimeFirstPixel", "slantRangeTime"),
    "first_azimuth_time": ("firstLineAzimuthTime", "firstAzimuthTime", "azimuthTimeFirstLine", "productFirstLineUtcTime"),
    "range_time_interval": ("rangeTimeInterval", "rangeSamplingTime"),
    "azimuth_time_interval": ("azimuthTimeInterval", "azimuthLineInterval"),
    "range_spacing": ("rangePixelSpacing", "rangeSpacing", "slantRangeSpacing"),
    "azimuth_spacing": ("azimuthPixelSpacing", "azimuthSpacing"),
}

# products
def find_product_dirs(stack_dir):
    """Product folders of one stack (those with measurement/), ordered by repeat cycle."""
    found = {}
    for d in sorted(pathlib.Path(stack_dir).rglob("BIO_*_STA__1S_*")):
        if d.is_dir() and PRODUCT_PATTERN.match(d.name) and (d / "measurement").is_dir():
            found[d.name] = d
    if not found:
        raise FileNotFoundError(f"no L1c STA__1S product folders with measurement/ under {stack_dir}")
    return sorted(found.values(), key=lambda p: int(PRODUCT_PATTERN.match(p.name)["cycle"]))


def product_file(product_dir, subdir, pattern):
    """The single file matching pattern in product_dir/subdir. subdir 'annotation' prefers the
    coregistered annotation (the stack grid) over the primary one."""
    product_dir = pathlib.Path(product_dir)
    subdirs = ["annotation_coregistered", "annotation", "annotation_primary"] if subdir == "annotation" else [subdir]
    for sd in subdirs:
        hits = sorted((product_dir / sd).glob(pattern))
        if len(hits) > 1:
            raise RuntimeError(f"{len(hits)} files match {sd}/{pattern} in {product_dir.name}")
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{pattern} not found in {product_dir.name}/{'|'.join(subdirs)}")


def annotation_value(xml_path, names):
    """Text of the first element whose tag (namespace stripped, case-insensitive) is one of names."""
    names = {n.lower() for n in names}
    for el in ET.parse(xml_path).getroot().iter():
        if el.tag.rsplit("}", 1)[-1].lower() in names and el.text and el.text.strip():
            return el.text.strip()
    return None


def is_primary(product_dir):
    """True for the coregistration primary: zero normal baseline in the annotation or, failing that,
    no /interferometry layers in its LUT."""
    ann = product_file(product_dir, "annotation", "*_annot.xml")
    baseline = annotation_value(ann, ("normalBaseline",))
    if baseline is not None:
        return float(baseline) == 0.0
    return not _lut_has(product_file(product_dir, "annotation", "*_lut.nc"), "/interferometry/coherence")


def read_vrt(product_dir):
    """Band number per polarisation, global metadata and no-data value from the product .vrt."""
    root = ET.parse(product_file(product_dir, "measurement", "*_i.vrt")).getroot()
    metadata = {m.get("key"): (m.text or "").strip() for m in root.find("Metadata")}
    bands = {b.findtext("Description").strip().upper(): int(b.get("band")) for b in root.findall("VRTRasterBand")}
    nodata = root.find("VRTRasterBand").findtext("NoDataValue")
    return bands, metadata, float(nodata) if nodata else None


def acquisition_info(product_dirs):
    """Repeat cycle, product id, start time and days since the first acquisition, per product."""
    ids = [PRODUCT_PATTERN.match(p.name) for p in product_dirs]
    times = [datetime.strptime(m["start"], "%Y%m%dT%H%M%S") for m in ids]
    return {"cycle": [int(m["cycle"]) for m in ids], "product_id": [p.name for p in product_dirs], "time": times,
            "baseline_days": [(t - times[0]).total_seconds() / 86400 for t in times]}


def frame_shape(product_dir):
    """(azimuth lines, range samples) of the full frame."""
    with rio.open(product_file(product_dir, "measurement", "*_i_abs.tiff")) as src:
        return src.height, src.width

# pixel grid
def _parse_time(s):
    s = s.strip().rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "UTC=%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(s[:26], fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognised time {s!r}")


def read_annotation(product_dir):
    """Pixel-grid parameters from the coregistered annotation (times in s, spacings in m)."""
    xml_path = product_file(product_dir, "annotation", "*_annot.xml")
    out = {}
    for key, names in ANNOT_FIELDS.items():
        val = annotation_value(xml_path, names)
        if val is None:
            raise KeyError(f"{key}: none of {names} in {xml_path.name}; extend ANNOT_FIELDS")
        out[key] = _parse_time(val) if key == "first_azimuth_time" else float(val)
    return out


def crop_bounds(n_az, n_rg, crop):
    """(az0, az1, rg0, rg1) from a crop (None, {"azimuth": (a0, a1), "range": (r0, r1)} or 4-tuple), clipped."""
    if crop is None:
        return 0, n_az, 0, n_rg
    if isinstance(crop, dict):
        (az0, az1), (rg0, rg1) = crop.get("azimuth", (0, n_az)), crop.get("range", (0, n_rg))
    else:
        az0, az1, rg0, rg1 = crop
    az0, az1, rg0, rg1 = max(0, int(az0)), min(n_az, int(az1)), max(0, int(rg0)), min(n_rg, int(rg1))
    if az1 <= az0 or rg1 <= rg0:
        raise ValueError(f"empty crop {crop} for frame {n_az} x {n_rg}")
    return az0, az1, rg0, rg1


def build_grid(product_dir, bounds):
    """SLC grid of the crop: along-track position x_ax and slant range r_ax [m] (course convention), and
    the time axes used to interpolate the LUTs."""
    az0, az1, rg0, rg1 = bounds
    a = read_annotation(product_dir)
    lines, samples = np.arange(az0, az1), np.arange(rg0, rg1)
    rg_time = a["first_range_time"] + samples * a["range_time_interval"]
    return {"x_ax": lines * a["azimuth_spacing"],
            "r_ax": rg_time * C0 / 2,
            "az_time_rel": lines * a["azimuth_time_interval"],     # s since the first line
            "rg_time": rg_time,                                     # two-way slant-range time [s]
            "first_azimuth_time": a["first_azimuth_time"],
            "annotation": a}

# SLC
def read_slc(product_dir, band, bounds):
    """Complex SLC of one polarisation band on the crop, |s| exp(j*phase); no-data -> 0.
    Returns (slc, number of no-data samples)."""
    az0, az1, rg0, rg1 = bounds
    win = Window(rg0, az0, rg1 - rg0, az1 - az0)
    parts, bad = [], None
    for name in ("*_i_abs.tiff", "*_i_phase.tiff"):
        with rio.open(product_file(product_dir, "measurement", name)) as src:
            a = src.read(band, window=win).astype(np.float32)
            b = ~np.isfinite(a) | ((a == np.float32(src.nodata)) if src.nodata is not None else False)
        parts.append(a)
        bad = b if bad is None else bad | b
    slc = (parts[0] * np.exp(1j * parts[1])).astype(np.complex64)
    slc[bad] = 0
    return slc, int(bad.sum())


# LUT
def _get_var(nc, var_path):
    *groups, name = [p for p in var_path.split("/") if p]
    grp = nc
    for g in groups:
        if g not in grp.groups:
            raise KeyError(var_path)
        grp = grp.groups[g]
    if name not in grp.variables:
        raise KeyError(var_path)
    return grp, grp.variables[name]


def _lut_has(nc_path, var_path):
    from netCDF4 import Dataset
    with Dataset(nc_path) as nc:
        try:
            _get_var(nc, var_path)
            return True
        except KeyError:
            return False


def _axis_variable(grp, dim):
    """1-D coordinate of a dimension: a same-named variable in the group or a parent, else a 1-D
    variable on that dimension whose name mentions azimuth/range/time."""
    for match in (lambda n, v: n == dim, lambda n, v: v.dimensions == (dim,) and re.search(r"azimuth|range|time", n, re.I)):
        g = grp
        while g is not None:
            for n, v in g.variables.items():
                if v.ndim == 1 and match(n, v):
                    return v
            g = g.parent
    raise KeyError(f"no coordinate variable for dimension {dim!r} in {grp.path}")


def _is_azimuth(name):
    return bool(re.search(r"az|line|row|y", name, re.I)) and not re.search(r"range", name, re.I)


def lut_axes(nc_path, var_path):
    """One LUT layer with its axes: (azimuth axis, range axis, data (n_az, n_rg) with fill -> NaN,
    variable attributes, axis names, axis attributes)."""
    from netCDF4 import Dataset
    with Dataset(nc_path) as nc:
        grp, var = _get_var(nc, var_path)
        var.set_auto_mask(False)
        data = np.squeeze(np.asarray(var[:], dtype=np.float64))
        attrs = {k: var.getncattr(k) for k in var.ncattrs()}
        axes = [_axis_variable(grp, d) for d in var.dimensions]
        names = [a.name for a in axes]
        vals = [np.asarray(a[:], dtype=np.float64) for a in axes]
        axattrs = [{k: a.getncattr(k) for k in a.ncattrs()} for a in axes]
    if not (_is_azimuth(names[0]) or (not _is_azimuth(names[1]) and vals[1].max() < 1e-2)):
        data, vals, names, axattrs = data.T, vals[::-1], names[::-1], axattrs[::-1]
    fill = attrs.get("_FillValue", attrs.get("missing_value"))
    if fill is not None:
        data[data == fill] = np.nan
    return vals[0], vals[1], data, attrs, names, axattrs


def _relative_azimuth(az_axis, az_attrs, grid):
    """LUT azimuth axis in s since the first SLC line, shifted if the axis names its own reference epoch."""
    for key in ("reference", "referenceTime", "reference_time", "units"):
        v = az_attrs.get(key)
        m = re.search(r"\d{4}-\d\d-\d\dT[\d:.]+", v) if isinstance(v, str) else None
        if m:
            return az_axis + (_parse_time(m.group(0)) - grid["first_azimuth_time"]).total_seconds()
    return az_axis


def _interp_separable(az_axis, rg_axis, data, az_out, rg_out):
    """Bilinear interpolation on a rectilinear grid (range, then azimuth), constant extrapolation."""
    if az_axis[0] > az_axis[-1]:
        az_axis, data = az_axis[::-1], data[::-1]
    if rg_axis[0] > rg_axis[-1]:
        rg_axis, data = rg_axis[::-1], data[:, ::-1]

    def rows(a, x, x_out):
        out = np.full((a.shape[0], x_out.size), np.nan)
        for i, row in enumerate(a):
            ok = np.isfinite(row)
            if ok.sum() > 1:
                out[i] = np.interp(x_out, x[ok], row[ok])
        return out

    return rows(rows(data, rg_axis, rg_out).T, az_axis, az_out).T.astype(np.float32)


def read_lut_on_grid(product_dir, spec, grid):
    """One LUT layer of one product interpolated onto the SLC grid.
    Returns (values (n_az, n_rg) float32, info) or (None, {}) if the layer is absent and has no
    primary_fill. info: units, lut_variable, interpolation, filled."""
    nc_path = product_file(product_dir, "annotation", "*_lut.nc")
    shape = (grid["x_ax"].size, grid["r_ax"].size)
    try:
        az, rg, data, attrs, names, axattrs = lut_axes(nc_path, spec["path"])
    except KeyError:
        if spec.get("primary_fill") is None:
            return None, {}
        return (np.full(shape, spec["primary_fill"], np.float32),
                {"filled": True, "units": "", "lut_variable": spec["path"], "interpolation": "constant"})
    vals = _interp_separable(_relative_azimuth(az, axattrs[0], grid), rg, data, grid["az_time_rel"], grid["rg_time"])
    return vals, {"filled": False, "units": str(attrs.get("units", attrs.get("unit", ""))), "lut_variable": spec["path"],
                  "interpolation": "azimuth: relative azimuth time, range: slant-range time",
                  "axes": names, "lut_shape": list(data.shape)}

# inspection
def inspect_product(product_dir):
    """Print what this module finds in a product: files, annotation grid, SLC, LUT variables and axes."""
    from netCDF4 import Dataset
    p = pathlib.Path(product_dir)
    print(p.name)
    for sd in ("measurement", "annotation_coregistered", "annotation_primary"):
        for f in sorted((p / sd).glob("*")):
            print(f"  {sd}/{f.name}  {f.stat().st_size / 1e6:.1f} MB")
    try:
        print("  annotation:", read_annotation(p))
    except KeyError as e:
        print("  annotation:", e)
    with rio.open(product_file(p, "measurement", "*_i_abs.tiff")) as src:
        print(f"  SLC: {src.count} bands, {src.height} az x {src.width} rg, dtype {src.dtypes[0]}, nodata {src.nodata}")
    lut = product_file(p, "annotation", "*_lut.nc")
    with Dataset(lut) as nc:
        def walk(g):
            for name, v in g.variables.items():
                print(f"  {g.path.rstrip('/')}/{name} {v.dimensions} {v.shape} {getattr(v, 'units', '')}")
            for sub in g.groups.values():
                walk(sub)
        walk(nc)
    for name, spec in LUT_VARIABLES.items():
        try:
            az, rg, _, _, names, _ = lut_axes(lut, spec["path"])
            print(f"  {name:<24} axes {names}  az {az[0]:.4g}..{az[-1]:.4g} ({az.size})  rg {rg[0]:.4g}..{rg[-1]:.4g} ({rg.size})")
        except KeyError as e:
            print(f"  {name:<24} absent ({e})")

# stack netCDF
def _add_variable(nc, name, dims, data=None, dtype="f4", attrs=None, **kwargs):
    var = nc.createVariable(name, dtype, dims, **kwargs)
    if data is not None:
        var[:] = data
    var.setncatts(attrs or {})
    return var


def build_stack(stack_dir, out_path, crop=None, polarisations=POLARISATIONS, site="site",
                optional_layers=True, aux_complevel=4, azimuth_chunk=512, progress=print):
    """Write all products under stack_dir to one netCDF-4 file (layout in the module docstring).
    crop: None (full frame), {"azimuth": (a0, a1), "range": (r0, r1)} or (az0, az1, rg0, rg1) in frame pixels.
    optional_layers: also write the LUT layers that are not required for tomography.
    SLCs are written uncompressed (speckle does not compress); LUT layers with zlib."""
    from netCDF4 import Dataset

    product_dirs = find_product_dirs(stack_dir)
    n_acq = len(product_dirs)
    pol_bands, vrt_meta, _ = read_vrt(product_dirs[0])
    missing = [p for p in polarisations if p not in pol_bands]
    if missing:
        raise KeyError(f"polarisations {missing} not in the product (available {list(pol_bands)})")
    full_shape = frame_shape(product_dirs[0])
    az0, az1, rg0, rg1 = bounds = crop_bounds(*full_shape, crop)
    n_az, n_rg = az1 - az0, rg1 - rg0
    grid = build_grid(product_dirs[0], bounds)
    acq = acquisition_info(product_dirs)
    layers = {k: v for k, v in LUT_VARIABLES.items() if v["required"] or optional_layers}
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress(f"{n_acq} acquisitions, crop {bounds} of frame {full_shape} -> {out_path}")

    with Dataset(out_path, "w", format="NETCDF4") as nc:
        for name, size in {"polarisation": len(polarisations), "acquisition": n_acq, "azimuth": n_az, "range": n_rg}.items():
            nc.createDimension(name, size)
        pv = nc.createVariable("polarisation", str, ("polarisation",))
        for i, p in enumerate(polarisations):
            pv[i] = p
        _add_variable(nc, "acquisition", ("acquisition",), acq["cycle"], "i4", {"long_name": "repeat cycle number"})
        pid = nc.createVariable("product_id", str, ("acquisition",))
        for i, name in enumerate(acq["product_id"]):
            pid[i] = name
        _add_variable(nc, "acquisition_time", ("acquisition",), [(t - acq["time"][0]).total_seconds() for t in acq["time"]],
                      "f8", {"units": f"seconds since {acq['time'][0].isoformat()}", "long_name": "acquisition start time"})
        _add_variable(nc, "temporal_baseline", ("acquisition",), acq["baseline_days"], "f8",
                      {"units": "days", "long_name": "time since first acquisition"})
        _add_variable(nc, "azimuth", ("azimuth",), grid["x_ax"], "f8", {"units": "m", "long_name": "along-track position"})
        _add_variable(nc, "range", ("range",), grid["r_ax"], "f8", {"units": "m", "long_name": "slant range"})
        _add_variable(nc, "azimuth_line", ("azimuth",), np.arange(az0, az1), "i4", {"long_name": "azimuth line in the frame"})
        _add_variable(nc, "range_sample", ("range",), np.arange(rg0, rg1), "i4", {"long_name": "range sample in the frame"})

        dims = ("polarisation", "acquisition", "azimuth", "range")
        slc_vars = {part: _add_variable(nc, f"slc_{part}", dims, chunksizes=(1, 1, min(azimuth_chunk, n_az), n_rg),
                                        attrs={"comment": "s = slc_real + 1j*slc_imag = |s| exp(j*phase); no-data set to 0"})
                    for part in ("real", "imag")}
        nodata = np.zeros((len(polarisations), n_acq), dtype=np.int64)
        for ip, pol in enumerate(polarisations):
            for ia, p in enumerate(product_dirs):
                progress(f"  SLC {pol} {ia + 1}/{n_acq}")
                slc, nodata[ip, ia] = read_slc(p, pol_bands[pol], bounds)
                slc_vars["real"][ip, ia] = slc.real
                slc_vars["imag"][ip, ia] = slc.imag
        _add_variable(nc, "nodata_pixels", ("polarisation", "acquisition"), nodata, "i8",
                      {"long_name": "number of no-data samples set to 0"})

        for name, spec in layers.items():
            # read every acquisition first, so a layer missing from one product is skipped as a whole
            values, infos, filled = [], [], []
            for p in product_dirs if spec["per_acquisition"] else product_dirs[:1]:
                v, info = read_lut_on_grid(p, spec, grid)
                if v is None:
                    values = None
                    break
                values.append(v)
                (filled if info["filled"] else infos).append(p.name if info["filled"] else info)
            if values is None:
                if spec["required"]:
                    raise KeyError(f"required LUT layer {spec['path']} missing")
                progress(f"  {name}: absent, skipped")
                continue
            info = infos[0] if infos else {"units": "", "lut_variable": spec["path"], "interpolation": "constant"}
            attrs = {"units": info["units"], "source": info["lut_variable"],
                     "interpolation": f"bilinear; {info['interpolation']}", "per_acquisition": int(spec["per_acquisition"])}
            if filled:
                attrs["constant_fill"] = f"{spec['primary_fill']} in {', '.join(filled)} (primary acquisition)"
            var = _add_variable(nc, name, ("acquisition", "azimuth", "range") if spec["per_acquisition"] else ("azimuth", "range"),
                                attrs=attrs, zlib=True, complevel=aux_complevel, shuffle=True)
            var[:] = np.stack(values) if spec["per_acquisition"] else values[0]
            progress(f"  {name}: done")

        nc.setncatts({
            "title": f"Biomass L1c stack {site}", "site": site, "source_products": str(stack_dir),
            "processor": vrt_meta.get("SOFTWARE", ""), "polarisation_sequence": vrt_meta.get("PolarisationsSequence", ""),
            "swath": vrt_meta.get("Swath", ""), "crop_azimuth": f"{az0}:{az1}", "crop_range": f"{rg0}:{rg1}",
            "slc_definition": "s = |s| exp(j*phase) from *_i_abs.tiff and *_i_phase.tiff",
            "phase_sign_convention": "LUT values stored as delivered; no sign applied",
            "history": f"{datetime.now(timezone.utc).isoformat()} created by biomass_tomo.l1c.build_stack",
        })
    return out_path

def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Write a BIOMASS L1c stack to one netCDF file.")
    ap.add_argument("stack_dir", help="folder containing the product folders of one stack")
    ap.add_argument("out", nargs="?", help="output netCDF (omit with --inspect)")
    ap.add_argument("--crop", nargs=4, type=int, metavar=("AZ0", "AZ1", "RG0", "RG1"), help="frame pixel crop")
    ap.add_argument("--site", default="site")
    ap.add_argument("--inspect", action="store_true", help="only print what is found in the primary product")
    args = ap.parse_args()
    dirs = find_product_dirs(args.stack_dir)
    if args.inspect or not args.out:
        inspect_product(next((d for d in dirs if is_primary(d)), dirs[0]))
        return
    build_stack(args.stack_dir, args.out, crop=args.crop, site=args.site)


if __name__ == "__main__":
    _main()
