#!/usr/bin/env python3
"""
Multi-baseline multi-polarimetric (MBMP) coherence of the extracted L1C stacks.

Works on the output of extract_l1c_roi.py, stack by stack (no cross-stack
combinations).

Workflow
--------
1. For every stack folder of the ROI, the LUT of every product is downloaded
   once and its ``flatteningPhaseScreen`` / ``waveNumbers`` cropped to the
   window are cached in ``flattening_lut.nc``. The crops are flattened with
   ``s * exp(1j * flat)`` (flat phase bilinearly interpolated onto the pixels,
   as BIOMASSStackProductReader(fit_to_l1c_raster=True) does).
2. The averaging window is given in metres on ground and converted to pixels
   with the BPS rule (covariance_options_struct_filler)::

       NW = int(L / pixel_spacing_ground / 2) * 2 + 1,  subsampling = (NW - 1) / 2

   ground range spacing = slant range spacing / sin(mean incidence) as in
   main_covariance_estimation_SR, azimuth spacing measured on latlon.tif.
   The effective window in metres is printed for every stack and must be
   confirmed: check that it is close to a square.
3. MBMP correlation (BPS port of MPMBCovarianceEstimation +
   Covariance4D2Correlation4D): channels HH, XP = (HV + VH) / 2, VV, each with
   all the acquisitions -> (3 Nb) x (3 Nb) matrix (21 x 21 for 7 dates),
   block order pol-major: index = pol * Nb + acquisition.
4. Single-pol coherence on the complex data of HH, HV, VH and VV separately,
   same window and grid: gamma_nm = <s_n s_m*> / sqrt(<|s_n|^2> <|s_m|^2>).

If the ``bps`` package is importable, the ported MBMP estimator is checked
against bps.l2a_processor main_correlation_estimation_SR on a small block.

Output (in every stack folder)
------------------------------
    flattening_lut.nc            flattening phase + kz of every product (LUT grid)
    coherence_w<L>m.npz          mbmp_corr   (Naz, Nrg, 3Nb, 3Nb) complex64
                                 mbmp_power  (Naz, Nrg, 3Nb)      float32
                                 coh_pol     (Naz, Nrg, 4, Nb, Nb) complex64 [HH HV VH VV]
                                 power_pol   (Naz, Nrg, 4, Nb)     float32
                                 valid_fraction (Naz, Nrg)  share of the window inside
                                                            the common area with data
                                 lat, lon (Naz, Nrg), az_index, rg_index (crop pixels)
    coherence_w<L>m.json         window, spacings, dates, primary, mean kz, ...

Requires: numpy, scipy, rasterio, netCDF4, requests, pystac-client


examples:
python compute_coherence.py                         # asks ROI folder and window
python compute_coherence.py --roi-dir Lac_Tele --window 250 --yes
"""

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import netCDF4
import numpy as np
import rasterio
import requests
from scipy.ndimage import map_coordinates
from scipy.sparse import csr_matrix

# -------------------- Configuration --------------------

LIGHTSPEED = 299792458.0
POLARISATIONS = ["HH", "HV", "VH", "VV"]
MBMP_POLS = ["HH", "XP", "VV"]  # BPS FH/TOMO order, XP = (HV + VH) / 2
LUT_LAYERS = ("flatteningPhaseScreen", "waveNumbers")
# cells whose window has less than this share of valid pixels are flagged in the json stats
MIN_VALID_FRACTION = 0.5

SCRIPT_DIR = Path(__file__).resolve().parent


# -------------------- BPS port (bps.l2a_processor.tomo_fh.tomo_fh_commands) --------------------


def bps_window(window_m, pixel_spacing_grd):
    """covariance_options_struct_filler: odd window size and subsampling in pixels."""
    nw = int(window_m / pixel_spacing_grd / 2) * 2 + 1
    nw_mono = (nw - 1) // 2
    return nw, nw_mono, nw_mono


def build_filtering_matrix(n_in, nw_mono, subs_f):
    """Sparse one-dimensional moving average + decimation, normalised by the samples inside the image."""
    x_out = np.arange(0, n_in, subs_f)
    taps = np.arange(-nw_mono, nw_mono + 1)
    col = (x_out[:, None] + taps[None, :]).ravel()
    row = np.repeat(np.arange(x_out.size), taps.size)
    ok = (col >= 0) & (col < n_in)
    f = csr_matrix((np.ones(ok.sum()), (row[ok], col[ok])), shape=(x_out.size, n_in))
    norm = 1.0 / np.asarray(f.sum(axis=1)).ravel()
    return csr_matrix(f.multiply(norm[:, None])), x_out


def average(fa, frt, image):
    """Boxcar average + decimation of a (Naz, Nrg) image: Fa @ image @ Fr.T"""
    return fa @ image @ frt


def correlation_from_covariance(cov):
    """Covariance4D2Correlation4D on a (Naz, Nrg, N, N) array."""
    diag = np.real(np.diagonal(cov, axis1=2, axis2=3))
    norm = 1.0 / np.sqrt(diag + np.spacing(1))
    return cov * norm[..., :, None] * norm[..., None, :], diag


def mbmp_covariance(channels, fa, frt, n_az_out, n_rg_out):
    """MPMBCovarianceEstimation: upper triangle of <c_i c_j*> + hermitian completion.

    channels: list of N (Naz, Nrg) complex arrays, NaN already set to 0.
    """
    n = len(channels)
    cov = np.zeros((n_az_out, n_rg_out, n, n), dtype=np.complex64)
    for i in range(n):
        ci = channels[i].astype(np.complex128)
        for j in range(i, n):
            prod = ci * np.conj(channels[j].astype(np.complex128))
            cov[:, :, i, j] = average(fa, frt, prod).astype(np.complex64)
            if j != i:
                cov[:, :, j, i] = np.conj(cov[:, :, i, j])
    return cov


# -------------------- Stack inputs --------------------


def load_stack_info(folder):
    info = json.loads((folder / "stack_info.json").read_text())
    info["products"] = [p for p in info["products"] if p.get("status") in ("ok", "skipped (exists)")
                        and (folder / p["file"]).exists()]
    info["products"].sort(key=lambda p: p["datetime"])
    return info


def lut_crop_slice(info, lut_shape):
    """LUT nodes covering the crop window (one node of margin)."""
    w = info["window"]
    step_az, step_rg = info["lut_step_azimuth_range"]
    i0 = max(w["row_off"] // step_az - 1, 0)
    j0 = max(w["col_off"] // step_rg - 1, 0)
    i1 = min(-(-(w["row_off"] + w["height"] - 1) // step_az) + 2, lut_shape[0])
    j1 = min(-(-(w["col_off"] + w["width"] - 1) // step_rg) + 2, lut_shape[1])
    return i0, i1, j0, j1


def fetch_flattening_luts(folder, info, token_file, workers):
    """Download the LUT of every product and cache flattening phase + kz cropped to the window."""
    out_path = folder / "flattening_lut.nc"
    ids = [p["id"] for p in info["products"]]
    if out_path.exists():
        with netCDF4.Dataset(out_path) as ds:
            if list(ds["product_id"][:]) == ids:
                return out_path

    # only needed to download the LUTs (keeps the rest runnable where pystac is not installed)
    from extract_l1c_roi import COLLECTION, STAC_API_URL, find_token_file, load_token_manager
    from pystac_client import Client

    tokens = load_token_manager(find_token_file(token_file))
    catalog = Client.open(STAC_API_URL)
    items = {it.id: it for it in catalog.search(collections=[COLLECTION], ids=ids, method="GET").items()}
    missing = [i for i in ids if i not in items]
    if missing:
        raise SystemExit(f"Products not found in the catalogue: {missing}")

    def download(pid):
        href = items[pid].assets["enclosure_nc"].href
        response = requests.get(href, headers={"Authorization": f"Bearer {tokens.get()}"}, timeout=600)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
            tmp.write(response.content)
            tmp.flush()
            del response
            with netCDF4.Dataset(tmp.name) as ds:
                nodata = float(ds.noDataValue)
                grp = ds["coregistration"]
                shape = grp["flatteningPhaseScreen"].shape
                i0, i1, j0, j1 = lut_crop_slice(info, shape)
                layers = {}
                for name in LUT_LAYERS:
                    data = np.ma.filled(grp[name][i0:i1, j0:j1].astype(np.float64), np.nan)
                    data[data == nodata] = np.nan
                    layers[name] = data
                srt = ds["slantRangeTime"][:]
                azt = ds["relativeAzimuthTime"][:]
        print(f"    LUT {pid[:47]}... ok")
        return layers, (i0, i1, j0, j1), (float(np.median(np.diff(np.asarray(srt)))), float(np.median(np.diff(np.asarray(azt)))))

    print(f"  downloading {len(ids)} LUTs (flattening phase screen)...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(download, ids))

    sl = results[0][1]
    if any(r[1] != sl for r in results):
        raise RuntimeError("LUT grids differ between the products of the stack")
    step_az, step_rg = info["lut_step_azimuth_range"]
    w = info["window"]
    tmp_path = out_path.with_suffix(".nc.part")
    with netCDF4.Dataset(str(tmp_path), "w") as ds:
        ds.createDimension("product", len(ids))
        ds.createDimension("azimuth", sl[1] - sl[0])
        ds.createDimension("range", sl[3] - sl[2])
        ds.description = (
            "Flattening phase / kz of every product, LUT grid cropped. LUT node (a, r) is crop pixel "
            "(a*azimuth_step + lut_row_offset, r*range_step + lut_col_offset)."
        )
        ds.azimuth_step, ds.range_step = step_az, step_rg
        ds.lut_row_offset = sl[0] * step_az - w["row_off"]
        ds.lut_col_offset = sl[2] * step_rg - w["col_off"]
        # pixel sampling of the image, from the LUT time axes
        ds.slant_range_time_step = results[0][2][0] / step_rg
        ds.azimuth_time_step = results[0][2][1] / step_az
        var = ds.createVariable("product_id", str, ("product",))
        for k, pid in enumerate(ids):
            var[k] = pid
        for name in LUT_LAYERS:
            var = ds.createVariable(name, "f8", ("product", "azimuth", "range"), fill_value=np.nan, zlib=True)
            var[:] = np.stack([r[0][name] for r in results])
    tmp_path.rename(out_path)
    return out_path


def lut_to_crop(lut, ds, shape):
    """Bilinear interpolation of a LUT layer onto every crop pixel (BPS fit_to_l1c_raster)."""
    rows = (np.arange(shape[0]) - ds.lut_row_offset) / ds.azimuth_step
    cols = (np.arange(shape[1]) - ds.lut_col_offset) / ds.range_step
    out = np.empty(shape, dtype=np.float64)
    for r0 in range(0, shape[0], 1024):
        rr, cc = np.meshgrid(rows[r0:r0 + 1024], cols, indexing="ij")
        out[r0:r0 + 1024] = map_coordinates(lut, [rr, cc], order=1, mode="nearest")
    return out


# -------------------- Window --------------------


def pixel_spacings(folder, flat_path):
    """Slant range / ground range / azimuth spacing [m] and mean incidence [rad] of a stack."""
    with netCDF4.Dataset(flat_path) as ds:
        slant = LIGHTSPEED / 2 * float(ds.slant_range_time_step)
    with netCDF4.Dataset(folder / "geometry_lut.nc") as ds:
        inc = np.deg2rad(np.nanmean(np.ma.filled(ds["incidenceAngle"][:].astype(float), np.nan)))
    with rasterio.open(folder / "latlon.tif") as src:
        lat, lon = np.deg2rad(src.read().astype(np.float64))
    radius = 6371008.8
    d_az = radius * np.hypot(np.diff(lat, axis=0), np.diff(lon, axis=0) * np.cos(lat[1:]))
    return {
        "slant_range_m": slant,
        "ground_range_m": float(slant / np.sin(inc)),  # main_covariance_estimation_SR
        "azimuth_m": float(np.nanmedian(d_az)),
        "incidence_mean_deg": float(np.rad2deg(inc)),
    }


def describe_window(window_m, sp):
    nw_r, mono_r, subs_r = bps_window(window_m, sp["ground_range_m"])
    nw_a, mono_a, subs_a = bps_window(window_m, sp["azimuth_m"])
    ext_az, ext_rg = nw_a * sp["azimuth_m"], nw_r * sp["ground_range_m"]
    return {
        "window_m": window_m,
        "NW_a": nw_a, "NW_r": nw_r, "NWmono_a": mono_a, "NWmono_r": mono_r,
        "subs_F_a": subs_a, "subs_F_r": subs_r,
        "window_ground_az_m": ext_az, "window_ground_rg_m": ext_rg,
        "aspect_rg_over_az": ext_rg / ext_az,
        "output_spacing_az_m": subs_a * sp["azimuth_m"],
        "output_spacing_rg_m": subs_r * sp["ground_range_m"],
        "looks": nw_a * nw_r,
    }


def print_window_table(stacks):
    print("\n  stack                                          window (az x rg)       on ground (az x rg)   "
          "rg/az  looks  output grid")
    for name, win in stacks:
        print(f"  {name:<46s} {win['NW_a']:3d} x {win['NW_r']:2d} px   "
              f"{win['window_ground_az_m']:7.0f} x {win['window_ground_rg_m']:5.0f} m   "
              f"{win['aspect_rg_over_az']:5.2f}  {win['looks']:5d}  "
              f"{win['output_spacing_az_m']:.0f} x {win['output_spacing_rg_m']:.0f} m")
    print()


# -------------------- Coherence --------------------


def read_stack(folder, info, flat_path):
    """Flattened complex data {pol: [Nb x (Naz, Nrg)]} and the joint valid mask."""
    data = {pol: [] for pol in POLARISATIONS}
    valid = None
    with netCDF4.Dataset(flat_path) as ds:
        flat_all = np.ma.filled(ds["flatteningPhaseScreen"][:], np.nan)
        for k, prod in enumerate(info["products"]):
            with rasterio.open(folder / prod["file"]) as src:
                shape = (src.height, src.width)
                ramp = np.exp(1j * lut_to_crop(flat_all[k], ds, shape)).astype(np.complex64)
                for band, pol in enumerate(POLARISATIONS, start=1):
                    s = src.read(band) * ramp
                    ok = np.isfinite(s)
                    valid = ok if valid is None else valid & ok
                    data[pol].append(s)
            print(f"    read + flattened {prod['file'][:47]}...")
    with rasterio.open(folder / "overlap_mask.tif") as src:
        valid &= src.read(1).astype(bool)
    for pol in POLARISATIONS:
        for s in data[pol]:
            s[~np.isfinite(s)] = 0  # BPS: no-data set to 0 before averaging
    return data, valid


def check_against_bps(channels_by_pol, win, sp):
    """Compare the port with bps main_correlation_estimation_SR on a small block (if bps is installed)."""
    try:
        from bps.l2a_processor.tomo_fh.tomo_fh_commands import main_correlation_estimation_SR
    except ImportError:
        print("  (bps not importable: skipping the cross-check against the BPS implementation)")
        return None
    nb = len(channels_by_pol["HH"])
    n_az, n_rg = channels_by_pol["HH"][0].shape
    a0, r0 = n_az // 2, n_rg // 2
    sl = (slice(a0, a0 + 8 * win["NW_a"]), slice(r0, r0 + 12 * win["NW_r"]))
    # BPS convention: {acq: {pol: (Rg, Az)}}, slant range spacing + incidence
    stack = {k: {pol: channels_by_pol[pol][k][sl].T.astype(np.complex128) for pol in MBMP_POLS}
             for k in range(nb)}
    inc = np.deg2rad(sp["incidence_mean_deg"])
    corr_bps, *_ = main_correlation_estimation_SR(
        stack, win["window_m"], sp["slant_range_m"], sp["azimuth_m"], inc, 435e6, 6e6)
    corr_bps = np.transpose(corr_bps, (3, 2, 0, 1))
    fa, _ = build_filtering_matrix(sl[0].stop - sl[0].start, win["NWmono_a"], win["subs_F_a"])
    fr, _ = build_filtering_matrix(sl[1].stop - sl[1].start, win["NWmono_r"], win["subs_F_r"])
    chans = [channels_by_pol[pol][k][sl] for pol in MBMP_POLS for k in range(nb)]
    corr, _ = correlation_from_covariance(mbmp_covariance(chans, fa, fr.T.tocsc(), *corr_bps.shape[:2]))
    err = float(np.max(np.abs(corr - corr_bps)))
    print(f"  cross-check vs BPS main_correlation_estimation_SR: max |diff| = {err:.2e}")
    return err


def process_stack(folder, info, flat_path, win, sp):
    t0 = time.time()
    data, valid = read_stack(folder, info, flat_path)
    data["XP"] = [(hv + vh) / 2 for hv, vh in zip(data["HV"], data["VH"])]
    nb = len(info["products"])
    n_az, n_rg = valid.shape

    bps_err = check_against_bps(data, win, sp)

    fa, az_out = build_filtering_matrix(n_az, win["NWmono_a"], win["subs_F_a"])
    fr, rg_out = build_filtering_matrix(n_rg, win["NWmono_r"], win["subs_F_r"])
    frt = fr.T.tocsc()
    shape_out = (az_out.size, rg_out.size)
    print(f"  crop {n_az} x {n_rg} px -> coherence grid {shape_out[0]} x {shape_out[1]}")

    print(f"  MBMP covariance ({len(MBMP_POLS) * nb} x {len(MBMP_POLS) * nb})...")
    channels = [data[pol][k] for pol in MBMP_POLS for k in range(nb)]
    mbmp_corr, mbmp_power = correlation_from_covariance(mbmp_covariance(channels, fa, frt, *shape_out))
    del data["XP"], channels

    print("  single-pol coherence (HH, HV, VH, VV)...")
    coh_pol = np.zeros(shape_out + (len(POLARISATIONS), nb, nb), dtype=np.complex64)
    power_pol = np.zeros(shape_out + (len(POLARISATIONS), nb), dtype=np.float32)
    for p, pol in enumerate(POLARISATIONS):
        corr, power = correlation_from_covariance(mbmp_covariance(data[pol], fa, frt, *shape_out))
        coh_pol[:, :, p] = corr
        power_pol[:, :, p] = power

    valid_fraction = average(fa, frt, valid.astype(np.float64)).astype(np.float32)
    with rasterio.open(folder / "latlon.tif") as src:
        latlon = src.read()[:, az_out][:, :, rg_out]

    with netCDF4.Dataset(flat_path) as ds:
        kz_mean = [float(np.nanmean(ds["waveNumbers"][k])) for k in range(nb)]

    stem = f"coherence_w{win['window_m']:g}m"
    np.savez(
        folder / f"{stem}.npz",
        mbmp_corr=mbmp_corr.astype(np.complex64), mbmp_power=mbmp_power.astype(np.float32),
        coh_pol=coh_pol, power_pol=power_pol, valid_fraction=valid_fraction,
        lat=latlon[0], lon=latlon[1], az_index=az_out, rg_index=rg_out,
    )
    good = valid_fraction >= MIN_VALID_FRACTION
    meta = {
        "stack": folder.name,
        "stack_id": info["stack_id"],
        "orbit_state": info.get("orbit_state"),
        "products": [p["id"] for p in info["products"]],
        "dates": [p["datetime"][:10] for p in info["products"]],
        "repeat_cycles": [p["repeat_cycle"] for p in info["products"]],
        "primary_index": next((k for k, p in enumerate(info["products"]) if p["coregistration_primary"]), None),
        "kz_mean_rad_per_m": kz_mean,
        "mbmp_pols": MBMP_POLS,
        "mbmp_order": "index = pol * Nb + acquisition, pols HH, XP=(HV+VH)/2, VV",
        "coh_pols": POLARISATIONS,
        "flattened": True,
        "spacings": sp,
        "window": win,
        "grid_shape": list(shape_out),
        "min_valid_fraction": MIN_VALID_FRACTION,
        "cells_valid": int(good.sum()),
        "mean_abs_mbmp_valid_cells": float(np.abs(mbmp_corr[good]).mean()) if good.any() else None,
        "bps_crosscheck_max_abs_diff": bps_err,
        "processing_seconds": round(time.time() - t0, 1),
    }
    (folder / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    print(f"  -> {folder.name}/{stem}.npz  ({time.time() - t0:.0f} s)")


# -------------------- Main --------------------


def ask(prompt, default=None):
    if not sys.stdin.isatty():
        if default is None:
            raise SystemExit(f"Missing input '{prompt}' (pass it as a flag)")
        return default
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roi-dir", help="ROI folder written by extract_l1c_roi.py (e.g. Lac_Tele)")
    parser.add_argument("--window", type=float, help="averaging window side on ground [m]")
    parser.add_argument("--stacks", help="comma separated stack folder names (default: all)")
    parser.add_argument("--token-file", help="MAAP credentials.txt / token.txt (to download the LUTs)")
    parser.add_argument("--workers", type=int, default=4, help="parallel LUT downloads")
    parser.add_argument("--yes", action="store_true", help="do not ask to confirm the window")
    args = parser.parse_args()

    roi_dir = Path(args.roi_dir or ask("ROI folder", "Lac_Tele"))
    if not roi_dir.is_absolute() and not roi_dir.exists():
        roi_dir = SCRIPT_DIR / roi_dir
    folders = sorted(p.parent for p in roi_dir.glob("*/stack_info.json"))
    if args.stacks:
        wanted = {s.strip() for s in args.stacks.split(",")}
        folders = [f for f in folders if f.name in wanted]
    if not folders:
        raise SystemExit(f"No stack folders in {roi_dir}")
    print(f"{len(folders)} stacks in {roi_dir}: " + ", ".join(f.name for f in folders))

    stacks = []
    for folder in folders:
        print(f"\n[{folder.name}]")
        info = load_stack_info(folder)
        flat_path = fetch_flattening_luts(folder, info, args.token_file, args.workers)
        sp = pixel_spacings(folder, flat_path)
        print(f"  {len(info['products'])} products, spacing: slant range {sp['slant_range_m']:.2f} m, "
              f"ground range {sp['ground_range_m']:.2f} m (incidence {sp['incidence_mean_deg']:.1f} deg), "
              f"azimuth {sp['azimuth_m']:.2f} m")
        stacks.append((folder, info, flat_path, sp))

    # window: the user checks that the averaged cell is about a square on ground
    window = args.window
    while True:
        if window is None:
            window = float(ask("Averaging window side on ground [m]", "250"))
        wins = [describe_window(window, sp) for *_, sp in stacks]
        print_window_table([(f.name, w) for (f, *_), w in zip(stacks, wins)])
        if args.yes or not sys.stdin.isatty():
            break
        answer = ask("Is the window close enough to a square? y = go, n = try another size", "y")
        if answer.lower().startswith("y"):
            break
        window = None

    for (folder, info, flat_path, sp), win in zip(stacks, wins):
        print(f"\n[{folder.name}]  window {win['NW_a']} x {win['NW_r']} px")
        process_stack(folder, info, flat_path, win, sp)


if __name__ == "__main__":
    main()
