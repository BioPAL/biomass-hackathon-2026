"""QGIS-friendly rasters from the extracted complex stacks.

QGIS cannot render CFloat32 (complex) rasters, so for every product tif written
by the extractor we also write a float32 intensity GeoTIFF in dB with the same
GCP georeferencing, a declared NaN nodata and built-in overviews. Open the
<product_id>_dB.tif files in QGIS; the complex originals stay untouched.
"""

import numpy as np
import rasterio


def is_product_tif(path):
    """True for the extractor's complex product tifs (not masks, latlon or dB)."""
    name = path.name
    return (
        name.startswith("BIO_")
        and name.endswith(".tif")
        and not name.endswith("_dB.tif")
    )


def convert_product(path, overwrite=False):
    """Write <id>_dB.tif: per-band intensity (dB, float32) next to the complex tif."""
    out_path = path.with_name(path.stem + "_dB.tif")
    if out_path.exists() and not overwrite:
        with rasterio.open(path) as ref:
            expected = (ref.count, ref.height, ref.width)
        try:  # a leftover partial file (e.g. after an aborted run) must not be trusted
            with rasterio.open(out_path) as old:
                ok = (old.count, old.height, old.width) == expected
        except Exception:
            ok = False
        if ok:
            return out_path, "skipped (exists)"

    tmp_path = out_path.with_suffix(".tif.part")
    with rasterio.open(path) as src:
        gcps, crs = src.gcps
        descriptions = list(src.descriptions)
        profile = dict(
            driver="GTiff",
            width=src.width,
            height=src.height,
            count=src.count,
            dtype="float32",
            nodata=float("nan"),
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="deflate",
        )
        with rasterio.open(tmp_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                data = src.read(band)
                power = np.abs(data) ** 2
                dst.write((10 * np.log10(power + 1e-12)).astype("float32"), band)
                if descriptions and band <= len(descriptions):
                    dst.set_band_description(band, descriptions[band - 1])
                del data, power
            if gcps:
                dst.gcps = (gcps, crs)
            dst.update_tags(source=str(path.name), units="dB (uncalibrated)")
    tmp_path.rename(out_path)
    with rasterio.open(out_path, "r+") as dst:
        dst.build_overviews([2, 4, 8])
    return out_path, "ok"


def convert_stack(stack_dir, overwrite=False):
    """Convert every product tif of a stack folder. Returns [(name, status)]."""
    results = []
    for path in sorted(stack_dir.glob("BIO_*.tif")):
        if is_product_tif(path):
            _, status = convert_product(path, overwrite=overwrite)
            results.append((path.name, status))
    return results


def convert_roi(roi_dir, overwrite=False):
    """Convert the products of every stack folder under an ROI folder."""
    done = []
    for stack_dir in sorted(p for p in roi_dir.iterdir() if p.is_dir()):
        results = convert_stack(stack_dir, overwrite=overwrite)
        if results:
            print(f"  {stack_dir.name}: {len(results)} QGIS dB rasters")
        done.extend(results)
    return done
