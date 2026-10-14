from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import earthaccess
import numpy as np
import xarray as xr
import h5netcdf

from shapely import wkt
from shapely.geometry import Polygon
from shapely.ops import transform as transform_geometry
from pyproj import Transformer

ROI_WKT = """POLYGON Z ((
-65.68807416302047 -2.7244108894438956 0,
-65.458802 -1.643179 0,
-65.33633787313144 -1.6690300717478612 0,
-65.11302859315191 -2.719746573201742 0,
-65.12751593782825 -2.7884256995503343 0,
-65.65041031544851 -2.8982542274449736 0,
-65.65148074881392 -2.8980294395334445 0,
-65.68807416302047 -2.7244108894438956 0
))"""


COLLECTION = "NISAR_L2_GCOV_PROVISIONAL_V1"

# Covariance terms used for reciprocal quad-pol Pauli powers.
COVARIANCE_TERMS = ["HHHH", "HVHV", "VVVV", "HHVV"]


@dataclass
class NISARPauliReader:
    roi_wkt: str = ROI_WKT
    collection: str = COLLECTION
    frequency: str = "A"
    target_crs: str = "EPSG:32620"
    count: int = 2000

    # Keep remote streams alive while the datasets are in use.
    _streams: list = field(default_factory=list, init=False)
    granules: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.roi = wkt.loads(self.roi_wkt).buffer(0)
        if self.roi.is_empty or not self.roi.is_valid:
            raise ValueError("Invalid ROI polygon")

        self.to_projected = Transformer.from_crs(
            "EPSG:4326", self.target_crs, always_xy=True
        ).transform

        self.roi_projected = transform_geometry(self.to_projected, self.roi)

    def search(self, start=None, end=None):
        """Search the PROVISIONAL GCOV collection for ROI overlaps."""
        earthaccess.login()

        # Earthaccess polygon uses (latitude, longitude) pairs.
        polygon = [(lat, lon) for lon, lat in self.roi.exterior.coords]

        kwargs = {
            "short_name": self.collection,
            "polygon": polygon,
            "count": self.count,
        }
        if start and end:
            kwargs["temporal"] = (start, end)

        self.granules = earthaccess.search_data(**kwargs)
        return self.granules

    @staticmethod
    def _get_dataset(tree, path):
        """Get a dataset from an Xarray DataTree path."""
        return tree[path].to_dataset()

    def _open_granule(self, granule):
        """
        Stream a granule and open its HDF5 hierarchy.

        The file object is retained to preserve the lazy remote handle.
        """
        links = granule.data_links(access="external")
        if not links:
            raise RuntimeError("No external data link for granule")

        fs = earthaccess.get_fsspec_https_session()
        stream = fs.open(
            links[0],
            cache_type="background",
            block_size=16 * 1024 * 1024,
        )
        self._streams.append(stream)

        tree = xr.open_datatree(
            stream,
            engine="h5netcdf",
            decode_timedelta=False,
            phony_dims="access",
        )
        return tree

    @staticmethod
    def _find_xy(ds):
        """Locate projected coordinate variables."""
        x_names = ("xCoordinates", "x", "projection_x_coordinate")
        y_names = ("yCoordinates", "y", "projection_y_coordinate")

        x = next((n for n in x_names if n in ds.coords or n in ds), None)
        y = next((n for n in y_names if n in ds.coords or n in ds), None)

        if x is None or y is None:
            raise ValueError(
                "Could not identify x/y coordinates. Inspect the "
                "GCOV grid metadata and adapt _find_xy()."
            )
        return x, y

    def _clip_dataset(self, ds):
        """
        Spatially subset by the ROI bounding box, then mask pixels
        outside the polygon. Data variables remain lazy.
        """
        x_name, y_name = self._find_xy(ds)
        x = ds[x_name]
        y = ds[y_name]

        # This reader assumes projected, 1-D x/y coordinates in
        # target_crs. Verify the projection metadata for each granule.
        minx, miny, maxx, maxy = self.roi_projected.bounds

        xv = np.asarray(x.values)
        yv = np.asarray(y.values)

        # Handle either ascending or descending y coordinates.
        x_slice = slice(minx, maxx) if xv[0] < xv[-1] else slice(maxx, minx)
        y_slice = slice(miny, maxy) if yv[0] < yv[-1] else slice(maxy, miny)

        clipped = ds.sel({x_name: x_slice, y_name: y_slice})

        # Rasterize the ROI mask on the clipped coordinate grid.
        # This creates only a small 2-D boolean mask; SAR arrays
        # remain lazy Dask-backed arrays.
        from rasterio.features import geometry_mask
        from rasterio.transform import from_origin

        xc = np.asarray(clipped[x_name].values)
        yc = np.asarray(clipped[y_name].values)

        if len(xc) < 2 or len(yc) < 2:
            raise ValueError("ROI intersection has fewer than 2 pixels")

        dx = float(np.median(np.diff(xc)))
        dy = float(np.median(np.diff(yc)))

        # Rasterio affine transform uses top-left origin.
        transform = from_origin(
            xc.min() - abs(dx) / 2,
            yc.max() + abs(dy) / 2,
            abs(dx),
            abs(dy),
        )

        mask_array = geometry_mask(
            [self.roi_projected.__geo_interface__],
            out_shape=(len(yc), len(xc)),
            transform=transform,
            invert=True,
        )

        # Coordinate order may be descending or ascending. Align
        # mask with dataset dimension order.
        if yc[0] < yc[-1]:
            mask_array = mask_array[::-1, :]

        mask_da = xr.DataArray(
            mask_array,
            dims=(y_name, x_name),
            coords={y_name: clipped[y_name], x_name: clipped[x_name]},
        )

        return clipped.where(mask_da)

    @staticmethod
    def _complex_values(da):
        """
        Convert a covariance term to complex values when stored
        as real/imag compound data. Real-valued terms are preserved.
        """
        data = da.data
        dtype = getattr(data, "dtype", None)

        if dtype is not None and dtype.fields:
            fields = dtype.fields
            names = list(fields)
            if len(names) >= 2:
                return da.copy(data=data[names[0]] + 1j * data[names[1]])

        return da

    def _pauli(self, ds):
        """
        Compute Pauli power channels from covariance terms.

        Assumes HHVV represents <HH * conj(VV)> and that HHHH,
        HVHV, VVVV are real-valued covariance powers.
        """
        missing = [name for name in COVARIANCE_TERMS if name not in ds.data_vars]
        if missing:
            raise ValueError(f"Missing required covariance terms: {missing}")

        hh = ds["HHHH"].real
        hv = ds["HVHV"].real
        vv = ds["VVVV"].real
        hhvv = self._complex_values(ds["HHVV"])

        cross = hhvv.real

        # Pauli power components, with a consistent normalization.
        red = (hh + vv - 2 * cross) / 2
        green = 2 * hv
        blue = (hh + vv + 2 * cross) / 2

        # Numerical roundoff can create tiny negative powers.
        pauli = xr.Dataset(
            {
                "pauli_red": red.clip(min=0),
                "pauli_green": green.clip(min=0),
                "pauli_blue": blue.clip(min=0),
            }
        )

        pauli.attrs["description"] = (
            "Pauli RGB powers derived from GCOV covariance terms; "
            "red=HH-VV, green=cross-pol, blue=HH+VV."
        )
        return pauli

    def open_one(self, granule):
        """Open one granule, select frequency, clip, and derive RGB."""
        tree = self._open_granule(granule)

        grid_path = f"/science/LSAR/GCOV/grids/frequency{self.frequency}"
        ds = self._get_dataset(tree, grid_path)

        # Preserve original covariance terms and apply ROI clipping.
        clipped = self._clip_dataset(ds)

        # Keep the full covariance dataset and add derived channels.
        pauli = self._pauli(clipped)
        output = xr.merge([clipped, pauli], compat="override")

        output.attrs["source_granule"] = str(
            granule["meta"].get("native-id", "unknown")
        )
        output.attrs["collection"] = self.collection
        output.attrs["frequency"] = self.frequency
        output.attrs["roi_crs"] = self.target_crs

        return output

    def open_all(self):
        """
        Return a dictionary of lazy clipped Xarray datasets,
        keyed by granule ID.
        """
        if not self.granules:
            self.search()

        output = {}
        for granule in self.granules:
            granule_id = granule["meta"].get("native-id", f"granule_{len(output)}")
            try:
                output[granule_id] = self.open_one(granule)
                print("Opened lazily:", granule_id)
            except Exception as exc:
                print(f"Skipped {granule_id}: {exc}")

        return output

    def close(self):
        """Close datasets/streams when finished."""
        for stream in self._streams:
            try:
                stream.close()
            except Exception:
                pass
        self._streams.clear()


reader = NISARPauliReader(frequency="A")

# Optional date limits; omit them for the archive-wide search.
granules = reader.search()

print("Search results:", len(granules))

# Each result stays lazy; no full-scene raster is loaded into RAM.
scenes = reader.open_all()

for granule_id, ds in scenes.items():
    print(granule_id, ds)
    print(ds["pauli_red"].data)


import matplotlib.pyplot as plt
import numpy as np


def show_pauli(ds, percentile=(2, 98), title=None):
    channels = [
        ds["pauli_red"],
        ds["pauli_green"],
        ds["pauli_blue"],
    ]

    rgb = []
    for channel in channels:
        values = channel.compute().values.astype("float32")
        valid = np.isfinite(values)

        if not valid.any():
            rgb.append(np.zeros_like(values))
            continue

        lo, hi = np.nanpercentile(values[valid], percentile)

        if hi <= lo:
            scaled = np.zeros_like(values)
        else:
            scaled = np.clip((values - lo) / (hi - lo), 0, 1)

        scaled[~valid] = 0
        rgb.append(scaled)

    image = np.stack(rgb, axis=-1)

    # Xarray dimensions are normally (y, x).
    plt.figure(figsize=(10, 8))
    plt.imshow(image, origin="upper")
    plt.title(title or ds.attrs.get("source_granule", "NISAR Pauli"))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# Display the first successful scene.
if scenes:
    first_id = next(iter(scenes))
    show_pauli(scenes[first_id], title=first_id)
