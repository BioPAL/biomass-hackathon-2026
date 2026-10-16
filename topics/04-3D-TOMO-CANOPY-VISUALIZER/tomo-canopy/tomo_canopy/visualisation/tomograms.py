"""Display prepared TOM layers without hidden scientific processing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
import xarray as xr

from tomo_canopy.transforms import geographic_to_local_xy
from tomo_canopy.types import DataCube, GeometryType, Modality
from tomo_canopy.visualisation.config import (
    Tomogram3DConfig,
    TomogramIsosurfaceConfig,
    TomogramPlotConfig,
)


def _tomogram(cube: DataCube, layer: str) -> xr.Dataset:
    return cube.get_layer(
        layer, modality=Modality.TOMOGRAM, geometry=GeometryType.VOLUME
    ).data


def _variable(
    dataset: xr.Dataset, name: str, dimensions: Iterable[str]
) -> xr.DataArray:
    if name not in dataset:
        raise ValueError(
            f"Prepared TOM layer has no {name!r} variable; create it with "
            "TomogramProcessor.prepare_vertical_structure(...)"
        )
    variable = dataset[name]
    missing = set(dimensions).difference(variable.dims)
    if missing:
        raise ValueError(f"Variable {name!r} is missing dimensions: {sorted(missing)}")
    return variable


def _one_polarization(variable: xr.DataArray, polarization: str | None) -> xr.DataArray:
    if "polarization" not in variable.dims:
        return variable
    if polarization is not None:
        return variable.sel(polarization=polarization)
    if variable.sizes["polarization"] != 1:
        choices = ", ".join(map(str, variable.polarization.values))
        raise ValueError(f"Choose one polarization from: {choices}")
    return variable.isel(polarization=0, drop=True)


def plot_tomogram_bands(
    cube: DataCube,
    *,
    layer: str,
    variable: str = "vertical_contribution",
    polarization: str | None = None,
    config: TomogramPlotConfig | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot prepared height bands as comparable small multiples."""
    plt = import_module("matplotlib.pyplot")
    values = _one_polarization(
        _variable(_tomogram(cube, layer), variable, ("height_band", "y", "x")),
        polarization,
    )
    settings = config or TomogramPlotConfig()
    if settings.columns <= 0:
        raise ValueError("config.columns must be positive")
    count = values.sizes["height_band"]
    rows = int(np.ceil(count / settings.columns))
    vmax = settings.contribution_max
    if vmax is None:
        vmax = max(0.2, float(values.quantile(0.99, skipna=True)))
    fig, axes = plt.subplots(
        rows,
        settings.columns,
        figsize=(settings.figure_width, settings.panel_height * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    artist = None
    for axis, height in zip(axes.flat, values.height_band.values):
        band = values.sel(height_band=height)
        artist = axis.pcolormesh(
            band.x,
            band.y,
            band,
            cmap=settings.cmap,
            vmin=0,
            vmax=vmax,
            shading="auto",
        )
        axis.set_title(f"height band centred at {float(height):g} m")
        axis.set_aspect("equal")
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    if artist is not None:
        fig.colorbar(
            artist,
            ax=axes.flat[:count].tolist(),
            label="fraction of column power in band",
            shrink=0.8,
        )
    return fig, axes


def plot_tomogram_metrics(
    cube: DataCube,
    *,
    layer: str,
    metrics: Iterable[str] = (
        "scattering_height_50",
        "scattering_height_90",
        "scattering_centroid",
        "vertical_entropy",
        "upper_contribution",
    ),
    polarization: str | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot already-derived TOM metrics; no metric is computed here."""
    plt = import_module("matplotlib.pyplot")
    dataset = _tomogram(cube, layer)
    names = tuple(metrics)
    if not names:
        raise ValueError("At least one metric is required")
    columns = min(3, len(names))
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.5 * columns, 4 * rows),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for axis, name in zip(axes.flat, names):
        values = _one_polarization(
            _variable(dataset, name, ("y", "x")), polarization
        )
        artist = axis.pcolormesh(values.x, values.y, values, shading="auto")
        axis.set_title(name.replace("_", " "))
        axis.set_aspect("equal")
        fig.colorbar(artist, ax=axis, shrink=0.8)
    for axis in axes.flat[len(names):]:
        axis.set_visible(False)
    return fig, axes


def plot_tomogram_vertical_slice(
    cube: DataCube,
    *,
    layer: str,
    axis: str = "x",
    coordinate: float | None = None,
    variable: str = "tomographic_power_db",
    polarization: str | None = None,
    height_range: tuple[float, float] | None = None,
    cmap: str = "magma",
    robust_percentiles: tuple[float, float] = (2.0, 98.0),
    figsize: tuple[float, float] = (11.0, 4.0),
) -> tuple[Any, Any]:
    """Plot the nearest vertical plane from an already-prepared TOM variable.

    ``axis`` is the coordinate held constant: ``axis="x"`` produces a
    north/height section, while ``axis="y"`` produces an east/height section.
    Selecting a plane, restricting ``height_range``, and setting colour limits
    are visual operations; conversion to power or decibels remains an explicit
    processor operation. The source layer in the cube is not modified.
    """
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    if len(robust_percentiles) != 2 or not (
        0 <= robust_percentiles[0] < robust_percentiles[1] <= 100
    ):
        raise ValueError("robust_percentiles must be increasing values in [0, 100]")
    if (
        len(figsize) != 2
        or not np.isfinite(figsize).all()
        or figsize[0] <= 0
        or figsize[1] <= 0
    ):
        raise ValueError("figsize must contain two positive finite values")

    dataset = _tomogram(cube, layer)
    values = _one_polarization(
        _variable(dataset, variable, ("z", "y", "x")), polarization
    )
    if height_range is not None:
        if (
            len(height_range) != 2
            or not np.isfinite(height_range).all()
            or height_range[0] >= height_range[1]
        ):
            raise ValueError("height_range must be two increasing finite values")
        values = values.sel(z=slice(float(height_range[0]), float(height_range[1])))
        if values.sizes["z"] == 0:
            raise ValueError("height_range does not intersect the tomogram")
    selected_coordinate = (
        float(values[axis].values[values.sizes[axis] // 2])
        if coordinate is None
        else float(coordinate)
    )
    if not np.isfinite(selected_coordinate):
        raise ValueError("coordinate must be finite")
    section = values.sel({axis: selected_coordinate}, method="nearest")
    across = "y" if axis == "x" else "x"

    if str(dataset.attrs.get("crs", "")).upper() in {"EPSG:4326", "EPSG:4979"}:
        local_x, local_y = geographic_to_local_xy(values.x.values, values.y.values)
        horizontal = local_y if across == "y" else local_x
        horizontal_label = "north offset (m)" if across == "y" else "east offset (m)"
    else:
        horizontal = np.asarray(values[across].data, dtype=float)
        horizontal_label = across

    display = np.asarray(section.transpose("z", across).data, dtype=float)
    finite = display[np.isfinite(display)]
    if finite.size == 0:
        raise ValueError("Selected vertical slice contains no finite values")
    vmin, vmax = np.nanpercentile(finite, robust_percentiles)
    plt = import_module("matplotlib.pyplot")
    figure, axis_object = plt.subplots(figsize=figsize, constrained_layout=True)
    artist = axis_object.pcolormesh(
        horizontal,
        section.z,
        display,
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        shading="auto",
    )
    actual = float(section[axis])
    units = section.attrs.get("units", "")
    label = variable.replace("_", " ") + (f" ({units})" if units else "")
    figure.colorbar(artist, ax=axis_object, label=label)
    axis_object.set(
        xlabel=horizontal_label,
        ylabel="relative tomographic height (m)",
        title=f"{polarization or ''} vertical TOM slice at {axis}={actual:.6g}".strip(),
    )
    if height_range is not None:
        axis_object.set_ylim(*height_range)
    return figure, axis_object


@dataclass
class CanopyViewer:
    """Convenience facade over display functions for one prepared cube."""

    cube: DataCube

    def tomogram_bands(self, *, layer: str, **kwargs: Any):
        return plot_tomogram_bands(self.cube, layer=layer, **kwargs)

    def tomogram_metrics(self, *, layer: str, **kwargs: Any):
        return plot_tomogram_metrics(self.cube, layer=layer, **kwargs)

    def tomogram_vertical_slice(self, *, layer: str, **kwargs: Any):
        return plot_tomogram_vertical_slice(self.cube, layer=layer, **kwargs)

    def interactive_tomogram(self, *, layer: str, **kwargs: Any):
        """Create the slider-driven TOM point viewer for this cube."""
        from tomo_canopy.visualisation.interactive import InteractiveTomogramViewer

        return InteractiveTomogramViewer(self.cube, layer=layer, **kwargs)

    def tomogram_isosurfaces_3d(
        self,
        *,
        layer: str,
        variable: str = "tomographic_power_db",
        polarization: str | None = None,
        config: TomogramIsosurfaceConfig | None = None,
        notebook: bool = True,
    ) -> Any:
        """Render nested isosurfaces of an explicitly prepared TOM variable."""
        try:
            pv = import_module("pyvista")
        except ImportError as error:
            raise ImportError(
                "3D TOM visualisation requires `pip install tomo-canopy[visualisation]`"
            ) from error
        dataset = _tomogram(self.cube, layer)
        source = dataset.get(variable)
        if source is None:
            raise ValueError(f"Prepared TOM layer has no {variable!r} variable")
        vertical = "height_band" if "height_band" in source.dims else "z"
        values = _one_polarization(
            _variable(dataset, variable, (vertical, "y", "x")), polarization
        ).transpose(vertical, "y", "x")
        settings = config or TomogramIsosurfaceConfig()
        if not (
            len(settings.levels)
            == len(settings.colors)
            == len(settings.opacities)
        ):
            raise ValueError("Isosurface levels, colors, and opacities must align")
        if not settings.levels or not np.isfinite(settings.levels).all():
            raise ValueError("Isosurface levels must contain finite values")
        if settings.vertical_exaggeration <= 0:
            raise ValueError("vertical_exaggeration must be positive")

        data = np.asarray(values.data, dtype=float)
        floor = float(min(settings.levels)) - max(
            1.0, float(np.ptp(settings.levels))
        )
        data = np.where(np.isfinite(data), data, floor)
        if str(dataset.attrs.get("crs", "")).upper() in {"EPSG:4326", "EPSG:4979"}:
            display_x, display_y = geographic_to_local_xy(
                values.x.values, values.y.values
            )
        else:
            display_x = np.asarray(values.x.data, dtype=float)
            display_y = np.asarray(values.y.data, dtype=float)
        height = np.asarray(values[vertical].data, dtype=float)
        grid = pv.RectilinearGrid(
            display_x,
            display_y,
            height * settings.vertical_exaggeration,
        )
        grid.point_data[variable] = np.transpose(data, (2, 1, 0)).ravel(order="F")
        plotter = pv.Plotter(notebook=notebook, window_size=settings.window_size)
        plotter.set_background(settings.background)
        for level, color, opacity in zip(
            settings.levels, settings.colors, settings.opacities
        ):
            surface = grid.contour(isosurfaces=[level], scalars=variable)
            if surface.n_points:
                plotter.add_mesh(
                    surface,
                    color=color,
                    opacity=opacity,
                    smooth_shading=True,
                    label=f"{variable} ≥ {level:g}",
                )
        plotter.add_mesh(
            pv.Plane(
                center=(0, 0, 0),
                i_size=float(np.ptp(display_x)),
                j_size=float(np.ptp(display_y)),
            ),
            color=settings.ground_color,
            opacity=settings.ground_opacity,
        )
        plotter.enable_depth_peeling()
        plotter.add_legend(bcolor="#202830", face="rectangle")
        plotter.show_grid(
            xtitle="east / x",
            ytitle="north / y",
            ztitle=f"display height ({settings.vertical_exaggeration:g}×)",
        )
        plotter.camera_position = "iso"
        return plotter

    def tomogram_contribution_3d(
        self,
        *,
        layer: str,
        variable: str = "vertical_contribution",
        polarization: str | None = None,
        config: Tomogram3DConfig | None = None,
    ) -> Any:
        """Render contribution with height colour and contribution opacity."""
        try:
            pv = import_module("pyvista")
        except ImportError as error:
            raise ImportError(
                "3D TOM visualisation requires `pip install tomo-canopy[visualisation]`"
            ) from error
        values = _one_polarization(
            _variable(
                _tomogram(self.cube, layer),
                variable,
                ("height_band", "y", "x"),
            ),
            polarization,
        )
        settings = config or Tomogram3DConfig()
        data = np.asarray(values.data, dtype=float)
        keep = np.isfinite(data) & (data >= settings.contribution_floor)
        iz, iy, ix = np.where(keep)
        contribution = data[iz, iy, ix]
        if contribution.size == 0:
            raise ValueError("No contributions remain above the display floor")
        low, high = np.nanpercentile(contribution, settings.opacity_percentiles)
        span = high - low
        opacity = (
            np.ones_like(contribution)
            if span <= 0
            else 0.05 + 0.95 * np.clip((contribution - low) / span, 0, 1) ** 0.8
        )
        height = np.asarray(values.height_band.data)[iz]
        dataset = _tomogram(self.cube, layer)
        if str(dataset.attrs.get("crs", "")).upper() in {"EPSG:4326", "EPSG:4979"}:
            display_x, display_y = geographic_to_local_xy(
                values.x.values, values.y.values
            )
            horizontal_title = "local display offset (m)"
        else:
            display_x, display_y = values.x.values, values.y.values
            horizontal_title = "x / y"
        points = pv.PolyData(
            np.column_stack(
                [
                    display_x[ix],
                    display_y[iy],
                    height * settings.vertical_exaggeration,
                ]
            )
        )
        points["height"] = height
        points["vertical_contribution"] = contribution
        plotter = pv.Plotter(notebook=True, window_size=(1100, 750))
        plotter.set_background(settings.background)
        plotter.add_mesh(
            points,
            scalars="height",
            cmap=settings.cmap,
            opacity=opacity,
            point_size=settings.point_size,
            render_points_as_spheres=True,
        )
        plotter.show_grid(
            xtitle=horizontal_title,
            ytitle=horizontal_title,
            ztitle=f"display height ({settings.vertical_exaggeration:g}×)",
        )
        plotter.camera_position = "iso"
        return plotter
