"""Interactive PyVista views for already-prepared tomogram layers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import numpy as np

from tomo_canopy.transforms import geographic_to_local_xy
from tomo_canopy.types import DataCube
from tomo_canopy.visualisation.config import InteractiveTomogramConfig
from tomo_canopy.visualisation.tomograms import _one_polarization, _tomogram, _variable


@dataclass
class InteractiveTomogramViewer:
    """Interactive point rendering of a prepared TOM power volume.

    The complete selected volume is materialized once because PyVista callback
    widgets require in-memory arrays. Open a spatial and vertical subset before
    constructing this viewer. Scientific smoothing and power conversion must
    already have been performed by processors or transforms.
    """

    cube: DataCube
    layer: str
    polarization: str | None = None
    variable: str = "tomographic_power"
    config: InteractiveTomogramConfig = field(
        default_factory=InteractiveTomogramConfig
    )
    state: dict[str, Any] = field(init=False)
    _values: np.ndarray = field(init=False, repr=False)
    _z: np.ndarray = field(init=False, repr=False)
    _x: np.ndarray = field(init=False, repr=False)
    _y: np.ndarray = field(init=False, repr=False)
    _scale_cache: dict[bool, np.ndarray | float] = field(
        init=False, default_factory=dict, repr=False
    )
    _threshold_cache: dict[tuple[float, bool | None], np.ndarray] = field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        self._validate_config()
        dataset = _tomogram(self.cube, self.layer)
        source = _one_polarization(
            _variable(dataset, self.variable, ("z", "y", "x")),
            self.polarization,
        ).transpose("z", "y", "x")
        if np.issubdtype(source.dtype, np.complexfloating):
            raise ValueError(
                "InteractiveTomogramViewer requires power, not complex amplitude; "
                "run TomogramProcessor.to_power(...) first"
            )
        self._values = np.asarray(source.data, dtype=float)
        if not np.isfinite(self._values).any():
            raise ValueError(
                f"Tomogram variable {self.variable!r} has no finite values"
            )
        if np.nanmin(self._values) < 0:
            raise ValueError(
                "InteractiveTomogramViewer requires non-negative linear power, "
                "not decibels"
            )
        self._z = np.asarray(source.z.data, dtype=float)
        if str(dataset.attrs.get("crs", "")).upper() in {"EPSG:4326", "EPSG:4979"}:
            self._x, self._y = geographic_to_local_xy(
                source.x.values, source.y.values
            )
        else:
            self._x = np.asarray(source.x.data, dtype=float)
            self._y = np.asarray(source.y.data, dtype=float)
        settings = self.config
        self.state = {
            "normalize": settings.normalize_height_bins,
            "filter_after": settings.filter_after_normalization,
            "column_quantile": settings.column_quantile,
            "exaggeration": settings.vertical_exaggeration,
            "opacity": settings.point_opacity,
            "point_size": settings.point_size,
            "color_by_energy": settings.color_by_normalized_energy,
            "z_min": float(self._z.min()),
            "z_max": float(self._z.max()),
        }

    def _validate_config(self) -> None:
        settings = self.config
        for value, name in (
            (settings.normalization_quantile, "normalization_quantile"),
            (settings.column_quantile, "column_quantile"),
        ):
            if not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if settings.vertical_exaggeration <= 0:
            raise ValueError("vertical_exaggeration must be positive")
        if not 0 < settings.point_opacity <= 1:
            raise ValueError("point_opacity must be in (0, 1]")
        if settings.point_size <= 0:
            raise ValueError("point_size must be positive")

    def _bin_scale(self, volume: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scale = np.nanquantile(
                volume, self.config.normalization_quantile, axis=(1, 2)
            )
        return np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)

    def _scale(self, volume: np.ndarray, normalize: bool) -> np.ndarray | float:
        if normalize:
            return self._bin_scale(volume)[:, None, None]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scale = np.nanquantile(volume, self.config.normalization_quantile)
        return float(scale) if np.isfinite(scale) and scale > 0 else np.nan

    def _full_scale(self, normalize: bool) -> np.ndarray | float:
        if normalize not in self._scale_cache:
            self._scale_cache[normalize] = self._scale(self._values, normalize)
        return self._scale_cache[normalize]

    def _column_threshold(
        self, quantile: float, normalize_first: bool | None = None
    ) -> np.ndarray:
        key = (quantile, normalize_first)
        if key not in self._threshold_cache:
            source = self._values
            if normalize_first is not None:
                source = source / self._full_scale(normalize_first)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self._threshold_cache[key] = np.nanquantile(
                    source, quantile, axis=0
                )
        return self._threshold_cache[key]

    @staticmethod
    def _filter(volume: np.ndarray, threshold: np.ndarray) -> np.ndarray:
        return np.where(volume >= threshold[None, ...], volume, np.nan)

    def _energy(self, quantile: float, normalize: bool, filter_after: bool):
        if filter_after:
            energy = self._values / self._full_scale(normalize)
            threshold = self._column_threshold(quantile, normalize)
            return self._filter(energy, threshold)
        threshold = self._column_threshold(quantile)
        filtered = self._filter(self._values, threshold)
        return filtered / self._scale(filtered, normalize)

    def build(self, *, notebook: bool = True) -> Any:
        """Build and return the PyVista plotter with interactive controls."""
        try:
            pv = import_module("pyvista")
        except ImportError as error:
            raise ImportError(
                "Interactive TOM visualisation requires "
                "`pip install tomo-canopy[visualisation]`"
            ) from error

        settings = self.config
        plotter = pv.Plotter(notebook=notebook, window_size=settings.window_size)
        plotter.set_background(settings.background)
        plotter.add_mesh(
            pv.Plane(
                center=(0, 0, 0),
                i_size=float(np.ptp(self._x)),
                j_size=float(np.ptp(self._y)),
            ),
            color=settings.ground_color,
            opacity=settings.ground_opacity,
            name="ground",
        )

        def update_scene(reset_camera: bool = False) -> None:
            quantile = round(float(self.state["column_quantile"]), 2)
            energy = self._energy(
                quantile,
                bool(self.state["normalize"]),
                bool(self.state["filter_after"]),
            )
            z_ok = (self._z >= self.state["z_min"]) & (
                self._z <= self.state["z_max"]
            )
            keep = np.isfinite(energy) & z_ok[:, None, None]
            iz, iy, ix = np.where(keep)
            if iz.size:
                exaggeration = float(self.state["exaggeration"])
                points = pv.PolyData(
                    np.column_stack(
                        [self._x[ix], self._y[iy], self._z[iz] * exaggeration]
                    )
                )
                points["height (m)"] = self._z[iz]
                points["normalized energy"] = energy[iz, iy, ix]
                if self.state["color_by_energy"]:
                    scalars = "normalized energy"
                    cmap = settings.energy_cmap
                    clim = settings.normalized_energy_clim
                else:
                    scalars = "height (m)"
                    cmap = settings.height_cmap
                    clim = (float(self._z.min()), float(self._z.max()))
                for title in list(plotter.scalar_bars.keys()):
                    plotter.remove_scalar_bar(title)
                plotter.add_mesh(
                    points,
                    scalars=scalars,
                    cmap=cmap,
                    clim=clim,
                    render_points_as_spheres=True,
                    point_size=float(self.state["point_size"]),
                    opacity=float(self.state["opacity"]),
                    name="scatter",
                    reset_camera=reset_camera,
                    scalar_bar_args={"title": scalars},
                )
            else:
                plotter.remove_actor("scatter")
            pct = self.config.normalization_quantile * 100
            order = "after" if self.state["filter_after"] else "before"
            normalization = (
                f"height-bin P{pct:g}"
                if self.state["normalize"]
                else "scene P"
            )
            plotter.add_text(
                f"Top {(1 - quantile) * 100:.0f}% per column; filter {order} "
                f"{normalization} normalization — {iz.size:,} points",
                position="upper_edge",
                font_size=11,
                name="title",
            )
            plotter.remove_bounds_axes()
            exaggeration = float(self.state["exaggeration"])
            plotter.show_grid(
                xtitle="east / x",
                ytitle="north / y",
                ztitle=f"display height ({exaggeration:g}×)",
                bounds=(
                    float(self._x.min()),
                    float(self._x.max()),
                    float(self._y.min()),
                    float(self._y.max()),
                    float(self._z.min() * exaggeration),
                    float(self._z.max() * exaggeration),
                ),
            )
            plotter.render()

        def slider(key: str, title: str, limits, row: int, fmt="%.2f") -> None:
            def callback(value) -> None:
                self.state[key] = float(value)
                update_scene()

            y = 0.82 - row * 0.09
            plotter.add_slider_widget(
                callback,
                rng=limits,
                value=self.state[key],
                title=title,
                pointa=(0.02, y),
                pointb=(0.30, y),
                fmt=fmt,
                style="modern",
                interaction_event="end",
            )

        slider(
            "column_quantile",
            "Column quantile",
            settings.column_quantile_range,
            0,
        )
        slider(
            "exaggeration",
            "Vertical exaggeration",
            settings.vertical_exaggeration_range,
            1,
            "%.0f",
        )
        z_limits = (float(self._z.min()), float(self._z.max()))
        slider("z_min", "Min height (m)", z_limits, 2, "%.0f")
        slider("z_max", "Max height (m)", z_limits, 3, "%.0f")
        slider("opacity", "Point opacity", settings.point_opacity_range, 4)
        slider("point_size", "Point size", settings.point_size_range, 5, "%.0f")

        def toggle(key: str, label: str, row: int) -> None:
            def callback(flag) -> None:
                self.state[key] = bool(flag)
                update_scene()

            y = 15 + row * 40
            plotter.add_checkbox_button_widget(
                callback,
                value=bool(self.state[key]),
                position=(15, y),
                size=28,
                color_on="#e8a33d",
                color_off="#3a4250",
            )
            plotter.add_text(
                label, position=(55, y + 5), font_size=10, name=f"{key}_label"
            )

        toggle("color_by_energy", "Colour by normalized energy", 0)
        toggle("normalize", "Normalize by height bin", 1)
        toggle("filter_after", "Column filter after normalization", 2)
        update_scene(reset_camera=True)
        plotter.camera_position = settings.camera_position
        return plotter

    def show(self, *, notebook: bool = True, **kwargs: Any) -> Any:
        """Build and display the interactive viewer."""
        return self.build(notebook=notebook).show(**kwargs)
