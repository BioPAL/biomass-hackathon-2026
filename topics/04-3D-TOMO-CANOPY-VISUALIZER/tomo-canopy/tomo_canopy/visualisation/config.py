"""Display-only configuration for TOM visualisations."""

from __future__ import annotations

from dataclasses import dataclass, field

from matplotlib.colors import LinearSegmentedColormap


@dataclass(frozen=True)
class TomogramPlotConfig:
    """Configuration for static height-band maps."""

    cmap: str = "magma"
    columns: int = 3
    contribution_max: float | None = None
    figure_width: float = 13.0
    panel_height: float = 4.0


@dataclass(frozen=True)
class Tomogram3DConfig:
    """Display choices for a prepared vertical-contribution volume."""

    cmap: str = "terrain"
    contribution_floor: float = 0.02
    vertical_exaggeration: float = 12.0
    point_size: float = 9.0
    background: str = "#10151c"
    opacity_percentiles: tuple[float, float] = (10.0, 99.0)



def _height_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("trees-like", ["#8b4513", "green"])


def _energy_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("energy-like", ["white", "purple"])


@dataclass(frozen=True)
class InteractiveTomogramConfig:
    """Display and widget settings for the interactive TOM point viewer."""

    normalization_quantile: float = 0.95
    normalize_height_bins: bool = True
    filter_after_normalization: bool = False
    column_quantile: float = 0.80
    vertical_exaggeration: float = 12.0
    point_opacity: float = 0.35
    point_size: float = 15.0
    color_by_normalized_energy: bool = False
    column_quantile_range: tuple[float, float] = (0.0, 0.99)
    vertical_exaggeration_range: tuple[float, float] = (1.0, 40.0)
    point_opacity_range: tuple[float, float] = (0.02, 1.0)
    point_size_range: tuple[float, float] = (1.0, 40.0)
    window_size: tuple[int, int] = (1100, 750)
    background: str = "#eeeeee"
    height_cmap: LinearSegmentedColormap = field(default_factory=_height_cmap)
    energy_cmap: LinearSegmentedColormap = field(default_factory=_energy_cmap)
    normalized_energy_clim: tuple[float, float] = (0.0, 1.0)
    ground_color: str = "tan"
    ground_opacity: float = 0.35
    camera_position: str = "iso"


@dataclass(frozen=True)
class TomogramIsosurfaceConfig:
    """Display choices for nested surfaces of a prepared TOM variable."""

    levels: tuple[float, ...] = (-8.0, -4.0, -2.0)
    colors: tuple[str, ...] = ("#8c510a", "#dfc27d", "#fff7bc")
    opacities: tuple[float, ...] = (0.10, 0.35, 0.85)
    vertical_exaggeration: float = 12.0
    background: str = "#10151c"
    ground_color: str = "#b8a47c"
    ground_opacity: float = 0.08
    window_size: tuple[int, int] = (1100, 700)
