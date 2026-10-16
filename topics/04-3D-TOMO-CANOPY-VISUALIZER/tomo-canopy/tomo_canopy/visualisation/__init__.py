"""Visualisation tools that consume prepared :class:`DataCube` layers.

Matplotlib and PyVista are imported only when a plot is requested. Scientific
derivations belong in processors, not in this package.
"""

from .config import (
    InteractiveTomogramConfig,
    Tomogram3DConfig,
    TomogramIsosurfaceConfig,
    TomogramPlotConfig,
)
from .interactive import InteractiveTomogramViewer
from .tomograms import (
    CanopyViewer,
    plot_tomogram_bands,
    plot_tomogram_metrics,
    plot_tomogram_vertical_slice,
)

__all__ = [
    "CanopyViewer",
    "InteractiveTomogramConfig",
    "InteractiveTomogramViewer",
    "Tomogram3DConfig",
    "TomogramIsosurfaceConfig",
    "TomogramPlotConfig",
    "plot_tomogram_bands",
    "plot_tomogram_metrics",
    "plot_tomogram_vertical_slice",
    "TomographicPlotter",
]


def __getattr__(name: str):
    """Keep the legacy PyVista application available without eager imports."""
    if name == "TomographicPlotter":
        from .tomographic_plotter import TomographicPlotter

        return TomographicPlotter
    raise AttributeError(name)
