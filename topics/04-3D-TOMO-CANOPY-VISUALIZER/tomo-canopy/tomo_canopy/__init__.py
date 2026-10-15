"""Tools for multi-scale LiDAR and BIOMASS tomogram visualisation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tomo-canopy")
except PackageNotFoundError:  # Package is being used directly from its source tree.
    __version__ = "0.1.0"

__all__ = ["__version__"]

