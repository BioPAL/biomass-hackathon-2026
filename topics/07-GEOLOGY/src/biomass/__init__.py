"""Search and download ESA Biomass products from the MAAP catalogue."""
from .auth import TokenProvider
from .catalog import BiomassClient, assets_table, items_table
from .download import Downloader
from .exceptions import AuthError, BiomassError, DownloadError, QueryError
from .query import SearchQuery
from .raster import inspect_raster
from .viz import footprint_map, plot_raster, raster_map

__all__ = [
    "TokenProvider", "BiomassClient", "SearchQuery", "Downloader",
    "items_table", "assets_table", "inspect_raster", "footprint_map", "plot_raster", "raster_map",
    "BiomassError", "AuthError", "DownloadError", "QueryError",
]
