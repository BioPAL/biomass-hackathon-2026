"""Search and download ESA Biomass, NASA NISAR and Copernicus Sentinel-1/2 products."""
from .auth import CopernicusAuth, EarthdataToken, TokenProvider
from .catalog import BiomassClient, assets_table, items_table
from .download import Downloader
from .exceptions import AuthError, BiomassError, DownloadError, QueryError
from .nisar import NisarClient, gcov_map, inspect_gcov, nisar_items_table, plot_gcov, read_gcov
from .query import SearchQuery
from .raster import inspect_raster
from .sentinel import SentinelClient, optical_map, plot_optical, read_optical, sentinel_items_table
from .viz import footprint_map, plot_raster, raster_map

__all__ = [
    "TokenProvider", "BiomassClient", "SearchQuery", "Downloader",
    "items_table", "assets_table", "inspect_raster", "footprint_map", "plot_raster", "raster_map",
    "EarthdataToken", "NisarClient", "nisar_items_table", "inspect_gcov", "read_gcov", "plot_gcov", "gcov_map",
    "CopernicusAuth", "SentinelClient", "sentinel_items_table", "read_optical", "plot_optical", "optical_map",
    "BiomassError", "AuthError", "DownloadError", "QueryError",
]
