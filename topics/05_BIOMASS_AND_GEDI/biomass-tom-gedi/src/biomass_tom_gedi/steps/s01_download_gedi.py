"""Step 01 - download GEDI shots (L2A/L2B/L4A) for the TOM scene from gediDB, with the MAAP quality filters.

Skip this step if you already have a shot table: point `paths.gedi_table` at it (parquet or csv).
Output: data/interim/gedi_download.parquet
"""
import logging

from .. import gedi
from ..config import Config
from ..tom import TOMProduct

log = logging.getLogger(__name__)


def run(cfg: Config) -> None:
    bbox = cfg.get("gedi", "bbox")
    if not bbox:                                       # default: the footprint of the TOM scene itself
        bbox = TOMProduct(cfg["paths"]["tom_product"]).grid.bbox
    log.info("ROI (min_lon, min_lat, max_lon, max_lat) = %s", tuple(round(b, 4) for b in bbox))
    df = gedi.download(tuple(bbox), cfg.get("gedi", "start_time", default="2019-01-01"),
                       cfg.get("gedi", "end_time", default="2026-12-31"))
    out = cfg.interim_dir / "gedi_download.parquet"
    df.to_parquet(out, index=False)
    log.info("saved %s (%d shots)", out, len(df))
