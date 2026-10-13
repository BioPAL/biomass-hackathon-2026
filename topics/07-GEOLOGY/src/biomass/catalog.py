"""Read-only access to the MAAP STAC catalogue."""
import pandas as pd
import requests
from pystac_client import Client

from . import config
from .auth import TokenProvider
from .query import SearchQuery


class BiomassClient:
    """Discover collections and queryables, and search for Biomass products."""

    def __init__(self, auth: TokenProvider = None, catalog_url: str = config.CATALOG_URL):
        self.auth = auth or TokenProvider()
        self.catalog_url = catalog_url
        self._stac = None

    @property
    def stac(self) -> Client:
        if self._stac is None:
            self._stac = Client.open(self.catalog_url)
        return self._stac

    def collections(self, since: str = "2025-12-01") -> pd.DataFrame:
        res = self.stac.collection_search(filter="platform='Biomass'", datetime=f"{since}T00:00:00.000Z/..")
        cols = pd.json_normalize(res.collection_list_as_dict(), record_path=["collections"])
        return cols.sort_values("title", ignore_index=True)[["id", "title", "extent.temporal.interval"]]

    def queryables(self, collection: str) -> pd.DataFrame:
        r = requests.get(f"{self.catalog_url}collections/{collection}/queryables", timeout=60)
        r.raise_for_status()
        rows = [{"property": k, "type": v.get("type"),
                 "values": ", ".join(map(str, v["enum"])) if "enum" in v else "",
                 "description": v.get("description", "")[:80]}
                for k, v in r.json().get("properties", {}).items()]
        return pd.DataFrame(rows)

    def search(self, query: SearchQuery) -> list:
        search = self.stac.search(
            collections=[query.collection],
            filter=query.cql_filter(),
            datetime=query.datetime_range(),
            bbox=query.resolved_bbox(),
            method="GET",
            max_items=query.max_items,
        )
        items = list(search.items())
        print(f"{len(items)} products fetched | {search.matched()} match in total")
        return items


def items_table(items) -> pd.DataFrame:
    rows = []
    for i, it in enumerate(items):
        p = it.properties
        rows.append({"idx": i, "id": it.id, "start": p.get("start_datetime"),
                     "orbit": p.get("sat:orbit_state"), "type": p.get("product:type"),
                     "assets": len(it.assets)})
    return pd.DataFrame(rows)


def assets_table(item) -> pd.DataFrame:
    """Assets of one item, to choose which ones to download."""
    return pd.DataFrame([{"asset": k, "media_type": a.media_type, "href": a.href}
                         for k, a in item.assets.items()])
