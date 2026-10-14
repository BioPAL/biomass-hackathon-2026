#!/usr/bin/env python3
"""
Biomass Data Access Script
==========================
Query the ESA MAAP STAC catalog for Biomass products and save the quicklook image.

Usage:
    python biomass_query.py --level 1a --bbox -81.0 -20.0 -35.0 10.0
    python biomass_query.py --level 2a --bbox -10.0 5.0 15.0 20.0 --output my_quicklook.jpg
    python biomass_query.py --level 1b --bbox -81.0 -20.0 -35.0 10.0 --max-items 5 --index 2

Arguments:
    --level     Biomass processing level: 0, 1a, 1b, 1c, 2a, 2b (default: 1a)
    --bbox      Bounding box as four floats: west south east north
    --output    Output image filename (default: quicklook.jpg)
    --max-items Maximum number of catalog items to retrieve (default: 10)
    --index     Index of the product to preview from the results (default: 0)
    --start     Start date in YYYY-MM-DD format (default: 2026-01-01)
    --end       End date in YYYY-MM-DD format (default: 2026-07-01)
"""

import argparse
import sys

import requests
from pystac_client import Client
from PIL import Image
from io import BytesIO


CATALOG_URL = "https://catalog.maap.eo.esa.int/catalogue/"

LEVEL_TO_COLLECTION = {
    "0":  "BiomassLevel0",
    "1a": "BiomassLevel1a",
    "1b": "BiomassLevel1b",
    "1c": "BiomassLevel1c",
    "2a": "BiomassLevel2a",
    "2b": "BiomassLevel2b",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query ESA MAAP Biomass catalog and save a quicklook image."
    )
    parser.add_argument(
        "--level",
        type=str,
        default="1a",
        choices=LEVEL_TO_COLLECTION.keys(),
        help="Biomass processing level (default: 1a)",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Bounding box: west south east north",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="quicklook.jpg",
        help="Output image filename (default: quicklook.jpg)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Max items to retrieve from catalog (default: 10)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the product to preview (default: 0)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2026-01-01",
        help="Start date YYYY-MM-DD (default: 2026-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-07-01",
        help="End date YYYY-MM-DD (default: 2026-07-01)",
    )
    return parser.parse_args()


def query_catalog(collection, bbox, start, end, max_items):
    """Search the ESA MAAP STAC catalog and return matching items."""
    print(f"Connecting to catalog: {CATALOG_URL}")
    catalog = Client.open(CATALOG_URL)

    datetime_range = f"{start}T00:00:00Z/{end}T23:59:59Z"

    print(f"Searching collection '{collection}' ...")
    print(f"  BBOX:     {bbox}")
    print(f"  Datetime: {datetime_range}")
    print(f"  Max items: {max_items}")

    search = catalog.search(
        collections=[collection],
        datetime=datetime_range,
        bbox=bbox,
        method="GET",
        max_items=max_items,
    )

    items = list(search.items())
    matched = search.matched()
    print(f"  => {len(items)} items retrieved ({matched} total matched).")
    return items


def list_items(items):
    """Print a summary table of the retrieved items."""
    print(f"\n{'Idx':<5} {'Product ID':<80} {'Datetime'}")
    print("-" * 120)
    for i, item in enumerate(items):
        dt = item.properties.get("start_datetime", str(item.datetime))
        print(f"{i:<5} {item.id:<80} {dt}")
    print()


def fetch_quicklook(item, output_path):
    """Download the quicklook image for a given item and save it."""
    # Try several possible asset keys for the quicklook
    ql_keys = ["quicklook_jpg", "thumbnail", "enclosure_ql_pauli_png", "enclosure_ql_hsv_png"]
    ql_url = None
    for key in ql_keys:
        if key in item.assets:
            ql_url = item.assets[key].href
            print(f"Using asset '{key}' for quicklook.")
            break

    if ql_url is None:
        print("ERROR: No quicklook asset found for this product.")
        print("Available assets:", list(item.assets.keys()))
        sys.exit(1)

    print(f"Downloading quicklook from:\n  {ql_url}")
    resp = requests.get(ql_url, timeout=60)
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content))
    img.save(output_path)
    print(f"Quicklook saved to: {output_path}  ({img.size[0]}x{img.size[1]} px)")


def main():
    args = parse_args()

    collection = LEVEL_TO_COLLECTION[args.level]

    # Query the catalog
    items = query_catalog(collection, args.bbox, args.start, args.end, args.max_items)

    if not items:
        print("No items found for the given search parameters.")
        sys.exit(1)

    # List results
    list_items(items)

    # Validate index
    if args.index < 0 or args.index >= len(items):
        print(f"ERROR: --index {args.index} is out of range (0-{len(items) - 1}).")
        sys.exit(1)

    chosen = items[args.index]
    print(f"Selected product (index {args.index}): {chosen.id}")
    print(f"  Datetime: {chosen.datetime}")
    print(f"  BBOX:     {chosen.bbox}")

    # Fetch and save the quicklook
    fetch_quicklook(chosen, args.output)


if __name__ == "__main__":
    main()