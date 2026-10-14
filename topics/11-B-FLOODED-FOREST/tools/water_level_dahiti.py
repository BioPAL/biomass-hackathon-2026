#!/usr/bin/env python3
"""Download water-level time series from DAHITI (https://dahiti.dgfi.tum.de/).

The API key is not stored in the repo: set DAHITI_API_KEY in the environment
or pass --api-key. Get a key after registering on the DAHITI website.

Examples:
  python water_level_dahiti.py --dahiti-id 15307            # Rio Solimoes
  python water_level_dahiti.py --dahiti-id 15307 --format netcdf
"""

import argparse
import json
import os
import pprint
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flooded_forest.core import config as ffconfig

URL = "https://dahiti.dgfi.tum.de/api/v2/download-water-level/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dahiti-id", type=int, required=True, help="target id on DAHITI")
    parser.add_argument("--api-key", default=os.environ.get("DAHITI_API_KEY"))
    parser.add_argument("--format", choices=["json", "ascii", "netcdf", "csv"], default="json")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Set DAHITI_API_KEY (or pass --api-key); register on dahiti.dgfi.tum.de")

    params = {"api_key": args.api_key, "dahiti_id": args.dahiti_id, "format": args.format}
    response = requests.get(URL, params=params)
    if response.status_code != 200:
        raise SystemExit(f"{response.status_code}: {response.text}")

    if args.format == "json":
        pprint.pprint(json.loads(response.text))
    elif args.format in ("ascii", "csv"):
        print(response.text)
    elif args.format == "netcdf":
        out = ffconfig.load_config().outputs_dir / "validation" / f"{args.dahiti_id}_water_level_altimetry.nc"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"written: {out}")


if __name__ == "__main__":
    main()
