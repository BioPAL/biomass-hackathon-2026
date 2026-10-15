#!/usr/bin/env python3
"""Download water-level time series from DAHITI (https://dahiti.dgfi.tum.de/).

The API key is not stored in the repo: it is read from the credentials file
(DAHITI_API_KEY=..., see config.yaml credentials_path) or from the DAHITI_API_KEY
environment variable, or pass --api-key. Get a key after registering on the DAHITI
website. For a ROI-based figure see water_level_roi.py (also pipeline step 10).

Examples:
  python water_level_dahiti.py --dahiti-id 15307            # Rio Solimoes
  python water_level_dahiti.py --dahiti-id 15307 --format netcdf
"""

import argparse
import json
import pprint
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flooded_forest.core import auth, config as ffconfig

URL = "https://dahiti.dgfi.tum.de/api/v2/download-water-level/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dahiti-id", type=int, required=True, help="target id on DAHITI")
    parser.add_argument("--api-key", help="DAHITI API key (default: DAHITI_API_KEY in the "
                                          "credentials file or the environment)")
    parser.add_argument("--format", choices=["json", "ascii", "netcdf", "csv"], default="json")
    parser.add_argument("--config", help="alternative config.yaml")
    args = parser.parse_args()

    cfg = ffconfig.load_config(args.config)
    try:
        api_key = args.api_key or auth.dahiti_api_key(cfg.credentials_path)
    except ValueError as exc:
        raise SystemExit(str(exc))

    params = {"api_key": api_key, "dahiti_id": args.dahiti_id, "format": args.format}
    response = requests.get(URL, params=params)
    if response.status_code != 200:
        raise SystemExit(f"{response.status_code}: {response.text}")

    if args.format == "json":
        pprint.pprint(json.loads(response.text))
    elif args.format in ("ascii", "csv"):
        print(response.text)
    elif args.format == "netcdf":
        out = cfg.outputs_dir / "validation" / f"{args.dahiti_id}_water_level_altimetry.nc"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"written: {out}")


if __name__ == "__main__":
    main()
