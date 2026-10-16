"""Download the GEDI shots of every site in config/sites.json from the public gediDB (no login needed).

Usage, from the topic folder::

    python scripts/fetch_gedi.py                 # all sites
    python scripts/fetch_gedi.py --sites lope    # one site
    python scripts/fetch_gedi.py --overwrite     # download again

Writes data/<site>/gedi_<site>.parquet and a .query.json provenance file next to it.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load.gedi import fetch_gedi  # noqa: E402


def main() -> None:
    cfg = json.loads((ROOT / "config" / "sites.json").read_text())
    parser = argparse.ArgumentParser(description="Download GEDI shots for the sites in config/sites.json.")
    parser.add_argument("--sites", nargs="*", default=list(cfg["sites"]), help="site keys (default: all)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for key in args.sites:
        site = cfg["sites"][key]
        out = ROOT / "data" / key / f"gedi_{key}.parquet"
        if out.exists() and not args.overwrite:
            print(f"{site['name']}: already downloaded ({out.relative_to(ROOT)})")
            continue
        _, audit = fetch_gedi(
            site["bbox"],
            out,
            cfg["gedi_start"],
            cfg["gedi_end"],
            sensitivity_min=cfg["gedi_sensitivity_min"],
            overwrite=args.overwrite,
        )
        print(f"{site['name']}: {audit['n_returned']:,} GEDI shots -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
