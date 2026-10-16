"""Command line: `tomgedi run <step|all> --config configs/local.yaml` (also `python -m biomass_tom_gedi ...`)."""
from __future__ import annotations

import argparse
import importlib
import logging
import sys

from .config import Config

STEPS = {
    "01": "s01_download_gedi", "02": "s02_extract_tom_profiles", "03": "s03_slope_mask", "04": "s04_aggregate_cells",
    "05": "s05_correlations", "06": "s06_ridge_cv", "07": "s07_filter_experiment", "08": "s08_plot_obs_vs_pred",
}
DEFAULT_ALL = ["02", "03", "04", "05", "06", "07", "08"]   # 01 needs network access and is optional


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tomgedi", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one or more steps")
    r.add_argument("steps", nargs="+", help=f"step numbers ({', '.join(STEPS)}) or 'all' (= {' '.join(DEFAULT_ALL)})")
    r.add_argument("--config", "-c", required=True, help="YAML config (see configs/lope.example.yaml)")
    sub.add_parser("list", help="list the steps")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if a.cmd == "list":
        for k, v in STEPS.items():
            mod = importlib.import_module(f"biomass_tom_gedi.steps.{v}")
            print(f"{k}  {v:28s} {(mod.__doc__ or '').strip().splitlines()[0]}")
        return 0
    cfg = Config.load(a.config)
    todo = DEFAULT_ALL if a.steps == ["all"] else [s.zfill(2) for s in a.steps]
    for s in todo:
        if s not in STEPS:
            p.error(f"unknown step {s!r}; choose from {list(STEPS)} or 'all'")
    for s in todo:
        logging.getLogger("tomgedi").info("=== step %s: %s ===", s, STEPS[s])
        importlib.import_module(f"biomass_tom_gedi.steps.{STEPS[s]}").run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
