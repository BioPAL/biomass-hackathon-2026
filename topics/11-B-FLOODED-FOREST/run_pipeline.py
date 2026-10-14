#!/usr/bin/env python3
"""Run the flooded-forest pipeline for one ROI.

Steps
-----
1. resolve ROI: preset name from rois.json, 4 coordinates, or interactive prompt
2. search the MAAP STAC catalogue        -> assets/catalogue/sta_level1c_by_roi.json
3. optional Pauli quicklooks             -> assets/stacks/<roi>/quicklooks/
4. stack selection + extraction (marcos_dev, interactive) -> assets/stacks/<roi>/
5. QGIS dB rasters, per-stack quicklooks and figures -> assets/stacks/<roi>/

Examples
--------
  python run_pipeline.py --roi "Lac Télé Community Reserve (LTCR)" --start 2026-05-01 --end 2026-10-13
  python run_pipeline.py --bbox 16.98 0.94 17.48 1.44 --name "Lac Tele" --start 2026-05-01 --end 2026-10-13
  python run_pipeline.py --roi bolobo --start 2026-05-01 --skip-search --skip-extract --show
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flooded_forest import auth, config, extract, plots, qgis, quicklook, roi as roi_mod, search


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ROI
    roi_group = parser.add_argument_group("ROI (one of)")
    roi_group.add_argument("--roi", help="ROI name from rois.json (substring match, see --list-rois)")
    roi_group.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                           help="custom ROI as 4 coordinates")
    roi_group.add_argument("--name", help="name of a custom --bbox (also the output folder name)")
    parser.add_argument("--list-rois", action="store_true", help="print the preset ROIs and exit")
    # catalogue
    parser.add_argument("--start", help="start date YYYY-MM-DD")
    parser.add_argument("--end", help="end date YYYY-MM-DD")
    parser.add_argument("--quicklook", action="store_true", help="save Pauli quicklooks of the scene")
    parser.add_argument("--pauli", action="store_true",
                        help="Pauli plots from one catalogue product (remote COG tiles, no extraction needed)")
    # steps to skip
    parser.add_argument("--skip-search", action="store_true", help="reuse the existing catalogue JSON")
    parser.add_argument("--skip-extract", action="store_true", help="skip the stack extraction")
    parser.add_argument("--skip-plots", action="store_true", help="skip the figures")
    # extraction passthrough (marcos_dev/1_data_extraction/extract_l1c_roi.py)
    parser.add_argument("--stacks", help='stack indices to keep, e.g. "0,2,5-7" or "all"')
    parser.add_argument("--yes", action="store_true", help="keep the recommended stacks without asking")
    parser.add_argument("--orbit", choices=["both", "ascending", "descending"])
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--min-coverage", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true", help="extractor: stop before downloading")
    # output
    parser.add_argument("--show", action="store_true", help="display the figures interactively")
    parser.add_argument("--config", help="alternative config.yaml")
    return parser.parse_args(argv)


def post_extract(cfg, roi_entry, roi_dir, token):
    """Per-stack outputs: QGIS dB rasters, quicklooks and figures."""
    print("QGIS-friendly dB rasters:")
    qgis.convert_roi(roi_dir)

    print("Stack quicklooks:")
    for info_path in sorted(roi_dir.glob("*/stack_info.json")):
        stack_info = json.loads(info_path.read_text())
        stack_info["_dir"] = info_path.parent
        quicklook.quicklooks_for_stack(cfg, roi_entry, stack_info, token)


def main(argv=None):
    args = parse_args(argv)
    cfg = config.load_config(args.config)
    cfg.ensure_dirs()

    if args.list_rois:
        roi_mod.list_rois(roi_mod.load_rois(cfg.roi_json_path))
        return

    if not args.start:
        raise SystemExit("--start (YYYY-MM-DD) is required")
    if not (args.roi or args.bbox):
        roi = roi_mod.resolve_roi(roi_json_path=cfg.roi_json_path)
    else:
        roi = roi_mod.resolve_roi(name=args.name or args.roi, bbox=args.bbox, roi_json_path=cfg.roi_json_path)
    print(f"ROI: {roi['name']}  bbox {roi['bbox']}")

    token = auth.load_token(cfg)
    try:
        token.get()  # fail early on a bad token
    except Exception as exc:
        print(f"WARNING: MAAP token check failed ({exc}); asset downloads may not work")

    # 2. catalogue search
    catalogue_path = cfg.catalogue_dir / search.CATALOGUE_FILENAME
    if args.skip_search and catalogue_path.exists():
        data = json.loads(catalogue_path.read_text(encoding="utf-8"))
        matches = [r for r in data["rois"] if r["name"] == roi["name"]]
        if not matches:
            raise SystemExit(f"ROI '{roi['name']}' not in {catalogue_path}; run the search first")
        roi_entry = matches[0]
        roi_entry["slug"] = roi["slug"]
    else:
        roi_entry = search.run_search(cfg, roi, token)

    # 3. quicklooks
    if args.quicklook:
        quicklook.show_quicklooks(cfg, roi_entry, token, show=args.show and sys.stdin.isatty())
    if args.pauli:
        plots.plot_catalogue_pauli(cfg, roi_entry, token, show=args.show and sys.stdin.isatty())

    # 4. stack selection + extraction (interactive, in marcos_dev)
    roi_dir = None
    if not args.skip_extract:
        roi_dir = extract.run_extraction(
            cfg, roi, args.start, args.end or str(date.today()),
            cfg.credentials_path,
            stacks=args.stacks, yes=args.yes, orbit=args.orbit,
            max_files=args.max_files if args.max_files is not None else cfg.max_files,
            min_coverage=args.min_coverage if args.min_coverage is not None else cfg.min_coverage,
            workers=args.workers if args.workers is not None else cfg.workers,
            dry_run=args.dry_run,
        )

    # 5. per-stack outputs + plots from the extracted stacks
    if args.skip_plots or args.dry_run:
        return
    if roi_dir is None:
        candidates = sorted(cfg.stacks_dir.glob(f"{roi['slug']}*/extraction_summary.json"))
        if not candidates:
            print("No extracted stacks found for this ROI, skipping plots")
            return
        roi_dir = candidates[-1].parent
    post_extract(cfg, roi_entry, roi_dir, token)
    print(f"Plotting from: {roi_dir}")
    plots.plot_stack_outputs(cfg, roi_dir, show=args.show and sys.stdin.isatty())


if __name__ == "__main__":
    main()
