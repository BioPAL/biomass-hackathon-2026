#!/usr/bin/env python3
"""Run the flooded-forest pipeline for one ROI.

Steps (everything lands in outputs/, see config.yaml)
-----
1. resolve ROI: preset name from rois.json, 4 coordinates, or interactive prompt
2. search the MAAP STAC catalogue          -> outputs/catalogue/sta_level1c_by_roi.json
3. optional Pauli quicklooks (--quicklook)  -> outputs/<roi>/quicklooks/catalogue/
4. stack selection (interactive) + extraction, SKP + flattening applied
                                            -> outputs/<roi>/extraction/
5. stack quicklooks                         -> outputs/<roi>/quicklooks/<stack>/
6. MBMP coherence                           -> outputs/<roi>/coherence/<stack>/
7. coherence figures + MANUAL SCENE REVIEW: per stack the MBMP matrix and the HH, HV, VV
   pair grids open in windows and the scenes to keep are typed in the terminal
                                            -> outputs/<roi>/extraction/<stack>/scene_selection.json
8. HH / time-series figures of the kept scenes
                                            -> outputs/<roi>/extraction/
9. PolSAR of the kept scenes (PolSARpro T3, Pauli, HSI, H/A/alpha_averaged) on the same averaging
   window/grid + per-scene figures, chronological stack comparison, mechanism time series
                                            -> outputs/<roi>/polsar/<stack>/

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

from flooded_forest.catalogue import quicklook, search
from flooded_forest.core import auth, config, roi as roi_mod
from flooded_forest.core.io import load_stack_info


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
    parser.add_argument("--end", help="end date YYYY-MM-DD (default: today)")
    parser.add_argument("--quicklook", action="store_true", help="save Pauli quicklooks of every catalogue product")
    # steps to skip
    parser.add_argument("--skip-search", action="store_true", help="reuse the existing catalogue JSON")
    parser.add_argument("--skip-extract", action="store_true", help="reuse the extracted stacks")
    parser.add_argument("--skip-coherence", action="store_true", help="skip the MBMP coherence and its figures")
    parser.add_argument("--skip-review", action="store_true",
                        help="no manual scene review: reuse the saved selection (all scenes the first time)")
    parser.add_argument("--skip-polsar", action="store_true", help="skip the PolSAR decompositions and their figures")
    parser.add_argument("--skip-plots", action="store_true", help="compute only, no figures")
    parser.add_argument("--force", action="store_true", help="recompute coherence / PolSAR even if up to date")
    # spatial averaging (coherence + PolSAR)
    parser.add_argument("--window", type=float,
                        help="averaging window side on ground [m] for coherence and PolSAR "
                             "(default: averaging_window in config.yaml)")
    # extraction
    parser.add_argument("--stacks", help='stack indices to keep, e.g. "0,2,5-7" or "all"')
    parser.add_argument("--yes", action="store_true",
                        help="never ask: keep the recommended stacks and the saved scene selection")
    parser.add_argument("--orbit", choices=["both", "ascending", "descending"], default="both")
    parser.add_argument("--swaths", nargs="+", default=["S1", "S2", "S3"])
    parser.add_argument("--max-files", type=int, help="max number of products; whole stacks are taken")
    parser.add_argument("--min-coverage", type=float, help="drop stacks covering less than this ROI fraction")
    parser.add_argument("--workers", type=int, help="parallel download threads")
    parser.add_argument("--dry-run", action="store_true", help="stop after the common area and windows")
    # output
    parser.add_argument("--show", action="store_true", help="display the figures interactively")
    parser.add_argument("--config", help="alternative config.yaml")
    return parser.parse_args(argv)


def load_roi_entry(cfg, roi, skip_search, token):
    """Catalogue entry of the ROI (search, or the saved JSON with --skip-search)."""
    catalogue_path = cfg.catalogue_dir / search.CATALOGUE_FILENAME
    if not skip_search:
        return search.run_search(cfg, roi, token)
    if not catalogue_path.exists():
        print(f"No catalogue JSON ({catalogue_path}): quicklooks skipped")
        return None
    data = json.loads(catalogue_path.read_text(encoding="utf-8"))
    match = next((r for r in data["rois"] if r["name"] == roi["name"]), None)
    if match is None:
        print(f"ROI '{roi['name']}' not in {catalogue_path}: quicklooks skipped")
        return None
    match["slug"] = roi["slug"]
    return match


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
    paths = cfg.roi_paths(roi["slug"])
    window = args.window if args.window is not None else cfg.averaging_window
    show = args.show and sys.stdin.isatty()
    print(f"ROI: {roi['name']}  bbox {roi['bbox']}  ->  {paths.root}")

    token = auth.load_token(cfg)
    try:
        token.get()  # fail early on a bad token
    except Exception as exc:
        print(f"WARNING: MAAP token check failed ({exc}); asset downloads may not work")

    # 2. catalogue search  /  3. catalogue quicklooks
    roi_entry = load_roi_entry(cfg, roi, args.skip_search, token)
    if args.quicklook and roi_entry:
        quicklook.show_quicklooks(cfg, roi_entry, token, show=show)

    # 4. stack selection + calibrated extraction
    if not args.skip_extract:
        from flooded_forest.extraction.extract import run_extraction

        run_extraction(
            cfg, roi, args.start, args.end or str(date.today()), token, paths,
            stacks=args.stacks, yes=args.yes, orbit=args.orbit, max_files=args.max_files,
            min_coverage=args.min_coverage, workers=args.workers, dry_run=args.dry_run, swaths=args.swaths,
        )
        if args.dry_run:
            return
    if not paths.stack_dirs():
        print(f"No extracted stacks in {paths.extraction}: run without --skip-extract first")
        return

    # 5. stack quicklooks
    if roi_entry:
        print("Stack quicklooks:")
        for stack_dir in paths.stack_dirs():
            quicklook.quicklooks_for_stack(cfg, roi_entry, load_stack_info(stack_dir), token)

    # 6. MBMP coherence  /  7. coherence figures + manual scene review
    review = not (args.skip_review or args.yes) and sys.stdin.isatty()
    if not args.skip_coherence:
        from flooded_forest.plotting.coherence import plot_coherence_outputs
        from flooded_forest.processing.coherence import run_coherence

        run_coherence(paths, window, force=args.force)
        if review or not args.skip_plots:
            plot_coherence_outputs(paths, window, review=review, show=show)

    # 8. stack figures of the kept scenes
    if not args.skip_plots:
        from flooded_forest.plotting.stacks import plot_stack_outputs

        print("Stack figures:")
        plot_stack_outputs(paths, show=show)

    # 9. PolSAR of the kept scenes on the same averaging window and grid (+ figures)
    if not args.skip_polsar:
        from flooded_forest.plotting.polsar import plot_polsar_outputs
        from flooded_forest.processing.polsar import run_polsar

        run_polsar(cfg, paths, window, force=args.force)
        if not args.skip_plots:
            plot_polsar_outputs(paths, window, show=show)


if __name__ == "__main__":
    main()
