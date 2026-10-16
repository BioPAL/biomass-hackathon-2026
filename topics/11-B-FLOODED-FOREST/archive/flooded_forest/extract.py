"""Stack extraction via the marcos_dev extractor, run as a subprocess (kept untouched)."""

import subprocess
import sys
from pathlib import Path

EXTRACT_SCRIPT = (
    Path(__file__).resolve().parent.parent / "marcos_dev" / "1_data_extraction" / "extract_l1c_roi.py"
)


def build_command(
    config,
    roi,
    start,
    end,
    token_file,
    stacks=None,
    yes=False,
    orbit=None,
    max_files=None,
    min_coverage=None,
    workers=None,
    dry_run=False,
):
    """Command line for marcos_dev/1_data_extraction/extract_l1c_roi.py."""
    cmd = [
        sys.executable,
        str(EXTRACT_SCRIPT),
        "--name", roi["name"],
        "--bbox", *[str(c) for c in roi["bbox"]],
        "--start", start,
        "--end", end,
        "--out-dir", str(config.stacks_dir),
        "--token-file", str(token_file),
    ]
    if stacks:
        cmd += ["--stacks", stacks]
    if yes:
        cmd.append("--yes")
    if orbit:
        cmd += ["--orbit", orbit]
    if max_files is not None:
        cmd += ["--max-files", str(max_files)]
    if min_coverage is not None:
        cmd += ["--min-coverage", str(min_coverage)]
    if workers is not None:
        cmd += ["--workers", str(workers)]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def run_extraction(
    config,
    roi,
    start,
    end,
    token_file,
    stacks=None,
    yes=False,
    orbit=None,
    max_files=None,
    min_coverage=None,
    workers=None,
    dry_run=False,
):
    """Run the extractor interactively in the terminal (stack selection happens there)."""
    cmd = build_command(
        config, roi, start, end, token_file,
        stacks=stacks, yes=yes, orbit=orbit, max_files=max_files,
        min_coverage=min_coverage, workers=workers, dry_run=dry_run,
    )
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return config.stacks_dir / roi["slug"]
