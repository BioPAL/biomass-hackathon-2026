"""Manual scene selection of a stack, shared by every step after the coherence review.

The choice is stored next to stack_info.json as scene_selection.json and lists the
REMOVED products: a product extracted later (not listed) is kept by default.
core.io.load_stack_info(stack_dir, kept_only=True) applies it.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

SELECTION_FILE = "scene_selection.json"


def parse_selection(text, n):
    """'0,2,5-7' -> [0, 2, 5, 6, 7]; 'all' -> every index."""
    text = text.strip().lower()
    if text == "all":
        return list(range(n))
    chosen = set()
    for part in filter(None, re.split(r"[,\s]+", text)):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            raise ValueError(f"cannot read '{part}'")
        first, last = int(match[1]), int(match[2] or match[1])
        if not (0 <= first <= last < n):
            raise ValueError(f"'{part}' is outside 0-{n - 1}")
        chosen.update(range(first, last + 1))
    return sorted(chosen)


def format_selection(indices):
    """[0, 1, 2, 5] -> '0-2,5'"""
    parts, run = [], []
    for k in sorted(indices):
        if run and k == run[-1] + 1:
            run.append(k)
            continue
        if run:
            parts.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
        run = [k]
    if run:
        parts.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
    return ",".join(parts)


def removed_ids(stack_dir):
    """Product ids removed by the manual review (empty when no review was saved)."""
    path = Path(stack_dir) / SELECTION_FILE
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("removed", []))


def save_selection(stack_dir, products, kept_indices):
    """Store the review of a stack: products = date-sorted product dicts, kept_indices into them."""
    kept = [products[k]["id"] for k in kept_indices]
    removed = [p["id"] for k, p in enumerate(products) if k not in set(kept_indices)]
    (Path(stack_dir) / SELECTION_FILE).write_text(json.dumps({
        "kept": kept,
        "removed": removed,
        "kept_dates": [products[k]["datetime"][:10] for k in kept_indices],
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=1))
    return kept, removed
