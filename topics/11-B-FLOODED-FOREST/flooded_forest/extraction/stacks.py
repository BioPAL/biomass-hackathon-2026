"""STA stacks over a ROI: catalogue search, completion and the recommended selection.

Products are grouped by stack (``eofeos:stack_id``) and every stack is completed
with all its members (the date filter may cut a stack in half). Recommendation:
one frame per track (when a ROI falls across two adjacent frames of the same
datatake, the one covering most of the ROI), stacks below the minimum ROI
coverage dropped, then whole stacks (best coverage first) up to max_files.
"""

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from shapely.geometry import box, shape

log = logging.getLogger("flooded_forest.extraction")

# Stacks of the same swath/orbit/cycle starting within this time are adjacent frames of the same datatake
SAME_TRACK_SECONDS = 300


def item_version(item):
    return (item.properties.get("version") or "", item.properties.get("created") or "")


def stack_folder_name(stack):
    return "STA{}_{}_{}-{}_{}".format(
        stack["swath"][1], stack["orbit"],
        stack["start"].strftime("%Y%m%d"), stack["end"].strftime("%Y%m%d"), stack["grid"],
    )


def search_stacks(catalog, collection, bbox, start, end, swaths=("S1", "S2", "S3"), orbit="both", workers=6):
    """STA stacks touching the bbox between start and end (YYYY-MM-DD), each with all its members."""
    flt = "(" + " or ".join(f"product:type='{s.upper()}_STA__1S'" for s in swaths) + ")"
    if orbit != "both":
        flt += f" and sat:orbit_state='{orbit}'"

    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start}T00:00:00Z/{end}T23:59:59Z",
        filter=flt,
        method="GET",
        limit=200,
    )
    hits = list(search.items())
    stack_ids = sorted({it.properties["eofeos:stack_id"] for it in hits})
    log.info("Search: %d products in %d stacks touch the bbox", len(hits), len(stack_ids))

    def complete(stack_id):
        members = catalog.search(
            collections=[collection],
            filter=f"eofeos:stack_id='{stack_id}'",
            method="GET",
            limit=50,
        )
        # one product per repeat cycle, newest version wins
        by_cycle = {}
        for it in members.items():
            cycle = int(it.properties["eofeos:repeat_cycle_id"])
            if cycle not in by_cycle or item_version(it) > item_version(by_cycle[cycle]):
                by_cycle[cycle] = it
        return stack_id, [by_cycle[c] for c in sorted(by_cycle)]

    roi = box(*bbox)
    stacks = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for stack_id, items in pool.map(complete, stack_ids):
            if not items:
                continue
            ref = next((it for it in items if it.properties.get("eofeos:is_coregistration_primary")), items[0])
            props = ref.properties
            times = sorted(datetime.fromisoformat(it.properties["datetime"].replace("Z", "+00:00")) for it in items)
            footprint = shape(ref.geometry)
            stacks.append({
                "stack_id": stack_id,
                "items": items,
                "reference": ref,
                "swath": props["product:type"][:2],
                "orbit": props["sat:orbit_state"],
                "grid": props.get("grid:code", "grid"),
                "major_cycle": props.get("eofeos:major_cycle_id"),
                "start": times[0],
                "end": times[-1],
                "stac_coverage": footprint.intersection(roi).area / roi.area,
                "stac_footprint": footprint,
            })
    for st in stacks:
        st["name"] = stack_folder_name(st)
    return sorted(stacks, key=lambda s: (s["orbit"], s["swath"], s["start"], s["grid"]))


def best_frame_per_track(stacks):
    """Of adjacent frames from the same datatake, keep the one covering most of the ROI."""
    groups = defaultdict(list)
    for st in stacks:
        groups[(st["swath"], st["orbit"], st["major_cycle"])].append(st)

    kept = []
    for members in groups.values():
        members.sort(key=lambda s: s["start"])
        cluster = [members[0]]
        for st in members[1:]:
            if (st["start"] - cluster[0]["start"]).total_seconds() <= SAME_TRACK_SECONDS:
                cluster.append(st)
            else:
                kept.append(max(cluster, key=lambda s: s["stac_coverage"]))
                cluster = [st]
        kept.append(max(cluster, key=lambda s: s["stac_coverage"]))
    return kept


def limit_files(stacks, max_files):
    """Take whole stacks (best ROI coverage first) until max_files is reached."""
    stacks = sorted(stacks, key=lambda s: (-s["stac_coverage"], s["start"]))
    chosen, total = [], 0
    for st in stacks:
        n = len(st["items"])
        if total + n > max_files:
            if not chosen:
                # limit smaller than one stack: keep a truncated first stack
                chosen.append(dict(st, items=st["items"][:max_files]))
            break
        chosen.append(st)
        total += n
    return sorted(chosen, key=lambda s: (s["orbit"], s["swath"], s["start"]))


def recommend(stacks, min_coverage, max_files):
    """Recommended stacks: best frame per track, min ROI coverage, whole stacks up to max_files."""
    kept = [st for st in best_frame_per_track(stacks) if st["stac_coverage"] >= min_coverage]
    return limit_files(kept, max_files) if kept else []
