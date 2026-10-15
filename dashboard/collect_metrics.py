#!/usr/bin/env python3
"""Collect hackathon metrics from the GitHub API → metrics.json for the dashboard."""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

OWNER = os.environ.get("GITHUB_REPOSITORY_OWNER", "BioPAL")
REPO = os.environ.get("GITHUB_REPOSITORY", "BioPAL/biomass-hackathon-2026").split("/")[-1]
TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
OUT = Path(__file__).resolve().parent / "metrics.json"
ROOT = Path(__file__).resolve().parents[1]

GROUPS = [
    "02-RFI",
    "03-QGIS",
    "04-3D-TOMO-CANOPY-VISUALIZER",
    "05_BIOMASS_AND_GEDI",
    "06A-CARBON",
    "06B-CARBON",
    "07-GEOLOGY",
    "08-ICE-QUEEN",
    "09-BIODIVERSITY",
    "10-TOMO-SAR-FOREST-HEIGHT",
    "11-A-SOIL-MOISTURE",
    "11-B-FLOODED-FOREST",
]
GROUP_SET = set(GROUPS)
# Tag when most added lines sit outside topics/ (docs, CI, dashboard, …)
REPO_TAG = "repo"
# Chosen team names (display); metrics keys stay as GROUPS / labels
GROUP_NAMES = {
    "02-RFI": "The Cleaning Guy",
    "03-QGIS": "SARcastic",
    "04-3D-TOMO-CANOPY-VISUALIZER": "Eye in the Sub",
    "05_BIOMASS_AND_GEDI": "The Canopy Matchmakers",
    "06A-CARBON": "Abovegrounders",
    "06B-CARBON": "Pikachu Group",
    "07-GEOLOGY": "Arrakis",
    "08-ICE-QUEEN": "Ice Queens",
    "09-BIODIVERSITY": "The Green Queens & Servant",
    "10-TOMO-SAR-FOREST-HEIGHT": "TomoSAR group",
    "11-A-SOIL-MOISTURE": "Soil moisture",
    "11-B-FLOODED-FOREST": "Flooded forests",
}
# Live board charts only cover the on-site hackathon window (hourly, local ESOC time)
CHART_START = date(2026, 10, 13)
CHART_END = date(2026, 10, 16)
CHART_TZ = ZoneInfo("Europe/Berlin")


def hour_key(iso: str) -> str | None:
    """Bucket an API timestamp into local YYYY-MM-DDTHH within the chart window."""
    if not iso:
        return None
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(CHART_TZ)
    if dt.date() < CHART_START or dt.date() > CHART_END:
        return None
    return dt.strftime("%Y-%m-%dT%H")


def chart_hours() -> list[str]:
    hours: list[str] = []
    cur = datetime(CHART_START.year, CHART_START.month, CHART_START.day, tzinfo=CHART_TZ)
    end = datetime(CHART_END.year, CHART_END.month, CHART_END.day, 23, tzinfo=CHART_TZ)
    while cur <= end:
        hours.append(cur.strftime("%Y-%m-%dT%H"))
        cur += timedelta(hours=1)
    return hours

# Folder under topics/ may match label, or legacy aliases
FOLDER_ALIASES = {
    "02-RFI": ["02-RFI", "02-rfi-removal"],
    "06B-CARBON": ["06B-CARBON", "06-biomass-validation"],
}

# Bulk data dumps inflate "lines added" without reflecting real work.
# PRs / issues still count; only LOC scoring and race charts skip these paths.
SKIP_LOC_SUFFIXES = (
    ".csv",
    ".tsv",
    ".parquet",
    ".feather",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".nc",
    ".grib",
    ".zip",
    ".gz",
    ".xz",
    ".7z",
)


def counts_toward_loc(filename: str) -> bool:
    """False for data dumps we do not want in line-score / race charts."""
    lower = (filename or "").lower()
    return not any(lower.endswith(suf) for suf in SKIP_LOC_SUFFIXES)


def topic_group_for_path(filename: str) -> str | None:
    """Return group id if path is under topics/<group>/, else None."""
    for g in GROUPS:
        folders = FOLDER_ALIASES.get(g, [g])
        if any(filename.startswith(f"topics/{folder}/") for folder in folders):
            return g
    return None


def api(path: str, params: dict | None = None):
    qs = f"?{urllib.parse.urlencode(params)}" if params else ""
    url = f"https://api.github.com{path}{qs}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            **({"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}),
            "User-Agent": "biomass-hackathon-dashboard",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp), dict(resp.headers)


def api_paginated(path: str, params: dict | None = None):
    params = dict(params or {})
    params.setdefault("per_page", 100)
    page = 1
    items = []
    while True:
        params["page"] = page
        batch, headers = api(path, params)
        if not batch:
            break
        items.extend(batch)
        link = headers.get("Link") or headers.get("link") or ""
        if 'rel="next"' not in link:
            break
        page += 1
        if page > 50:
            break
    return items


def group_labels(label_objs) -> list[str]:
    names = [l["name"] if isinstance(l, dict) else l for l in (label_objs or [])]
    return [n for n in names if n in GROUP_SET]


def collect_issues():
    # state=all includes open+closed; pull_request key distinguishes PRs
    raw = api_paginated(
        f"/repos/{OWNER}/{REPO}/issues",
        {"state": "all", "per_page": 100},
    )
    issues = [i for i in raw if "pull_request" not in i]
    by_group = {
        g: {"open": 0, "closed": 0, "closed_task": 0, "open_task": 0}
        for g in GROUPS
    }
    ungrouped = {"open": 0, "closed": 0}
    for issue in issues:
        labels = [l["name"] for l in issue.get("labels", [])]
        is_task = "task" in labels
        groups = group_labels(issue.get("labels", []))
        closed = issue.get("state") == "closed"
        if not groups:
            ungrouped["closed" if closed else "open"] += 1
            continue
        for g in groups:
            if closed:
                by_group[g]["closed"] += 1
                if is_task:
                    by_group[g]["closed_task"] += 1
            else:
                by_group[g]["open"] += 1
                if is_task:
                    by_group[g]["open_task"] += 1
    return issues, by_group, ungrouped


def collect_prs():
    pulls = api_paginated(
        f"/repos/{OWNER}/{REPO}/pulls",
        {"state": "all", "per_page": 100},
    )
    by_group = {
        g: {"merged": 0, "open": 0, "draft": 0, "additions": 0, "deletions": 0}
        for g in GROUPS
    }
    merged_total = 0
    open_total = 0
    contributors = defaultdict(
        lambda: {
            "merged_prs": 0,
            "additions": 0,
            "deletions": 0,
            "groups": defaultdict(int),  # group label -> merged PR count
            "group_additions": defaultdict(int),  # lines added under topics/<group>/
            "repo_additions": 0,  # lines outside topics/
        }
    )
    merged_events: list[dict] = []

    for pr in pulls:
        user = (pr.get("user") or {}).get("login") or "unknown"
        groups = group_labels(pr.get("labels", []))
        if pr.get("merged_at"):
            merged_total += 1
            contributors[user]["merged_prs"] += 1
            try:
                files = api_paginated(
                    f"/repos/{OWNER}/{REPO}/pulls/{pr['number']}/files"
                )
            except urllib.error.HTTPError:
                files = []

            hit = set(groups)
            path_groups: set[str] = set()
            by_group_adds: dict[str, int] = defaultdict(int)
            by_group_dels: dict[str, int] = defaultdict(int)
            add = dele = 0
            for f in files:
                fn = f.get("filename", "")
                file_add = int(f.get("additions") or 0)
                file_del = int(f.get("deletions") or 0)
                g = topic_group_for_path(fn)
                if g:
                    path_groups.add(g)
                # Topic touch still counts for merged-PR attribution;
                # only LOC / race scoring skips CSV and other data dumps.
                if not counts_toward_loc(fn):
                    continue
                add += file_add
                dele += file_del
                if g:
                    by_group_adds[g] += file_add
                    by_group_dels[g] += file_del
                    contributors[user]["group_additions"][g] += file_add
                else:
                    contributors[user]["repo_additions"] += file_add

            contributors[user]["additions"] += add
            contributors[user]["deletions"] += dele

            if not hit:
                hit = path_groups

            # Unlabeled PRs that touch many topics/ (scaffold/docs) should not
            # inflate every group's merged-PR count with the full diff.
            scaffold_multi = (not groups) and len(path_groups) >= 4

            for g in hit:
                ga = by_group_adds.get(g, 0)
                gd = by_group_dels.get(g, 0)
                by_group[g]["additions"] += ga
                by_group[g]["deletions"] += gd
                if scaffold_multi and g not in groups:
                    continue
                by_group[g]["merged"] += 1
                contributors[user]["groups"][g] += 1

            merged_events.append(
                {
                    "at": pr["merged_at"],
                    "login": user,
                    "additions": add,
                    "groups": sorted(hit),
                    "by_group": dict(by_group_adds),
                }
            )
        elif pr.get("state") == "open":
            open_total += 1
            for g in groups:
                by_group[g]["open"] += 1
                if pr.get("draft"):
                    by_group[g]["draft"] += 1

    def top_group(v: dict) -> str:
        """Prefer the topic with most added lines; if most lines are outside topics/, use repo."""
        topic_add = sum(v["group_additions"].values())
        repo_add = v["repo_additions"]
        if repo_add > topic_add:
            return REPO_TAG
        if v["group_additions"]:
            return max(v["group_additions"].items(), key=lambda kv: (kv[1], kv[0]))[0]
        if v["groups"]:
            return max(v["groups"].items(), key=lambda kv: (kv[1], kv[0]))[0]
        return REPO_TAG

    top = sorted(
        (
            {
                "login": u,
                "merged_prs": v["merged_prs"],
                "additions": v["additions"],
                "deletions": v["deletions"],
                "net": v["additions"] - v["deletions"],
                "top_group": top_group(v),
            }
            for u, v in contributors.items()
            if v["merged_prs"] > 0
        ),
        key=lambda x: (x["merged_prs"], x["additions"]),
        reverse=True,
    )[:15]
    return by_group, merged_total, open_total, top, merged_events


def build_charts(issues: list, merged_events: list[dict]) -> dict:
    """Hourly group + contributor cumulative lines for 13–16 Oct (Europe/Berlin)."""
    hours = chart_hours()
    hour_set = set(hours)

    by_user_hour: dict[str, dict[str, int]] = defaultdict(lambda: {h: 0 for h in hours})
    by_group_hour: dict[str, dict[str, int]] = {
        g: {h: 0 for h in hours} for g in GROUPS
    }

    for e in merged_events:
        h = hour_key(e.get("at") or "")
        if not h or h not in hour_set:
            continue
        add = int(e.get("additions") or 0)
        login = e.get("login") or "unknown"
        by_user_hour[login][h] += add
        file_by_group = e.get("by_group") or {}
        if file_by_group:
            for g, ga in file_by_group.items():
                if g in by_group_hour:
                    by_group_hour[g][h] += int(ga or 0)
        else:
            # Fallback: split PR additions across labeled/path groups
            gs = [g for g in (e.get("groups") or []) if g in by_group_hour]
            if gs:
                share = add // len(gs)
                for g in gs:
                    by_group_hour[g][h] += share

    def cumulative(hourly: dict[str, int]) -> list[int]:
        total = 0
        out = []
        for h in hours:
            total += hourly[h]
            out.append(total)
        return out

    finals = {u: sum(by_user_hour[u].values()) for u in by_user_hour}
    race_logins = [
        u for u, _ in sorted(finals.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
        if finals[u] > 0
    ]
    race_series = {u: cumulative(by_user_hour[u]) for u in race_logins}

    group_finals = {g: sum(by_group_hour[g].values()) for g in GROUPS}
    group_ids = [
        g for g, _ in sorted(group_finals.items(), key=lambda kv: (-kv[1], kv[0]))
        if group_finals[g] > 0
    ]
    groups_series = {g: cumulative(by_group_hour[g]) for g in group_ids}
    groups_names = {g: GROUP_NAMES.get(g, g) for g in group_ids}

    return {
        "resolution": "hour",
        "timezone": "Europe/Berlin",
        "hours": hours,
        "groups": {
            "ids": group_ids,
            "names": groups_names,
            "series": groups_series,
        },
        "race": {
            "logins": race_logins,
            "series": race_series,
        },
    }


def loc_by_group_from_git():
    """Approximate churn per group folder via git log --numstat (all history on this clone)."""
    result = {g: {"additions": 0, "deletions": 0, "commits": 0} for g in GROUPS}
    resources = {"additions": 0, "deletions": 0, "commits": 0}
    if not (ROOT / ".git").exists():
        return result, resources

    def numstat(paths: list[str]):
        try:
            out = subprocess.check_output(
                ["git", "-C", str(ROOT), "log", "--pretty=format:COMMIT", "--numstat", "--", *paths],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return 0, 0, 0
        add = dele = commits = 0
        for line in out.splitlines():
            if line == "COMMIT":
                commits += 1
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            a, d, path = parts
            if not counts_toward_loc(path):
                continue
            if a.isdigit():
                add += int(a)
            if d.isdigit():
                dele += int(d)
        return add, dele, commits

    for g in GROUPS:
        folders = FOLDER_ALIASES.get(g, [g])
        paths = [f"topics/{folder}" for folder in folders]
        existing = [p for p in paths if (ROOT / p).exists()]
        if not existing:
            continue
        a, d, c = numstat(existing)
        result[g] = {"additions": a, "deletions": d, "commits": c}

    if (ROOT / "resources").exists():
        a, d, c = numstat(["resources"])
        resources = {"additions": a, "deletions": d, "commits": c}
    return result, resources


def main():
    issues, issues_by_group, ungrouped = collect_issues()
    prs_by_group, merged_total, open_prs, top_contributors, merged_events = collect_prs()
    loc_git, loc_resources = loc_by_group_from_git()
    charts = build_charts(issues, merged_events)

    groups_out = []
    for g in GROUPS:
        ig = issues_by_group[g]
        pg = prs_by_group[g]
        lg = loc_git[g]
        groups_out.append(
            {
                "id": g,
                "name": GROUP_NAMES.get(g, g),
                "issues_open": ig["open"],
                "issues_closed": ig["closed"],
                "tasks_open": ig["open_task"],
                "tasks_closed": ig["closed_task"],
                "prs_merged": pg["merged"],
                "prs_open": pg["open"],
                "prs_draft": pg["draft"],
                "pr_additions": pg["additions"],
                "pr_deletions": pg["deletions"],
                "git_additions": lg["additions"],
                "git_deletions": lg["deletions"],
                "git_commits": lg["commits"],
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": f"{OWNER}/{REPO}",
        "totals": {
            "issues_open": sum(1 for i in issues if i["state"] == "open"),
            "issues_closed": sum(1 for i in issues if i["state"] == "closed"),
            "tasks_closed": sum(
                g["tasks_closed"] for g in groups_out
            ),
            "prs_merged": merged_total,
            "prs_open": open_prs,
            "ungrouped_issues_open": ungrouped["open"],
            "ungrouped_issues_closed": ungrouped["closed"],
            "resources_git_additions": loc_resources["additions"],
            "resources_git_deletions": loc_resources["deletions"],
            "resources_git_commits": loc_resources["commits"],
        },
        "groups": groups_out,
        "top_contributors": top_contributors,
        "charts": charts,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {OUT} ({len(groups_out)} groups, {merged_total} merged PRs, "
        f"{len(charts['hours'])} chart hours)"
    )


if __name__ == "__main__":
    main()
