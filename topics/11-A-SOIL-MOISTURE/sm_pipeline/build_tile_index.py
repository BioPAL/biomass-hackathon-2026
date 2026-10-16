"""Build the global index of BIOMASS L1C stacks (all ~174k items -> ~25k stacks) and the global tile map.
    python build_tile_index.py [--ismn ismn_locations.geojson] [--items-parquet l1c_items_index.parquet]
Takes ~15 min on the first run (87 pages of 2000 items from the public MAAP catalogue); re-uses the parquet if present.
Outputs: l1c_items_index.parquet, l1c_stack_index.csv, biomass_l1c_tiles_map.html, biomass_l1c_tiles_overview.png
"""
import argparse, json, time, pathlib, sys
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sm_pipeline import catalog, plots, auxiliary

def fetch_items(limit=2000):
    rows, start, fails = [], 1, 0
    while True:
        try: jj = catalog._get(f"{catalog.STAC.format(col='BiomassLevel1c')}?limit={limit}&startRecord={start}")
        except Exception:
            fails += 1; time.sleep(5)
            if fails > 10: raise
            continue
        feats = jj.get("features", [])
        if not feats: break
        for f in feats:
            p = f["properties"]; b = f["bbox"]
            rows.append(dict(pid=f["id"], dt=p["start_datetime"][:19], primary=p.get("eofeos:is_coregistration_primary"), stack_id=p.get("eofeos:stack_id"), orbit_state=p.get("sat:orbit_state"), w=b[0], s=b[1], e=b[2], n=b[3]))
        start += limit; print(f"  {len(rows)}/{jj.get('numberMatched')}", end="\r")
        if start > jj.get("numberMatched", 0): break
    df = pd.DataFrame(rows); df["stack"] = df["pid"].map(catalog.stack_key); return df

def stack_index(items, ismn_geojson=None):
    g = items.sort_values("dt").groupby("stack")
    st = g.agg(n_acq=("pid", "count"), first_date=("dt", "first"), last_date=("dt", "last"), w=("w", "min"), s=("s", "min"), e=("e", "max"), n=("n", "max"), first_pid=("pid", "first"), orbit_state=("orbit_state", "first")).reset_index()
    prim = items[items["primary"] == True].groupby("stack")["dt"].first().str[:10]; st["primary_date"] = st["stack"].map(prim)
    st["stations_inside"] = ""
    if ismn_geojson:   # bbox test first (cheap), exact footprint test only for the hits
        gj = json.load(open(ismn_geojson)); pts = [(f["properties"]["station"], f["properties"]["network"], *f["geometry"]["coordinates"][:2], f["properties"]["end"]) for f in gj["features"]]
        for i, r in st.iterrows():
            cand = [p for p in pts if r.w <= p[2] <= r.e and r.s <= p[3] <= r.n and p[4] >= r.first_date]
            if cand:
                try:
                    inside = auxiliary.stations_in_footprint(auxiliary.footprint(r.first_pid), ismn_geojson, active_after=r.first_date)
                    st.at[i, "stations_inside"] = ";".join(inside["station"]) if len(inside) else ""
                except Exception: st.at[i, "stations_inside"] = ";".join(f"{p[0]}?" for p in cand)
    return st.sort_values("first_date").reset_index(drop=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ismn"); ap.add_argument("--items-parquet", default="l1c_items_index.parquet"); ap.add_argument("--min-acq", type=int, default=6)
    a = ap.parse_args()
    if pathlib.Path(a.items_parquet).exists(): items = pd.read_parquet(a.items_parquet); print("items from", a.items_parquet, len(items))
    else: items = fetch_items(); items.to_parquet(a.items_parquet); print("fetched", len(items))
    st = stack_index(items, a.ismn); st.to_csv("l1c_stack_index.csv", index=False)
    print(f"{len(st)} stacks, {int((st.n_acq >= a.min_acq).sum())} with >= {a.min_acq} acquisitions, {int((st.stations_inside != '').sum())} with an active ISMN station inside")
    plots.global_tile_map("l1c_stack_index.csv", a.ismn, "biomass_l1c_tiles_map.html", "biomass_l1c_tiles_overview.png", a.min_acq); print("wrote map + overview")
