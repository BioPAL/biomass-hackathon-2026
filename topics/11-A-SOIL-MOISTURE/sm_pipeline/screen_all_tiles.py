"""Stage 0: screen EVERY L1C stack (tile) with a cheap model reference so that interesting tiles can be picked for case studies.

    python screen_all_tiles.py [--model ecmwf_ifs] [--min-acq 6] [--batch 100]

For each stack: ECMWF IFS (or ERA5-Land) daily rain, surface soil moisture, Tmin and snowfall at the stack centre over the
stack window (+10 days before). Derived: rain in window, largest rain sum between two consecutive acquisitions, soil-moisture
range in the window, minimum air temperature, snowfall sum, and a 'moisture signal' score. One call per tile footprint (3 903 tiles,
~40 calls of 100 locations, a few minutes); the free Open-Meteo quota is 10 000 locations per day and IP. Output: tile_screening.csv (joined to l1c_stack_index.csv columns) and
biomass_l1c_tiles_screening_map.html (tiles coloured by score; station tiles outlined).
Caveat: model values at a 9 km grid point for the tile CENTRE; a tile is ~50 x 150 km, so this is a first filter, not a result.
"""
import argparse, json, time, urllib.request, urllib.error, pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

OM = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={s}&end_date={e}"
      "&daily=precipitation_sum,soil_moisture_0_to_7cm_mean,temperature_2m_min,snowfall_sum&models={model}&timezone=UTC")

class QuotaExceeded(Exception):
    pass


def get(url, tries=3):
    """One Open-Meteo call. HTTP 429 = quota (600/min, 5 000/h, 10 000/day weighted by locations and data volume, per IP):
    wait 60 s, retry twice, then raise QuotaExceeded so the caller stops instead of burning the remaining batches."""
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r: return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"  429 quota: waiting 60 s ({i + 1}/{tries})", flush=True); time.sleep(60); last = e; continue
            last = e; time.sleep(5)
        except Exception as e:
            last = e; print(f"  retry {i + 1}: {str(e)[:60]}", flush=True); time.sleep(5 * (i + 1))
    if isinstance(last, urllib.error.HTTPError) and last.code == 429: raise QuotaExceeded(str(last))
    raise last

def screen(stacks, items, model="ecmwf_ifs", batch=100, log=None, partial="tile_screening_partial.csv"):
    """One Open-Meteo call per TILE FOOTPRINT (track x frame; ~3 900 instead of ~25 000 stacks) covering the whole index period;
    the per-stack windows are cut out locally. Keeps the run inside the free quota (10 000 locations/day/IP).
    Progress is logged per batch (flushed) and the finished tiles are appended to `partial`, so an interrupted run resumes."""
    log = log or (lambda m: print(m, flush=True))
    dates_by_stack = items.sort_values("dt").groupby("stack")["dt"].apply(lambda s: sorted(set(x[:10] for x in s)))
    stacks = stacks.copy(); stacks["tile"] = stacks["stack"].str.extract(r"(T\d+_F\d+)")[0]
    tiles = stacks.groupby("tile").agg(clat=("clat", "mean"), clon=("clon", "mean"), d0=("d0", "min"), d1=("d1", "max")).reset_index()
    rows = []; done = set()
    if partial and pathlib.Path(partial).exists():
        prev = pd.read_csv(partial); rows = prev.to_dict("records"); done = set(prev["tile"]); log(f"resuming: {len(done)} tiles already screened")
    # tiles sorted by their first date so that each batch of 100 shares a short date range (less data volume = less quota per call)
    tiles = tiles[~tiles.tile.isin(done)].sort_values(["d0", "d1"]); t0 = time.time(); n_calls = 0
    for k in range(0, len(tiles), batch):
        sub = tiles.iloc[k:k + batch]
        s_b = (pd.Timestamp(sub.d0.min()) - pd.Timedelta(days=10)).date().isoformat(); e_b = (pd.Timestamp(sub.d1.max()) + pd.Timedelta(days=2)).date().isoformat()
        url = OM.format(lat=",".join(f"{v:.3f}" for v in sub.clat), lon=",".join(f"{v:.3f}" for v in sub.clon), s=s_b, e=e_b, model=model)
        try: res = get(url)
        except QuotaExceeded:
            log(f"Open-Meteo quota reached after {len(rows)} stack rows ({k + len(done)} tiles). Progress is saved in {partial}; "
                "run the same command again later (next hour, or tomorrow 00:00 UTC) - it resumes where it stopped."); break
        except Exception as ex: log(f"batch {k} failed: {str(ex)[:80]}"); continue
        res = res if isinstance(res, list) else [res]; n_calls += 1
        for (_, tr), j in zip(sub.iterrows(), res):
            d = j.get("daily", {}); t = np.array(d.get("time", []))
            f = lambda key: np.array([x if x is not None else np.nan for x in d.get(key, [])], float)
            pr, sm, tmin, sn = f("precipitation_sum"), f("soil_moisture_0_to_7cm_mean"), f("temperature_2m_min"), f("snowfall_sum")
            for _, r in stacks[stacks.tile == tr.tile].iterrows():
                acq = dates_by_stack.get(r["stack"], [r["d0"], r["d1"]]); win = (t >= acq[0]) & (t <= acq[-1])
                if not len(t) or not win.any():
                    rows.append(dict(tile=tr.tile, stack=r["stack"], model=model)); continue
                ev = [np.nansum(pr[(t > a) & (t <= b)]) for a, b in zip(acq[:-1], acq[1:])]; smw = sm[win]
                rows.append(dict(tile=tr.tile, stack=r["stack"], rain_window_mm=float(np.nansum(pr[win])), max_event_between_acq_mm=float(np.nanmax(ev)) if ev else np.nan,
                                 rain_10d_before_mm=float(np.nansum(pr[(t < acq[0]) & (t >= (pd.Timestamp(acq[0]) - pd.Timedelta(days=10)).date().isoformat())])),
                                 sm_min=float(np.nanmin(smw)), sm_max=float(np.nanmax(smw)), sm_range=float(np.nanmax(smw) - np.nanmin(smw)),
                                 tmin_c=float(np.nanmin(tmin[win])), snowfall_cm=float(np.nansum(sn[win])), model=model, grid_elev_m=j.get("elevation")))
        log(f"{min(k + batch, len(tiles))}/{len(tiles)} tiles, {len(rows)} stack rows, {n_calls} calls, {round(time.time() - t0)} s")
        if partial: pd.DataFrame(rows).to_csv(partial, index=False)
    out = pd.DataFrame(rows)
    # score: moisture signal present (event >= 10 mm or SM range >= 0.08), penalise frost and snow
    out["score"] = (np.clip(out.max_event_between_acq_mm / 20, 0, 1.5) + np.clip(out.sm_range / 0.1, 0, 1.5)) * np.where(out.tmin_c < 0, 0.2, 1.0) * np.where(out.snowfall_cm > 1, 0.2, 1.0)
    return out.drop(columns=["tile"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="ecmwf_ifs"); ap.add_argument("--min-acq", type=int, default=6); ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--index", default="l1c_stack_index.csv"); ap.add_argument("--items", default="l1c_items_index.parquet"); ap.add_argument("--ismn")
    a = ap.parse_args()
    st = pd.read_csv(a.index); st = st[st.n_acq >= a.min_acq].copy(); st["clat"] = (st.s + st.n) / 2; st["clon"] = (st.w + st.e) / 2; st["d0"] = st.first_date.str[:10]; st["d1"] = st.last_date.str[:10]
    items = pd.read_parquet(a.items)
    scr = screen(st, items, a.model, a.batch)
    full = st.drop(columns=["d0", "d1"]).merge(scr, on="stack", how="left"); full.to_csv("tile_screening.csv", index=False)
    print(full.sort_values("score", ascending=False)[["stack", "first_date", "n_acq", "rain_window_mm", "max_event_between_acq_mm", "sm_range", "tmin_c", "snowfall_cm", "score", "stations_inside"]].head(25).to_string(index=False))
    # map coloured by score
    import folium, json as _j
    m = folium.Map(location=[0, 20], zoom_start=2, tiles="OpenStreetMap")
    def col(sc): return "#777777" if not np.isfinite(sc) else ("#1a9850" if sc >= 2 else "#fee08b" if sc >= 1 else "#d73027" if sc < 0.3 else "#fdae61")
    fg = folium.FeatureGroup(name="tiles coloured by moisture-signal score (green high, red none)")
    for _, r in full.iterrows():
        folium.Rectangle([[r.s, r.w], [r.n, r.e]], color=col(r.score), weight=0.6, fill=True, fill_opacity=0.25,
                         tooltip=f"{r.stack} {str(r.first_date)[:10]}..{str(r.last_date)[:10]} n={r.n_acq} | event {r.max_event_between_acq_mm:.0f} mm, SM range {r.sm_range:.2f}, Tmin {r.tmin_c:.0f} C, snow {r.snowfall_cm:.0f} cm, score {r.score:.2f}").add_to(fg)
    fg.add_to(m)
    fs = folium.FeatureGroup(name="tiles with ISMN station inside")
    for _, r in full[full.stations_inside.fillna("") != ""].iterrows():
        folium.Rectangle([[r.s, r.w], [r.n, r.e]], color="black", weight=2.5, fill=False, popup=f"{r.stack}: {r.stations_inside}").add_to(fs)
    fs.add_to(m)
    if a.ismn:
        g = _j.load(open(a.ismn)); fi = folium.FeatureGroup(name="ISMN stations")
        for f in g["features"]:
            lo, la = f["geometry"]["coordinates"][:2]; folium.CircleMarker([la, lo], radius=2.5, color="black", fill=True, tooltip=f["properties"]["station"]).add_to(fi)
        fi.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m); m.save("biomass_l1c_tiles_screening_map.html"); print("wrote tile_screening.csv and biomass_l1c_tiles_screening_map.html")
