"""Interactive tile picker for JupyterLab (ipywidgets; map via ipyleaflet if installed, else folium HTML).

    from sm_pipeline.interactive import TilePicker
    TilePicker(ismn_geojson="ismn_locations.geojson").show()

One panel: map (station tiles, screening hits, your stacks; all 3 903 tiles optional) -> paste a coordinate or a stack key ->
Find -> choose the stack -> Run (reference only = no token). Extras (candidate windows, reference preview) are folded away.
Results: tabs with the three main figures per point, the key numbers, and the check figures folded.
"""
import subprocess, sys, pathlib, json
import numpy as np, pandas as pd
from . import catalog, auxiliary, reference

class TilePicker:
    def __init__(self, ismn_geojson=reference.ISMN_GEOJSON_DEFAULT, ismn_dir=reference.ISMN_DIR_DEFAULT, out="~/data/runs", levels=("BiomassLevel1c",), map_backend="folium"):
        """map_backend: 'folium' (static map with footprints after Find stacks; always works, coordinates typed),
        'leaflet' (clickable map; needs the jupyter-leaflet frontend extension loaded, i.e. pip install ipyleaflet + reload the page,
        possibly a server restart), or 'none'."""
        import ipywidgets as W
        from IPython.display import display
        import os
        self.W, self.display = W, display; self.out, self.levels, self.ismn_dir = out, levels, ismn_dir
        self.ismn = os.path.expanduser(ismn_geojson) if ismn_geojson and os.path.exists(os.path.expanduser(ismn_geojson)) else None
        self.stations = pd.DataFrame()
        self.lat = W.FloatText(value=41.99651, description="lat"); self.lon = W.FloatText(value=70.63916, description="lon")
        self.name = W.Text(value="Maydantal", description="name"); self.half = W.IntText(value=500, description="half window m")
        self.find = W.Button(description="Find", button_style="primary", tooltip="stacks covering the coordinate, or the stack key"); self.stack = W.Dropdown(options=[], description="stack", layout=W.Layout(width="560px"))
        self.onlygood = W.Checkbox(value=False, description="only tiles with a station or screening score >= 1", indent=False, layout=W.Layout(width="380px"))
        self.onlygood.observe(lambda ch: self._folium_map(), names="value")
        self.refonly = W.Checkbox(value=False, description="reference only (no token)", indent=False, layout=W.Layout(width="220px"))
        self.paste = W.Text(value="", description="paste lat, lon", placeholder="click on the map, then Ctrl+V here", layout=W.Layout(width="360px"))
        def _on_paste(change):
            try: la, lo = [float(x) for x in change["new"].replace(";", ",").split(",")[:2]]
            except Exception: return                                     # not yet a complete "lat, lon" text
            self.lat.value, self.lon.value = round(la, 5), round(lo, 5); self.name.value = f"P_{la:.3f}_{lo:.3f}"
            try: self._find(None)
            except Exception as ex:
                with self.table: print(f"stack search failed: {type(ex).__name__}: {str(ex)[:160]} -> press Find again (catalogue timeout?)")
        self.paste.observe(_on_paste, names="value")
        self.key = W.Text(value="", description="or stack key", placeholder="e.g. S1_T009_F118_G01_M04 (tooltip on the map)", layout=W.Layout(width="400px"))
        self.ref = W.Button(description="Preview reference data"); self.sar = W.Button(description="Run", button_style="warning", tooltip="run_stack.py for the chosen stack; background, log below")
        self.log = W.Output(layout=W.Layout(max_height="300px", overflow="auto")); self.table = W.Output(); self.plot = W.Output(); self.mapout = W.Output(); self.results = W.Output(); self.run_dir = None
        self.refrun = W.Button(description="Gather reference only"); self.refrun.on_click(lambda b: self._sar(b, no_sar=True))
        self.cands, self.best = None, None
        self.candbtn = W.Button(description="Find candidate windows", tooltip="rate every 600 m block of the selected stack's footprint by WorldCover land cover and DEM slope")
        self.candsel = W.Dropdown(options=[], description="use candidate", layout=W.Layout(width="520px")); self.candbtn.on_click(self._candidates)
        def _on_cand(change):
            if change["new"] is None or self.best is None: return
            r = self.best.iloc[change["new"]]; self.lat.value, self.lon.value, self.name.value = round(float(r.lat), 5), round(float(r.lon), 5), f"cand{change['new'] + 1}_{r.lat:.3f}_{r.lon:.3f}"
            with self.table: print(f"point set to candidate {change['new'] + 1}: {r.lat:.4f}, {r.lon:.4f} (grass {r.grassland_pct:.0f} %, tree {r.tree_pct:.0f} %, slope {r.slope_mean_deg:.1f} deg). Stack stays '{self.stack.value}'. Press Run.")
            self._folium_map()
        self.candsel.observe(_on_cand, names="value")
        self.find.on_click(lambda b: self._by_key(b) if self.key.value.strip() else self._find(b)); self.ref.on_click(self._reference)
        self.sar.on_click(lambda b: self._sar(b, no_sar=self.refonly.value)); self.stacks = {}
        self.map_backend = map_backend; self.m = None; self.has_leaflet = False
        if map_backend == "leaflet": self._build_map()

    def _folium_map(self):
        """Static folium map (HTML) with marker, 500 m / 1 km circles and the footprints found; shown in self.mapout."""
        import folium
        self.mapout.clear_output()
        with self.mapout:
            m = folium.Map(location=[self.lat.value, self.lon.value], zoom_start=8, tiles=None)
            folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)
            folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri", name="Esri satellite").add_to(m)
            folium.WmsTileLayer(url="https://services.terrascope.be/wms/v2", layers="WORLDCOVER_2021_MAP", fmt="image/png", transparent=True, overlay=True, show=False,
                                name="ESA WorldCover 2021 (green tree, yellow grass, pink crop, red built, grey bare)", attr="ESA WorldCover").add_to(m)
            if self.cands is not None and len(self.cands):
                fc = folium.FeatureGroup(name=f"candidate windows ({int(self.cands.suitable.sum())} suitable blocks; stars = best, criteria.json)", show=True)
                su = self.cands[self.cands.suitable]
                for _, r in su.iterrows():
                    folium.CircleMarker([r.lat, r.lon], radius=2, color="#009E73", fill=True, fill_opacity=0.6, weight=0.5,
                                        tooltip=f"grass {r.grassland_pct:.0f} %, crop {r.cropland_pct:.0f} %, tree {r.tree_pct:.0f} %, slope {r.slope_mean_deg:.1f} deg").add_to(fc)
                for i, (_, r) in enumerate(self.best.iterrows()):
                    folium.Marker([r.lat, r.lon], icon=folium.Icon(color="orange", icon="star"), tooltip=f"candidate {i + 1}: {r.lat:.4f}, {r.lon:.4f} | grass {r.grassland_pct:.0f} %, tree {r.tree_pct:.0f} %, slope {r.slope_mean_deg:.1f} deg").add_to(fc)
                fc.add_to(m)
            folium.Marker([self.lat.value, self.lon.value], tooltip=self.name.value).add_to(m)
            for r, c in ((500, "white"), (1000, "yellow")): folium.Circle([self.lat.value, self.lon.value], radius=r, color=c, weight=1.5, fill=False).add_to(m)
            self._tile_layer(m)
            for key, (col, k, pids) in list(self.stacks.items())[:12]:
                try: folium.GeoJson({"type": "Feature", "geometry": auxiliary.footprint(pids[0], col)}, name=key, tooltip=key,
                                    style_function=lambda f, c=("#ffcc00" if col.endswith("1c") else "#66ccff"): {"color": c, "weight": 1.5, "fillOpacity": 0.03}).add_to(m)
                except Exception: pass
            # click anywhere: the coordinates are copied to the clipboard (and shown) -> paste into the 'paste lat, lon' field
            folium.ClickForLatLng(format_str='lat + ", " + lng', alert=True).add_to(m)
            if self.ismn:   # ISMN stations for orientation
                import json as _j
                for f in _j.load(open(self.ismn))["features"]:
                    lo, la = f["geometry"]["coordinates"][:2]; pp = f["properties"]
                    folium.CircleMarker([la, lo], radius=3, color="#D55E00", fill=True, tooltip=f"{pp['station']} ({pp['network']}) to {pp['end'][:10]}").add_to(m)
            folium.LayerControl(collapsed=True).add_to(m)
            legend = ('<div style="position: fixed; bottom: 18px; left: 18px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #999; '
                      'border-radius: 4px; font: 11px/1.5 sans-serif; box-shadow: 0 0 4px #aaa;">'
                      '<b>BIOMASS L1C tiles</b> (expected soil-moisture signal, IFS screening)<br>'
                      '<span style="color:#1a9850">&#9632;</span> strong (rain event and moisture change, no frost) &nbsp;'
                      '<span style="color:#fee08b">&#9632;</span> medium<br>'
                      '<span style="color:#fdae61">&#9632;</span> weak &nbsp; <span style="color:#d73027">&#9632;</span> none / frozen / snow &nbsp;'
                      '<span style="color:#777777">&#9632;</span> not screened yet<br>'
                      '<b style="color:black">&#9633;</b> ISMN station inside &nbsp; <b style="color:#0072B2">&#9633;</b> stacks at your point &nbsp; '
                      '<span style="color:#D55E00">&#9679;</span> ISMN station</div>')
            m.get_root().html.add_child(folium.Element(legend)); self.display(m)

    def _tile_layer(self, m):
        """ALL L1C tile footprints (one polygon per track x frame from l1c_stack_index.csv) as ONE GeoJSON layer - light enough for
        3 903 tiles. Colour = best screening score of the tile if tile_screening.csv exists (green >= 2, yellow >= 1, orange >= 0.3,
        red < 0.3, grey = not screened); station tiles get a black outline; tooltip = tile, stacks, best stack, period, numbers.
        The checkbox 'only tiles with station or score >= 1' thins the view, the default shows everything."""
        import folium
        root = pathlib.Path(__file__).resolve().parents[1]; idx = root / "l1c_stack_index.csv"; scr = root / "tile_screening.csv"
        if not idx.exists(): return
        st = pd.read_csv(idx); st = st[st.n_acq >= 6].copy(); st["tile"] = st["stack"].str.extract(r"(T\d+_F\d+)")[0]
        if scr.exists():
            sc = pd.read_csv(scr)[["stack", "score", "max_event_between_acq_mm", "sm_range", "tmin_c", "snowfall_cm"]]; st = st.merge(sc, on="stack", how="left")
        else: st["score"] = np.nan
        found = [v[1] for v in getattr(self, "stacks", {}).values()]
        def col(v): return "#777777" if not np.isfinite(v) else ("#1a9850" if v >= 2 else "#fee08b" if v >= 1 else "#fdae61" if v >= 0.3 else "#d73027")
        feats = []
        for tile, g in st.groupby("tile"):
            b = g.iloc[g["score"].fillna(-1).argmax()]; stations = "; ".join(sorted(set(g.stations_inside.dropna().astype(str)))) if "stations_inside" in g else ""
            score = float(b.score) if np.isfinite(b.score) else None
            if getattr(self, "onlygood", None) is not None and self.onlygood.value and not stations and (score is None or score < 1) and not g["stack"].isin(found).any(): continue
            feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[round(x, 3), round(y, 3)] for x, y in ((g.w.min(), g.s.min()), (g.e.max(), g.s.min()), (g.e.max(), g.n.max()), (g.w.min(), g.n.max()), (g.w.min(), g.s.min()))]]},
                          "properties": {"tile": tile, "stacks": int(len(g)), "best_stack": str(b["stack"]), "period": f"{str(b.first_date)[:10]}..{str(b.last_date)[:10]}",
                                         "score": "" if score is None else round(score, 2), "event_mm": "" if score is None else round(float(b.max_event_between_acq_mm), 0),
                                         "sm_range": "" if score is None else round(float(b.sm_range), 3), "station": stations[:80], "color": col(b.score), "here": bool(g["stack"].isin(found).any())}})
        if not feats: return
        folium.GeoJson({"type": "FeatureCollection", "features": feats}, name=f"L1C tiles: {len(feats)}{' (coloured by screening score; black outline = ISMN station)' if scr.exists() else ' (grey = not screened; black outline = ISMN station)'}",
                       style_function=lambda f: {"color": "black" if f["properties"]["station"] else ("#0072B2" if f["properties"]["here"] else f["properties"]["color"]),
                                                 "weight": 2.0 if (f["properties"]["station"] or f["properties"]["here"]) else 0.6, "fillColor": f["properties"]["color"], "fillOpacity": 0.25 if scr.exists() else 0.08},
                       tooltip=folium.GeoJsonTooltip(fields=["tile", "stacks", "best_stack", "period", "score", "event_mm", "sm_range", "station"],
                                                     aliases=["tile", "stacks", "best stack", "period", "score", "max rain event [mm]", "SM range", "ISMN"], sticky=False)).add_to(m)

    def _build_map(self):
        try:
            import ipyleaflet as L
            self.m = L.Map(center=(self.lat.value, self.lon.value), zoom=4, scroll_wheel_zoom=True, layout=self.W.Layout(height="420px"))
            self.m.add_layer(L.basemap_to_tiles(L.basemaps.Esri.WorldImagery)); self.marker = L.Marker(location=(self.lat.value, self.lon.value), draggable=True)
            self.m.add_layer(self.marker); self.fp_layer = L.LayerGroup(); self.m.add_layer(self.fp_layer)
            def on_click(**kw):
                if kw.get("type") == "click":
                    la, lo = kw["coordinates"]; self.lat.value, self.lon.value = round(la, 5), round(lo, 5); self.marker.location = (la, lo)
            self.m.on_interaction(on_click); self.has_leaflet = True
        except Exception:
            self.m = None; self.has_leaflet = False

    def show(self):
        W = self.W
        if self.map_backend == "folium": self._folium_map()
        head = ("click the map, drag the marker, or paste a coordinate" if self.has_leaflet else
                "click the map → the coordinate is copied → paste it into 'paste lat, lon' (or type a stack key) → Find → choose the stack → Run")
        extras = W.Accordion(children=[
            W.VBox([W.HTML("rate all 600 m blocks of the chosen stack by land cover and slope (criteria.json); green = suitable, stars = best; pick one to set the point"),
                    W.HBox([self.candbtn, self.candsel])]),
            W.VBox([W.HTML("all soil-moisture sources, rain and acquisition dates at the point, without token"), self.ref, self.plot]),
            W.VBox([self.onlygood, W.HTML("default: ALL L1C tiles (3 903 footprints); black outline = ISMN station inside, blue = stacks found at your point, colour = screening score once tile_screening.csv exists")])])
        for i, t in enumerate(["Candidate windows in the tile", "Reference preview", "Map options"]): extras.set_title(i, t)
        extras.selected_index = None
        self.logbox = W.Accordion(children=[self.log]); self.logbox.set_title(0, "Run log"); self.logbox.selected_index = None
        ui = W.VBox([W.HTML(f"<b>BIOMASS soil moisture — pick a tile and run</b> &nbsp; <small>{head}</small>"), self.m if self.m else self.mapout,
                     W.HBox([self.paste, self.key, self.find]), W.HBox([self.lat, self.lon, self.name, self.half]), self.table,
                     W.HBox([self.stack, self.refonly, self.sar]), extras, self.logbox, self.results])
        self.display(ui)

    def _candidates(self, _):
        if not self.stack.value: return
        col, k, pids = self.stacks[self.stack.value]
        with self.log:
            try:
                crit = auxiliary.load_criteria(); geom = auxiliary.footprint(pids[0], col); self.cands = auxiliary.candidate_windows(geom, block_m=2 * self.half.value, criteria=crit); self.best = auxiliary.pick_spread(self.cands, criteria=crit)
                pd.set_option("display.width", 200)
                print(self.best[["rank", "lat", "lon", "grassland_pct", "cropland_pct", "tree_pct", "bare_pct", "slope_mean_deg", "slope_gt5_pct"]].round(2).to_string(index=False) if len(self.best) else "no block fulfils the criteria")
                self.candsel.options = [(f"{i + 1}: {r.lat:.4f}, {r.lon:.4f}  grass {r.grassland_pct:.0f} %  tree {r.tree_pct:.0f} %  slope {r.slope_mean_deg:.1f} deg", i) for i, (_, r) in enumerate(self.best.iterrows())]
                self.candsel.value = None
            except Exception as ex: print("candidate search failed:", str(ex)[:120])
        self._folium_map()

    def _find(self, _):
        self.table.clear_output(); self.stacks = {}
        with self.table:
            rows = []
            for col in self.levels:
                try: found = catalog.stacks_at(self.lat.value, self.lon.value, col)
                except Exception as ex:
                    print(f"catalogue query failed ({type(ex).__name__}: {str(ex)[:120]}); press Find again - the MAAP catalogue times out now and then"); found = {}
                for k, dd in found.items():
                    dates = sorted(dd); key = f"{col.replace('BiomassLevel', 'L')} {k}  {dates[0]}..{dates[-1]}  n={len(dates)}"
                    self.stacks[key] = (col, k, [dd[d] for d in dates]); rows.append(dict(level=col.replace("BiomassLevel", "L"), stack=k, first=dates[0], last=dates[-1], n=len(dates)))
            df = pd.DataFrame(rows).sort_values(["level", "first"]) if rows else pd.DataFrame(columns=["level", "stack", "first", "last", "n"])
            if not rows: print(f"no L1C stack covers {self.lat.value:.4f}, {self.lon.value:.4f} in the catalogue (point outside every footprint, or query failed above)")
            self.display(df.reset_index(drop=True)); self.stack.options = list(self.stacks)
            if self.map_backend == "folium": self._folium_map()
            if self.has_leaflet and rows:
                import ipyleaflet as L; self.fp_layer.clear_layers()
                for key, (col, k, pids) in list(self.stacks.items())[:12]:
                    try: self.fp_layer.add_layer(L.GeoJSON(data={"type": "Feature", "geometry": auxiliary.footprint(pids[0], col)}, style={"color": "#ffcc00" if col.endswith("1c") else "#66ccff", "weight": 1.5, "fillOpacity": 0.03}, name=key))
                    except Exception: pass

    def _by_key(self, _):
        """Select any stack worldwide by its key: look it up in l1c_stack_index.csv (built by build_tile_index.py), move the point to the
        stack centre (or to the first ISMN station inside), then run Find stacks."""
        import os
        k = self.key.value.strip(); idx = pathlib.Path(__file__).resolve().parents[1] / "l1c_stack_index.csv"
        with self.table:
            if not idx.exists(): print("l1c_stack_index.csv missing: run build_tile_index.py first"); return
            gi = pd.read_csv(idx); g = gi[gi["stack"] == k]
            if g.empty: print("stack key not in index:", k); return
            g = g.iloc[0]; self.lat.value, self.lon.value = round((g.s + g.n) / 2, 5), round((g.w + g.e) / 2, 5); self.name.value = f"centre_{k}"
            if isinstance(g.get("stations_inside"), str) and g["stations_inside"] not in ("", "nan") and self.ismn:
                st = auxiliary.stations_in_footprint(auxiliary.footprint(g["first_pid"], "BiomassLevel1c"), self.ismn)
                if len(st): self.lat.value, self.lon.value, self.name.value = round(st.iloc[0].lat, 5), round(st.iloc[0].lon, 5), st.iloc[0].station
        self._find(None)
        for opt in self.stack.options:
            if k in opt: self.stack.value = opt

    def _reference(self, _):
        self.plot.clear_output(); self.log.clear_output()
        if not self.stack.value: return
        col, k, pids = self.stacks[self.stack.value]
        with self.log:
            tab = catalog.stack_table(pids, col); times = pd.to_datetime(tab["datetime"], utc=True)
            s_date, e_date = (times.min() - pd.Timedelta(days=10)).date().isoformat(), (times.max() + pd.Timedelta(days=3)).date().isoformat()
            print(f"{k}: {len(tab)} acquisitions {tab['datetime'].iloc[0]} .. {tab['datetime'].iloc[-1]} UTC, primary {tab.loc[tab['primary'], 'datetime'].str[:10].tolist()}")
            self.stations = auxiliary.stations_in_footprint(auxiliary.footprint(pids[0], col), self.ismn, active_after=str(times.min().date())) if self.ismn else pd.DataFrame()
            if len(self.stations):
                for _, srow in self.stations.iterrows():
                    f = reference.ismn_station_file(self.ismn_dir, srow.network, srow.station)
                    print(f"ISMN station in footprint: {srow.station} ({srow.network}) {srow.lat:.4f},{srow.lon:.4f} -> file {'found' if f else 'MISSING, models only'}")
            else: print("no ISMN station inside this footprint -> SMOS/models only")
            try: print("SMOS L2 SM items in window:", len(auxiliary.smos_items(self.lat.value, self.lon.value, s_date, e_date)))
            except Exception as e: print("SMOS listing failed:", str(e)[:60])
            try: print("Sentinel-1 GRD scenes in window:", len(auxiliary.sentinel1_items(self.lat.value, self.lon.value, s_date, e_date)))
            except Exception as e: print("Sentinel-1 listing failed:", str(e)[:60])
            try:
                aux = auxiliary.terrain_landcover(self.lat.value, self.lon.value, self.half.value)
                print(f"terrain in {self.half.value} m window: slope mean {aux['slope_mean_deg']:.1f} deg, >5 deg {aux['slope_frac_gt5_pct']:.0f} %, elev range {aux['elev_range_m']:.0f} m; land cover {aux['landcover_pct']}")
            except Exception as e: print("terrain/landcover failed:", str(e)[:60])
            models = reference.all_model_sources(self.lat.value, self.lon.value, s_date, e_date)
        with self.plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            for m, d in models.items():
                ax[0].plot(d.index, pd.to_numeric(d["soil_moisture_0_to_7cm"], errors="coerce"), lw=1.2, label=f"{m} 0-7 cm"); ax[0].plot(d.index, pd.to_numeric(d["soil_moisture_7_to_28cm"], errors="coerce"), lw=0.8, ls=":", label=f"{m} 7-28 cm")
            for _, srow in self.stations.iterrows():
                f = reference.ismn_station_file(self.ismn_dir, srow.network, srow.station)
                if f is None: continue
                ser = reference.ismn_series(f); t0, t1 = pd.Timestamp(s_date, tz="UTC"), pd.Timestamp(e_date, tz="UTC") + pd.Timedelta(days=1)
                for lab, sr in ser.items():
                    if lab.startswith("soil_moisture") and not lab.endswith("_allflags"):
                        sr = sr[(sr.index >= t0) & (sr.index <= t1)]
                        if len(sr): ax[0].plot(sr.index, sr.values, lw=1.6, alpha=.85, label=f"ISMN {srow.station} {lab.replace('soil_moisture_', '')} (flag G)")
                pk = [k for k in ser if k.startswith("precipitation") and not k.endswith("_allflags")]
                if pk:
                    pr = pd.to_numeric(ser[pk[0]], errors="coerce").dropna(); pr = pr[(pr.index >= t0) & (pr.index <= t1)]; ax[1].bar(pr.index, pr.values, width=0.04, color="k", alpha=.6, label=f"ISMN rain {srow.station}")
            # rain bars: IFS first (ERA5-Land hourly rain is empty for 2026), fall back to any model with finite values
            for m in [m for m in ("ecmwf_ifs", "era5", "era5_land") if m in models]:
                pr = pd.to_numeric(models[m]["precipitation"], errors="coerce").dropna()
                if len(pr) and pr.abs().sum() >= 0: ax[1].bar(pr.index, pr.values, width=0.04, color="tab:blue", alpha=.7, label=f"{m} rain mm/h"); break
            ax[1].set_ylim(bottom=0)
            for t in times:
                for a in ax: a.axvline(t, color="k", ls="--", lw=0.8)
            ax[0].set_ylabel("soil moisture [m³/m³]"); ax[1].set_ylabel("rain [mm/h]"); ax[0].legend(fontsize=7, ncol=3); ax[1].legend(fontsize=7)
            ax[0].set_title(f"{self.name.value}  {k}: all soil-moisture sources, acquisitions dashed"); fig.tight_layout(); plt.show()

    def _sar(self, _, no_sar=False):
        """Run run_stack.py in a background thread (the notebook stays usable); output streams into the log; results appear below."""
        if not self.stack.value: return
        import threading, os
        col, k, pids = self.stacks[self.stack.value]
        cmd = [sys.executable, str(pathlib.Path(__file__).resolve().parents[1] / "run_stack.py"), "--stack", k, "--collection", col, "--lat", str(self.lat.value), "--lon", str(self.lon.value),
               "--name", self.name.value, "--half-m", str(self.half.value), "--out", self.out] + (["--ismn", self.ismn] if self.ismn else []) + (["--ismn-dir", self.ismn_dir] if self.ismn_dir else []) + (["--no-sar"] if no_sar else [])
        self.run_dir = pathlib.Path(os.path.expanduser(self.out)) / k
        def work():
            with self.log:
                print("running:", " ".join(cmd), flush=True)
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in p.stdout: print(line, end="", flush=True)
                p.wait(); print(f"[exit {p.returncode}] results in {self.run_dir}", flush=True)
            self.show_results()
        self.log.clear_output(); self.results.clear_output()
        if getattr(self, "logbox", None) is not None: self.logbox.selected_index = 0
        threading.Thread(target=work, daemon=True).start()

    def show_results(self, run_dir=None):
        """Results of a run folder: tabs 'Main' (biomass_vs_reference per point), 'Coherence', 'Quicklooks', 'Key numbers',
        'Checks' (the other figures, folded). Called after a run; callable by hand: picker.show_results('~/data/runs/<stack>')."""
        import os
        from IPython.display import Image, Markdown
        W = self.W; run_dir = pathlib.Path(os.path.expanduser(str(run_dir))) if run_dir else self.run_dir
        self.results.clear_output()
        with self.results:
            if not run_dir or not run_dir.exists(): print("no run folder yet"); return
            tj = run_dir / "tier.json"; t = json.load(open(tj)) if tj.exists() else {}
            self.display(Markdown(f"**{t.get('stack', run_dir.name)}** — reference tier {t.get('tier', '?')}: {t.get('tier_text', '')}  \nstations: {t.get('stations') or 'none'}  \nfolder: `{run_dir}`"))
            points = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name not in ("lut", "quicklooks", ".ipynb_checkpoints")])
            def imgs(name, width=1000):
                out = W.Output()
                with out:
                    found = False
                    for d in points:
                        f = d / name
                        if f.exists(): self.display(Markdown(f"**{d.name}**")); self.display(Image(filename=str(f), width=width)); found = True
                    if not found: print(f"no {name} (SAR stage skipped or not run)")
                return out
            keynum = W.Output()
            with keynum:
                m = run_dir / "metrics.csv"
                if m.exists():
                    M = pd.read_csv(m)
                    if "reference" in M:
                        key = M[M["observable"].str.startswith(("sigma0 HH", "sigma0 VV", "closure phase HH"), na=False)]
                        self.display(Markdown("**Key numbers** — slope only where the reference range >= 0.05 m³/m³; r_diff = correlation of first differences; see INTERPRETATION_GUIDE.md"))
                        self.display(key[["station", "reference", "observable", "value", "r", "r_diff", "ref_range", "obs_range", "note"]].round(3).reset_index(drop=True))
                    else: self.display(M)
                for d in points:
                    si = d / "sar_info.json"
                    if si.exists():
                        i = json.load(open(si))
                        self.display(Markdown(f"**{d.name}**: window {i.get('window_m')} m, valid {i.get('valid_pct', 0):.0f} %, height {i.get('height_m')} m, incidence {i.get('incidence_deg')}°, "
                                              f"DEM slope {i.get('dem_slope_mean_deg')}°, land cover {i.get('landcover_pct')}  \nflattening: {i.get('flattening')}  \ndenoising: {i.get('denoising')}"))
            checks = W.Output()
            with checks:
                for name in ["reference_sources.png"]:
                    f = run_dir / name
                    if f.exists(): self.display(Markdown(f"`{name}`")); self.display(Image(filename=str(f), width=900))
                for d in points:
                    for name in ["sar_observables.png", "sigma0_vs_reference.png", "window_images.png", "window_geometry.png", "closure_check.png"]:
                        f = d / name
                        if f.exists(): self.display(Markdown(f"`{d.name}/{name}`")); self.display(Image(filename=str(f), width=900))
                cw = run_dir / "candidate_windows.png"
                if cw.exists(): self.display(Markdown("`candidate_windows.png`")); self.display(Image(filename=str(cw), width=900))
            tabs = W.Tab(children=[imgs("biomass_vs_reference.png"), imgs("coherence_pairs.png"), imgs("quicklooks_pauli.png"), keynum, checks])
            for i, name in enumerate(["Main figure", "Coherence", "Quicklooks", "Key numbers", "Check figures"]): tabs.set_title(i, name)
            self.display(tabs)
