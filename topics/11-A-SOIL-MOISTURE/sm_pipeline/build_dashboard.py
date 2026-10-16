#!/usr/bin/env python
"""Global dashboard: ONE static HTML file over all BIOMASS L1C tiles and stacks. No server, no kernel; works offline except for the
optional OSM / satellite base maps and the public BPS quicklooks. Built from what is complete on disk:

  l1c_stack_index.csv + l1c_items_index.parquet   all tiles, stacks, acquisition dates, product ids (public quicklooks)
  tile_landcover.csv                              WorldCover 2021 fractions and Copernicus DEM slope per tile (tile_landcover.py)
  <runs>/all_tiles_summary.csv  (optional)        BIOMASS numbers and main figure of computed stacks (run_stack.py / run_batch.py)

    python build_dashboard.py                     # -> dashboard.html (runs from ~/data/runs if present)
    python build_dashboard.py --runs ~/data/runs --out dashboard.html

In the page: colour tiles by land cover, slope or number of stacks; filter (grassland + cropland, tree cover, slope, ISMN station,
computed, acquisition month); computed points as dots (yellow: in-situ station data, orange: model reference only,
grey: SAR skipped); click a tile for its land cover, its stacks and their quicklooks and, for computed stacks, the BIOMASS numbers and the
main figure (BIOMASS vs soil-moisture reference). 'Copy as CSV' gives the filtered stacks as input for run_batch.py.
"""
import argparse, base64, gzip, io, json, os, pathlib, datetime as dt
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
DAY0 = pd.Timestamp("2025-01-01")
FIG_W = 900                                   # px width of embedded main figures (JPEG)
LC_COLS = ["grassland_pct", "cropland_pct", "tree_pct", "shrub_pct", "bare_pct", "built_pct", "snow_pct", "wetland_pct", "moss_pct", "mangrove_pct",
           "water_pct", "land_pct", "slope_mean_deg", "slope_p90_deg", "elev_mean_m"]
RUN_COLS = ["station", "network", "lat", "lon", "has_ismn", "ref_used", "ref_range", "dsigma0_HH_dB", "dsigma0_VV_dB", "r_diff_HH", "r_diff_p_HH", "mean_coh_HH",
            "dem_slope_deg", "grassland_pct", "tree_pct", "skipped"]


def _day(s):
    return ((pd.to_datetime(pd.Series(list(s)).astype(str).str[:10]) - DAY0).dt.days).astype(int).tolist()


def _num(x, nd):
    try: x = float(x)
    except (TypeError, ValueError): return None
    return round(x, nd) if np.isfinite(x) else None


def _fig_jpeg(png):
    from PIL import Image
    im = Image.open(png).convert("RGB"); im = im.resize((FIG_W, int(im.height * FIG_W / im.width)))
    b = io.BytesIO(); im.save(b, "JPEG", quality=72); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _tile_box(g):
    """Bounding box of the stacks of one tile; antimeridian-safe (crossing stacks -> [e, w + 360]; whole tile shifted if narrower)."""
    w, e = g.w.values.astype(float), g.e.values.astype(float)
    cross = (e - w) > 180; w, e = np.where(cross, e, w), np.where(cross, w + 360, e)
    wb, eb = np.where(w < 0, w + 360, w), np.where(w < 0, e + 360, e)
    if eb.max() - wb.min() < e.max() - w.min(): w, e = wb, eb
    return [round(float(w.min()), 3), round(float(g.s.min()), 3), round(float(e.max()), 3), round(float(g.n.max()), 3)]


def build(index, items, landcover, runs, out):
    st = pd.read_csv(index); st = st[st.n_acq >= 6].copy().reset_index(drop=True)
    k = st["stack"].str.extract(r"S(\d)_T(\d+)_F(\d+)_G\d+_M(\d+)"); st["tile"] = "T" + k[1] + "_F" + k[2]; st["sw"] = k[0].astype(int)
    tiles = []
    for tile, g in st.groupby("tile", sort=True):
        sta = sorted(set("; ".join(g.stations_inside.dropna().astype(str)).split("; ")) - {""})
        tiles.append(dict(id=tile, b=_tile_box(g), sta="; ".join(sta)[:200]))
    tidx = {t["id"]: i for i, t in enumerate(tiles)}
    it = pd.read_parquet(items, columns=["pid", "dt", "stack"]); it = it[it["stack"].isin(st["stack"])].sort_values("dt")
    acq = it.groupby("stack")["dt"].apply(_day); pids = it.groupby("stack")["pid"].apply(list)
    stacks = dict(key=st["stack"].tolist(), t=[tidx[t] for t in st.tile], sw=st.sw.tolist(),
                  bb=[[round(float(v), 3) for v in r] for r in st[["w", "s", "e", "n"]].values], f=_day(st.first_date), l=_day(st.last_date),
                  n=st.n_acq.astype(int).tolist(), o=[("a" if str(x).startswith("asc") else "d") for x in st.orbit_state],
                  acq=[acq.get(x, []) for x in st["stack"]], pid=[pids.get(x, []) for x in st["stack"]])
    lc = None
    if landcover and pathlib.Path(landcover).exists():
        L = pd.read_csv(landcover).drop_duplicates("tile", keep="last").set_index("tile")
        frac = [c for c in LC_COLS if c.endswith("_pct") and c not in ("water_pct", "land_pct") and c in L]
        L.loc[L["land_pct"].fillna(0) < 1, frac] = np.nan            # no WorldCover land in the box (open sea, Antarctica): no class values
        lc = {c: [_num(L[c].get(t["id"]), 1) for t in tiles] for c in LC_COLS if c in L}
        print(f"[dashboard] land cover / slope for {L.index.isin(tidx).sum()} of {len(tiles)} tiles")
    else:
        print("[dashboard] no tile_landcover.csv: land-cover colours and filters are left out")
    runs_out = {}
    rd = pathlib.Path(os.path.expanduser(runs)) if runs else None
    if rd is not None and (rd / "all_tiles_summary.csv").exists():
        R = pd.read_csv(rd / "all_tiles_summary.csv")
        if "lat" not in R:                                            # older summaries: point coordinates from the run folders
            ll = {}
            for sk in R["stack"].astype(str).unique():
                f = rd / sk / "reference_at_overpass.json"
                if f.exists():
                    for x in json.load(open(f)): ll[(sk, str(x["station"]))] = (x.get("lat"), x.get("lon"), x.get("network"))
            R["lat"] = [ll.get((str(a), str(b)), (None, None, None))[0] for a, b in zip(R["stack"], R["station"])]
            R["lon"] = [ll.get((str(a), str(b)), (None, None, None))[1] for a, b in zip(R["stack"], R["station"])]
            if "network" not in R: R["network"] = [ll.get((str(a), str(b)), (None, None, None))[2] for a, b in zip(R["stack"], R["station"])]
        for _, r in R.iterrows():
            rec = {c: (_num(r[c], 3) if isinstance(r[c], (int, float, np.number)) and not isinstance(r[c], bool) else (None if pd.isna(r[c]) else str(r[c])))
                   for c in RUN_COLS if c in R}
            rec["has_ismn"] = str(r.get("has_ismn")).strip().lower() in ("true", "1", "1.0")
            png = rd / str(r["stack"]) / str(r["station"]) / "biomass_vs_reference.png"
            if png.exists():
                try: rec["fig"] = _fig_jpeg(png)
                except Exception as ex: print(f"[dashboard] figure {png} skipped ({str(ex)[:60]})")
            runs_out.setdefault(str(r["stack"]), []).append(rec)
        print(f"[dashboard] computed stacks: {len(runs_out)} ({len(R)} stack x point rows) from {rd}")
    meta = dict(built=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), n_tiles=len(tiles), n_stacks=len(st), d_min=int(min(stacks["f"])), d_max=int(max(stacks["l"])),
                n_lc=0 if lc is None else int(sum(v is not None for v in lc.get("land_pct", []))), n_sta=int(sum(bool(t["sta"]) for t in tiles)))
    land = json.loads((HERE / "vendor" / "land-110m.geojson").read_text())
    raw = json.dumps(dict(meta=meta, tiles=tiles, stacks=stacks, lc=lc, runs=runs_out, land=land), separators=(",", ":")).encode()
    vend = HERE / "vendor"
    html = (TEMPLATE.replace("__LEAFLET_CSS__", (vend / "leaflet.css").read_text()).replace("__LEAFLET_JS__", (vend / "leaflet.js").read_text())
            .replace("__PAYLOAD__", base64.b64encode(gzip.compress(raw, 9)).decode()))
    pathlib.Path(out).write_text(html)
    print(f"[dashboard] {out}: {len(tiles)} tiles, {len(st)} stacks, {meta['n_lc']} tiles with land cover, {len(runs_out)} computed stacks; file {len(html) / 1e6:.1f} MB")
    return out


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>BIOMASS L1C tiles - soil-moisture test sites</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>__LEAFLET_CSS__</style>
<script>__LEAFLET_JS__</script>
<style>
html,body{margin:0;height:100%;font:13px/1.4 Helvetica,Arial,sans-serif;color:#222}
#app{display:grid;grid-template-columns:290px 1fr 470px;height:100%}
#ctl{overflow:auto;padding:12px 14px;background:#f6f6f6;border-right:1px solid #ddd}
#map{height:100%;background:#dfe9f2}
#side{overflow:auto;padding:12px 16px;border-left:1px solid #ddd;background:#fff}
h1{font-size:15px;margin:0 0 6px}h2{font-size:13px;margin:14px 0 5px;color:#0072B2}
label{display:block;margin:7px 0 2px}input[type=range]{width:100%}.val{font-weight:bold;font-variant-numeric:tabular-nums}
.small{font-size:11.5px;color:#555}.cnt{font-size:15px;font-weight:bold;margin:8px 0 2px}
button{font:12px Helvetica,Arial;padding:4px 9px;margin:3px 3px 3px 0;border:1px solid #888;background:#fff;border-radius:3px;cursor:pointer}
button:hover{background:#e8eef5}button.act{background:#0072B2;color:#fff;border-color:#0072B2;font-weight:bold}button.act:hover{background:#005a8c}select{font:12px Helvetica,Arial;width:100%;padding:2px}
table{border-collapse:collapse;font-size:11.5px;width:100%}th,td{border-bottom:1px solid #eee;padding:3px 4px;text-align:right}
th{background:#f0f0f0;position:sticky;top:0}td:first-child,th:first-child{text-align:left}tr.sel td{background:#fff3cd}
tr.st:hover td{background:#eef5fb;cursor:pointer}.thumbs{display:flex;gap:4px;overflow-x:auto;padding-bottom:4px}.thumbs figure{margin:0;text-align:center;font-size:10.5px}
.thumbs img{height:200px;border:1px solid #ccc}.tw{overflow-x:auto}.legend{background:#fff;padding:6px 8px;border:1px solid #999;border-radius:4px;font-size:11px;line-height:1.5}
.bar{height:10px;width:170px;border:1px solid #999}.fig img{width:100%;border:1px solid #ddd}textarea{width:100%;height:90px;font-size:11px}
.lcbar{display:flex;height:14px;border:1px solid #999;margin:4px 0}.lcbar div{height:100%}.key span{display:inline-block;margin-right:8px;font-size:11px}
.key i{display:inline-block;width:10px;height:10px;margin-right:3px;vertical-align:-1px}
.mon{display:grid;grid-template-columns:repeat(6,1fr);font-size:11.5px;margin:2px 0 4px}.mon label{margin:0;white-space:nowrap}
#loading{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;background:#fffd;font-size:16px;z-index:9999}
</style></head><body>
<div id="loading">loading ...</div>
<div id="app">
<div id="ctl">
 <h1>BIOMASS L1C: where can P-band soil-moisture sensitivity be tested?</h1>
 <div class="small" id="meta"></div>
 <div class="cnt" id="count"></div><div class="small" id="count2"></div>
 <h2>Colour tiles by</h2><select id="colour"></select><div id="hist"></div>
 <h2>Filters</h2>
 <div id="lcf"></div>
 <label><input type="checkbox" id="f_sta"> only tiles with an ISMN station</label>
 <label><input type="checkbox" id="f_run"> only tiles with computed points (results in the runs folder)</label>
 <div class="small" style="margin-top:8px"><b>months</b> (middle of the 18-day stack window)</div>
 <div id="months" class="mon"><label><input type="checkbox" id="m0" checked>Jan</label><label><input type="checkbox" id="m1" checked>Feb</label><label><input type="checkbox" id="m2" checked>Mar</label><label><input type="checkbox" id="m3" checked>Apr</label><label><input type="checkbox" id="m4" checked>May</label><label><input type="checkbox" id="m5" checked>Jun</label><label><input type="checkbox" id="m6" checked>Jul</label><label><input type="checkbox" id="m7" checked>Aug</label><label><input type="checkbox" id="m8" checked>Sep</label><label><input type="checkbox" id="m9" checked>Oct</label><label><input type="checkbox" id="m10" checked>Nov</label><label><input type="checkbox" id="m11" checked>Dec</label></div>
 <label><input type="checkbox" id="f_warm"> warm season only: May-Sep north of 23.5&deg; N, Nov-Mar south of 23.5&deg; S, tropics all year
  (proxy against frozen soil and snow; mountains can freeze in any month - see the frost check in the report)</label>
 <label><input type="checkbox" id="f_hide" checked> hide tiles outside the filters (unticked: show them grey)</label>
 <div style="margin-top:8px" id="presets"></div>
 <h2>Filtered stacks</h2><button id="exp">copy as CSV for run_batch.py</button><textarea id="exptxt" style="display:none"></textarea>
 <p class="small" id="caveat"></p>
</div>
<div id="map"></div>
<div id="side"><h1>Click a tile</h1><p class="small">Shows the land cover and terrain of the tile, all its BIOMASS stacks and, for a selected stack, the seven public BPS quicklooks and, where the stack was computed, the BIOMASS numbers and the main figure (BIOMASS backscatter and coherence against the soil-moisture reference).</p></div>
</div>
<script id="payload" type="application/octet-stream">__PAYLOAD__</script>
<script>
"use strict";
const OI={blue:"#0072B2",orange:"#E69F00",verm:"#D55E00"};
const PCOL={insitu:"#FFD700",model:"#F57C00",skip:"#BDBDBD"};   // computed points: in-situ station data / model reference only / SAR skipped
const pcol=r=>r.skipped?PCOL.skip:(r.has_ismn?PCOL.insitu:PCOL.model);
const dot=c=>`<svg width="11" height="11" style="vertical-align:-1px"><circle cx="5.5" cy="5.5" r="4.5" fill="${c}" stroke="#000" stroke-width="1"/></svg>`;
const PAL={viridis:["#440154","#3b528b","#21918c","#5ec962","#fde725"],greens:["#f7fcf5","#c7e9c0","#74c476","#238b45","#00441b"],slope:["#fcfdbf","#feb078","#f1605d","#b5367a","#3b0f70"]};
const WCC={tree:"#006400",shrub:"#ffbb22",grassland:"#ffff4c",cropland:"#f096ff",built:"#fa0000",bare:"#b4b4b4",snow:"#f0f0f0",wetland:"#0096a0",mangrove:"#00cf75",moss:"#fae6a0"};
let D,MAP,REN,PREN,ALL=[],KEY2I={},WINM=600,RECTS=[],TS=[],SELSTACK=null,SELTILE=null,PT=null,PTM=null,BOX=null,FOOT=null,PASS=null,TVAL=null,VARS,LEG;
const $=id=>document.getElementById(id);
function day2iso(d){return new Date(Date.UTC(2025,0,1)+d*864e5).toISOString().slice(0,10)}
function hex(c){return [1,3,5].map(i=>parseInt(c.slice(i,i+2),16))}
function ramp(p,x){x=Math.max(0,Math.min(1,x));const n=p.length-1,i=Math.min(n-1,Math.floor(x*n)),f=x*n-i,a=hex(p[i]),b=hex(p[i+1]);
  return "#"+a.map((v,k)=>Math.round(v+(b[k]-v)*f).toString(16).padStart(2,"0")).join("")}
async function load(){const b=atob($("payload").textContent.trim()),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);
  const s=new Blob([u]).stream().pipeThrough(new DecompressionStream("gzip"));return JSON.parse(await new Response(s).text())}
function fmt(x,n){return x===null||x===undefined?"":(+x).toFixed(n)}
function lcv(k,t){return D.lc&&D.lc[k]?D.lc[k][t]:null}
function gc(t){const g=lcv("grassland_pct",t);return g===null?null:g+(lcv("cropland_pct",t)||0)}
function vars(){const v=[];
  if(D.lc&&D.lc.grassland_pct){v.push({k:"gc",lab:"grassland + cropland [% of land]",f:gc,dom:[0,100],pal:"greens",nd:0});
    v.push({k:"tree",lab:"tree cover [% of land]",f:t=>lcv("tree_pct",t),dom:[0,100],pal:"greens",nd:0});
    v.push({k:"slope",lab:"mean slope [deg] (Copernicus DEM, 200 m)",f:t=>lcv("slope_mean_deg",t),dom:[0,15],pal:"slope",nd:1})}
  v.push({k:"nst",lab:"number of passing stacks",f:t=>TS[t].reduce((a,i)=>a+PASS[i],0),dom:[0,15],pal:"viridis",nd:0});return v}
function lcFilters(){if(!D.lc||!D.lc.grassland_pct)return;
  $("lcf").innerHTML='<label>grassland + cropland &ge; <span id="v_gc" class="val"></span> % of land</label><input type="range" id="f_gc" min="0" max="100" step="5" value="0">'+
   '<label>tree cover &le; <span id="v_tr" class="val"></span> % of land</label><input type="range" id="f_tr" min="0" max="100" step="5" value="100">'+
   '<label>mean slope &le; <span id="v_sl" class="val"></span> deg</label><input type="range" id="f_sl" min="0" max="20" step="0.5" value="20">';
  $("presets").innerHTML='<button id="p_open" title="grassland + cropland >= 50 %, tree cover <= 20 %, mean slope <= 5 deg">open land, gentle terrain</button><button id="p_reset">reset</button>'}
function F(){return{gc:$("f_gc")?+$("f_gc").value:0,tr:$("f_tr")?+$("f_tr").value:100,sl:$("f_sl")?+$("f_sl").value:20,sta:$("f_sta").checked,run:$("f_run").checked,hide:$("f_hide").checked,
  mon:[...Array(12).keys()].map(k=>$("m"+k).checked),warm:$("f_warm").checked}}
let MON=null,LATC=null;   // month (0-11) of the middle of each stack window; centre latitude of each tile
function seasonOK(i,f){const m=MON[i];if(!f.mon[m])return false;if(!f.warm)return true;const la=LATC[D.stacks.t[i]];
  return Math.abs(la)<23.5||(la>0?(m>=4&&m<=8):(m>=10||m<=2))}
function labels(f){if($("v_gc")){$("v_gc").textContent=f.gc;$("v_tr").textContent=f.tr>=100?"any":f.tr;$("v_sl").textContent=f.sl>=20?"any":f.sl}}
function tilePass(t,f){if(f.sta&&!D.tiles[t].sta)return false;
  if(f.gc>0){const g=gc(t);if(g===null||g<f.gc)return false}
  if(f.tr<100){const x=lcv("tree_pct",t);if(x===null||x>f.tr)return false}
  if(f.sl<20){const x=lcv("slope_mean_deg",t);if(x===null||x>f.sl)return false}return true}
function update(){const f=F();if(!MON){MON=D.stacks.f.map((a,i)=>new Date(Date.UTC(2025,0,1)+0.5*(a+D.stacks.l[i])*864e5).getUTCMonth());
    LATC=D.tiles.map(T=>0.5*(T.b[1]+T.b[3]))}labels(f);const S=D.stacks,nT=D.tiles.length,v=VARS[$("colour").selectedIndex];
  PASS=new Uint8Array(S.key.length);let ns=0,nt=0;const on=new Uint8Array(nT);
  for(let t=0;t<nT;t++){if(!tilePass(t,f))continue;for(const i of TS[t]){if(f.run&&!D.runs[S.key[i]])continue;if(!seasonOK(i,f))continue;PASS[i]=1;ns++;on[t]=1}if(on[t])nt++}
  TVAL=new Array(nT).fill(null);
  for(let t=0;t<nT;t++){const r=RECTS[t];if(!on[t]){r.setStyle(f.hide?{opacity:0,fillOpacity:0}:{color:"#999",weight:0.4,opacity:0.7,fillOpacity:0});continue}
    const val=v.f(t);TVAL[t]=val;const col=val===null?"#bbbbbb":ramp(PAL[v.pal],(val-v.dom[0])/(v.dom[1]-v.dom[0]));
    r.setStyle({color:"#555",weight:0.4,opacity:1,fillColor:col,fillOpacity:MAP.getZoom()>=7?0.12:0.65})}
  $("count").innerHTML=`${nt.toLocaleString()} of ${nT.toLocaleString()} tiles`;$("count2").innerHTML=`${ns.toLocaleString()} of ${S.key.length.toLocaleString()} stacks`;hist(v);legend(v)}
function hist(v){const vals=TVAL.filter(x=>x!==null),nb=20,[a,b]=v.dom,c=new Array(nb).fill(0);for(const x of vals)c[Math.max(0,Math.min(nb-1,Math.floor((x-a)/(b-a)*nb)))]++;
  const mx=Math.max(1,...c),W=260,H=60;let s=`<svg width="${W}" height="${H+28}" font-size="10">`;
  c.forEach((n,i)=>{const h=n/mx*H;s+=`<rect x="${i*W/nb+1}" y="${H-h}" width="${W/nb-2}" height="${h}" fill="${ramp(PAL[v.pal],(i+0.5)/nb)}" stroke="#666" stroke-width="0.3"><title>${(a+i*(b-a)/nb).toFixed(1)} to ${(a+(i+1)*(b-a)/nb).toFixed(1)}: ${n} tiles</title></rect>`});
  s+=`<text x="0" y="${H+12}">${a}</text><text x="${W}" y="${H+12}" text-anchor="end">&ge;${b}</text><text x="${W/2}" y="${H+25}" text-anchor="middle">${vals.length.toLocaleString()} tiles shown</text></svg>`;$("hist").innerHTML=s}
function legend(v){if(LEG)LEG.remove();LEG=L.control({position:"bottomleft"});LEG.onAdd=()=>{const d=L.DomUtil.create("div","legend");
  d.innerHTML=`<b>${v.lab}</b><br><div class="bar" style="background:linear-gradient(to right,${PAL[v.pal].join(",")})"></div>${v.dom[0]} &ndash; &ge;${v.dom[1]}<br>computed points (click: stack and report)<br>${dot(PCOL.insitu)} with in-situ station data (ISMN)<br>${dot(PCOL.model)} without in-situ data (model reference only)<br>${dot(PCOL.skip)} SAR skipped (no valid BIOMASS pixels)`;return d};LEG.addTo(MAP)}
function lcBar(t){if(!D.lc||lcv("land_pct",t)===null)return '<div class="small">land cover not available</div>';let bar='<div class="lcbar">',key='<div class="key">';
  for(const [c,col] of Object.entries(WCC)){const x=lcv(c+"_pct",t);if(!x||x<0.5)continue;bar+=`<div title="${c} ${x.toFixed(0)} %" style="width:${x}%;background:${col}"></div>`;if(x>=3)key+=`<span><i style="background:${col}"></i>${c} ${x.toFixed(0)} %</span>`}
  return bar+'</div>'+key+'</div>'}
function inBox(bb,ll){let lon=ll.lng;while(lon<bb[0]-180)lon+=360;while(lon>bb[0]+180)lon-=360;return ll.lat>=bb[1]&&ll.lat<=bb[3]&&lon>=bb[0]&&lon<=bb[2]}
function clearPt(){PT=null;if(PTM){PTM.remove();PTM=null}showPt()}
function setPoint(ll){const bb=D.stacks.bb[SELSTACK];if(!inBox(bb,ll)){$("ptmsg").innerHTML=`<span style="color:${OI.verm}">outside the footprint of ${D.stacks.key[SELSTACK]} - click inside the dashed box</span>`;return}
  PT={lat:+ll.lat.toFixed(5),lon:+(((ll.lng+540)%360)-180).toFixed(5)};if(PTM)PTM.remove();
  PTM=L.circleMarker(ll,{renderer:PREN,radius:7,color:"#000",weight:2,fillColor:"#ffffff",fillOpacity:1,interactive:false}).addTo(MAP);showPt()}
function showPt(){const e=$("ptmsg");if(!e)return;const k=D.stacks.key[SELSTACK];
  e.innerHTML=PT?`<b>point ${PT.lat.toFixed(4)}, ${PT.lon.toFixed(4)}</b> (${WINM} m window around it; ISMN stations inside the stack are added) &middot; <a href="#" onclick="clearPt();return false">use the stack centre instead</a>`
    :`no point chosen: the run uses the ISMN stations inside the stack, or the stack centre if there is none. <b>Click inside the dashed box</b> to choose your own point (Esri satellite helps).`;
  const b=$("b_run");if(b)b.textContent=PT?"compute full report at this point":(D.runs[k]?"recompute":"compute full report");
  const ba=$("b_all"),bl=$("allmsg");if(!ba)return;
  if(!PT){ba.style.display="none";bl.innerHTML="";return}
  const ll=L.latLng(PT.lat,PT.lon),S=D.stacks;ALL=[];let nall=0;
  for(let j=0;j<S.key.length;j++)if(inBox(S.bb[j],ll)){nall++;if(PASS[j])ALL.push(j)}
  ALL.sort((x,y)=>S.f[x]-S.f[y]);ba.style.display=ALL.length>1?"inline-block":"none";ba.textContent=`compute all ${ALL.length} stacks at this point`;
  bl.innerHTML=`${nall} stacks cover this point (all tracks, swaths and major cycles), ${ALL.length} pass the filters: `+ALL.map(j=>S.key[j]+" ("+day2iso(S.f[j]).slice(0,7)+")").join(", ")+
    (ALL.length>1?`. One run per stack, one after the other (several minutes each).`:"")}
function showTile(t){SELTILE=t;clearPt();if(FOOT)FOOT.remove();FOOT=L.layerGroup(TS[t].map(i=>{const b=D.stacks.bb[i];
    return L.rectangle([[b[1],b[0]],[b[3],b[2]]],{renderer:REN,color:"#333",weight:0.8,dashArray:"2 4",fill:false,interactive:false})})).addTo(MAP);const T=D.tiles[t],S=D.stacks,idx=TS[t].slice().sort((a,b)=>S.f[a]-S.f[b]),c=[(T.b[1]+T.b[3])/2,(((T.b[0]+T.b[2])/2+540)%360)-180];
  let h=`<h1>Tile ${T.id}</h1><div class="small">centre ${c[0].toFixed(2)} N, ${c[1].toFixed(2)} E &middot; ${idx.length} stacks${T.sta?"<br>ISMN stations: "+T.sta:""}</div>`;
  h+=`<h2>Land cover (WorldCover 2021, % of land; land = ${fmt(lcv("land_pct",t),0)} % of the box) and terrain</h2>`+lcBar(t)+`<div class="small">mean slope ${fmt(lcv("slope_mean_deg",t),1)} deg, 90th percentile ${fmt(lcv("slope_p90_deg",t),1)} deg, mean elevation ${fmt(lcv("elev_mean_m",t),0)} m (Copernicus DEM at 200 m; information)</div>`;
  h+=`<div class="small">The coloured tile box is the union of all stacks with this track and frame number. One stack covers only about
    95 km x 150 km (thin dotted outlines; the selected stack dashed): the three swaths S1-S3 are offset by about 40 km (median of all
    tiles), and the same track/frame lies about 100 km further west in each later major cycle M (median; 10-90 %: 31-130 km). The land cover above is the average of the whole box.</div>`;
  h+=`<h2>Stacks (click for quicklooks)</h2><table><tr><th>stack</th><th>first acquisition</th><th>days</th><th>n</th><th>orbit</th><th>computed</th></tr>`;
  for(const i of idx)h+=`<tr class="st" data-i="${i}"><td>${S.key[i]}</td><td>${day2iso(S.f[i])}</td><td>${S.l[i]-S.f[i]}</td><td>${S.n[i]}</td><td>${S.o[i]==="a"?"asc":"desc"}</td><td style="white-space:nowrap">${D.runs[S.key[i]]?D.runs[S.key[i]].slice(0,6).map(r=>dot(pcol(r))).join("")+(D.runs[S.key[i]].length>6?"&hellip;":"")+" "+D.runs[S.key[i]].length+" pt":""}</td></tr>`;
  h+=`</table><div id="detail"></div>`;$("side").innerHTML=h;
  for(const tr of $("side").querySelectorAll("tr.st"))tr.onclick=()=>showStack(+tr.dataset.i);
  const pick=idx.find(i=>D.runs[S.key[i]])??idx[idx.length-1];if(pick!==undefined)showStack(pick)}
function thumb(pid){const m=pid.match(/_(\d{4})(\d{2})(\d{2})T/);return m?`https://catalog.maap.eo.esa.int/data/biomass-pdgs-01/BiomassLevel1c/${m[1]}/${m[2]}/${m[3]}/${pid}/public/${pid}.BI.JPG`:""}
function target(){try{return window.SM_TARGET?window.parent.document.querySelector("."+window.SM_TARGET+" input"):null}catch(e){return null}}
function req(kind,key){const inp=target(),m=$("reqmsg");
  if(!inp){m.innerHTML=`Computing needs the notebook (select_tile.ipynb on MAAP). From a terminal: <code>python run_stack.py --stack ${key}${PT?` --lat ${PT.lat} --lon ${PT.lon}`:""} --half-m ${WINM/2} &amp;&amp; python stack_report.py ~/data/runs/${key}</code>`;return}
  inp.value=kind+"|"+key+"|"+Date.now()+"|"+(kind==="run"&&PT?PT.lat+"|"+PT.lon:"|")+"|"+WINM;inp.dispatchEvent(new Event("input",{bubbles:true}));
  m.innerHTML=kind==="run"?`<b>computing ${key}</b> - follow the run log below the dashboard; the full report opens there when it is done (several minutes)`:`the full report of <b>${key}</b> opens below the dashboard`}
function reqAll(){if(!PT||ALL.length<2)return;const inp=target(),m=$("reqmsg"),keys=ALL.map(j=>D.stacks.key[j]);
  if(!inp){m.innerHTML=`Computing needs the notebook. From a terminal: <code>for s in ${keys.join(" ")}; do python run_stack.py --stack $s --lat ${PT.lat} --lon ${PT.lon} --half-m ${WINM/2}; done</code>`;return}
  inp.value="runall|"+keys.join(",")+"|"+Date.now()+"|"+PT.lat+"|"+PT.lon+"|"+WINM;inp.dispatchEvent(new Event("input",{bubbles:true}));
  m.innerHTML=`<b>computing ${keys.length} stacks</b> at the point one after the other - follow the run log below the dashboard`}
function showStack(i){const S=D.stacks;SELSTACK=i;let h=`<h2>${S.key[i]}: ${day2iso(S.f[i])} to ${day2iso(S.l[i])}, ${S.n[i]} acquisitions, ${S.o[i]==="a"?"ascending":"descending"}</h2>`;
  const k=S.key[i];h+=D.runs[k]?`<button class="act" onclick="req('show','${k}')">open full report</button><button id="b_run" onclick="req('run','${k}')">recompute</button>`:`<button class="act" id="b_run" onclick="req('run','${k}')">compute full report</button>`;
  h+=`<div class="small">analysis window <select id="winm" onchange="WINM=+this.value;showPt()">${[200,400,600,1000,1500,2000].map(v=>`<option value="${v}"${v===WINM?" selected":""}>${v} m x ${v} m</option>`).join("")}</select>
    (cells of 100 m are averaged inside; larger = more pixels, less local)</div>`;
  h+=`<div class="small" id="ptmsg"></div><button id="b_all" style="display:none" onclick="reqAll()"></button><div class="small" id="allmsg"></div><div class="small" id="reqmsg"></div>`;
  h+=`<div class="small">BPS quicklooks (public, radar geometry)</div><div class="thumbs">`;
  S.pid[i].forEach((p,k)=>{h+=`<figure><img loading="lazy" src="${thumb(p)}" onerror="this.parentNode.style.display='none'"><figcaption>${S.acq[i][k]!==undefined?day2iso(S.acq[i][k]).slice(5):""}</figcaption></figure>`});h+=`</div>`;
  const R=D.runs[S.key[i]];
  if(R){h+=`<h2>BIOMASS computed (run_stack.py)</h2><div class="tw"><table><tr><th>point</th><th>reference</th><th>SM range</th><th>&Delta;&sigma;0 HH [dB]</th><th>r_diff</th><th>p</th><th>coh 3 d</th></tr>`;
    for(const r of R)h+=`<tr><td>${dot(pcol(r))} ${r.station}</td><td>${r.ref_used||""}</td><td>${fmt(r.ref_range,3)}</td><td>${fmt(r.dsigma0_HH_dB,2)}</td><td>${fmt(r.r_diff_HH,2)}</td><td>${fmt(r.r_diff_p_HH,3)}</td><td>${fmt(r.mean_coh_HH,2)}</td></tr>`;
    h+=`</table></div><p class="small">SM range = range of the soil-moisture reference over the 7 acquisitions [m3/m3]; r_diff = correlation of first differences of sigma0 HH and the reference (p = two-sided p-value; 7 acquisitions = 6 differences need |r| &ge; 0.81 for p &lt; 0.05, 6 acquisitions &ge; 0.88).</p>`;
    for(const r of R){if(r.skipped)h+=`<p class="small">${r.station}: SAR skipped - ${r.skipped}</p>`;else if(r.fig)h+=`<div class="fig"><div class="small"><b>${r.station}</b>: main figure</div><img src="${r.fig}"></div>`}}
  else h+=`<p class="small">Not computed yet: 'compute full report' runs BIOMASS for this stack (the ISMN stations inside its footprint and your point; without either, the stack centre).</p>`;
  $("detail").innerHTML=h;for(const tr of $("side").querySelectorAll("tr.st"))tr.classList.toggle("sel",+tr.dataset.i===i);
  const bb=S.bb[i];if(BOX)BOX.remove();BOX=L.rectangle([[bb[1],bb[0]],[bb[3],bb[2]]],{renderer:REN,color:"#000",weight:1.5,dashArray:"5 4",fill:false,interactive:false}).addTo(MAP);
  if(PT&&!inBox(bb,L.latLng(PT.lat,PT.lon)))clearPt();showPt()}
function preset(p){if($("f_gc")){$("f_gc").value=p.gc||0;$("f_tr").value=p.tr??100;$("f_sl").value=p.sl??20}$("f_sta").checked=false;$("f_run").checked=false;
  if(!Object.keys(p).length){for(let k=0;k<12;k++)$("m"+k).checked=true;$("f_warm").checked=false}update()}
function exportCsv(){const S=D.stacks;let s="stack,lat,lon,name\n",n=0;for(let i=0;i<S.key.length;i++)if(PASS[i]){const b=D.tiles[S.t[i]].b;n++;
  s+=`${S.key[i]},${((b[1]+b[3])/2).toFixed(4)},${((((b[0]+b[2])/2)+540)%360-180).toFixed(4)},centre_${S.key[i]}\n`}
  const ta=$("exptxt");ta.style.display="block";ta.value=s;ta.select();if(navigator.clipboard)navigator.clipboard.writeText(s).catch(()=>{});$("exp").textContent=`copied ${n} stacks`}
(async function(){D=await load();const M=D.meta;TS=D.tiles.map(()=>[]);D.stacks.t.forEach((t,i)=>TS[t].push(i));PASS=new Uint8Array(D.stacks.key.length).fill(1);VARS=vars();lcFilters();
  $("meta").innerHTML=`${M.n_tiles.toLocaleString()} tiles (track x frame), ${M.n_stacks.toLocaleString()} stacks with &ge; 6 acquisitions (3-day repeat), ${day2iso(M.d_min)} to ${day2iso(M.d_max)}; ${M.n_sta} tiles with an ISMN station; ${Object.keys(D.runs).length} stacks already computed. Built ${M.built}.`;
  $("caveat").innerHTML=`Land cover (ESA WorldCover 2021) and slope (Copernicus DEM, 200 m) are tile averages over the bounding box of the tile's stacks (about 5&deg; x 1.5&deg;) - information to find suitable test sites, not a BIOMASS result. The soil-moisture comparison is in the computed stacks.`;
  VARS.forEach(v=>{const o=document.createElement("option");o.textContent=v.lab;$("colour").appendChild(o)});
  MAP=L.map("map",{preferCanvas:true,worldCopyJump:true,minZoom:2}).setView([20,10],2);
  const osm=L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",{attribution:"OpenStreetMap",maxZoom:12});
  const esri=L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",{attribution:"Esri",maxZoom:12});
  MAP.createPane("land").style.zIndex=350;const none=L.layerGroup();none.addTo(MAP);[-360,0,360].forEach(k=>L.geoJSON(D.land,{pane:"land",interactive:false,coordsToLatLng:c=>L.latLng(c[1],c[0]+k),style:{color:"#777",weight:0.6,fillColor:"#f2f2ee",fillOpacity:1}}).addTo(none));
  L.control.layers({"coastlines only (offline)":none,"OpenStreetMap":osm,"Esri satellite":esri},{},{collapsed:false,position:"topright"}).addTo(MAP);
  REN=L.canvas({padding:0.5});MAP.createPane("pts").style.zIndex=450;PREN=L.svg({pane:"pts"});   // one canvas for all boxes (a second canvas would swallow the clicks); points as SVG on top
  D.stacks.key.forEach((k,i)=>KEY2I[k]=i);const ren=REN;
  D.tiles.forEach((T,t)=>{const r=L.rectangle([[T.b[1],T.b[0]],[T.b[3],T.b[2]]],{renderer:ren,weight:0.4});r.on("click",e=>{window.__lastclick="tile "+t;if(SELSTACK!==null&&inBox(D.stacks.bb[SELSTACK],e.latlng))return;
    L.DomEvent.stopPropagation(e);if(t===SELTILE&&SELSTACK!==null){setPoint(e.latlng);return}showTile(t)});
    r.bindTooltip(()=>{const v=VARS[$("colour").selectedIndex];return `${T.id} &middot; ${TS[t].length} stacks${TVAL&&TVAL[t]!==null?" &middot; "+(+TVAL[t]).toFixed(v.nd):""}${T.sta?" &middot; ISMN":""}`},{sticky:true});r.addTo(MAP);RECTS.push(r)});
  MAP.on("click",e=>{window.__lastclick=(window.__lastclick||"")+" map";if(SELSTACK!==null)setPoint(e.latlng)});   // setPoint warns if outside
  for(const [k,R] of Object.entries(D.runs)){const i=KEY2I[k];if(i===undefined)continue;
    for(const r of R){if(r.lat===null||r.lon===null||r.lat===undefined)continue;
      const mk=L.circleMarker([r.lat,r.lon],{renderer:PREN,radius:6,color:"#000",weight:1,fillColor:pcol(r),fillOpacity:1});
      mk.bindTooltip(`<b>${r.station}</b> (${r.network||""}) &middot; ${k}<br>${r.has_ismn?"in-situ station data (ISMN)":"no in-situ data (model reference only)"}<br>${r.skipped?"SAR skipped":`&Delta;&sigma;0 HH ${fmt(r.dsigma0_HH_dB,2)} dB &middot; r_diff ${fmt(r.r_diff_HH,2)} &middot; ref ${r.ref_used||""}`}`);
      mk.on("click",e=>{L.DomEvent.stopPropagation(e);showTile(D.stacks.t[i]);showStack(i)});mk.addTo(MAP)}}
  let pend=false;const sched=()=>{if(pend)return;pend=true;requestAnimationFrame(()=>{pend=false;update()})};
  MAP.on("zoomend",sched);                                   // zoomed in: almost transparent tiles, so the base map shows through
  document.querySelectorAll("#ctl input,#ctl select").forEach(e=>{e.addEventListener("input",sched);e.addEventListener("change",sched)});
  if($("p_open")){$("p_open").onclick=()=>preset({gc:50,tr:20,sl:5});$("p_reset").onclick=()=>preset({})}$("exp").onclick=exportCsv;
  update();$("loading").remove();window.__dash={showTile,showStack,update,setPoint,get PT(){return PT},get D(){return D},get MAP(){return MAP},get PASS(){return PASS}}})().catch(e=>{$("loading").textContent="could not load the data: "+e});
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(HERE / "l1c_stack_index.csv")); ap.add_argument("--items", default=str(HERE / "l1c_items_index.parquet"))
    ap.add_argument("--landcover", default=str(HERE / "tile_landcover.csv")); ap.add_argument("--runs", default="~/data/runs", help="run root with all_tiles_summary.csv ('' = none)")
    ap.add_argument("--out", default="dashboard.html")
    a = ap.parse_args()
    import sys; sys.path.insert(0, str(HERE)); from sm_pipeline import deps; deps.ensure(polsarpro=False)   # pyarrow, Pillow, ...
    build(a.index, a.items, a.landcover, a.runs or None, a.out)
