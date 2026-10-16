"""Dashboard app for the notebook: the global dashboard (all BIOMASS L1C tiles) with a 'compute full report' button per stack.

    from sm_pipeline.app import App
    App().show()

Click a tile, then a stack (optionally a point inside its dashed footprint and the analysis window), then 'compute full report':
run_stack.py runs for that stack (the ISMN stations inside the footprint and the chosen point; without either, the stack centre),
the log streams below the dashboard, and the full report (<out>/<stack>/report.html, all figures and numbers) is shown below and
the dashboard is rebuilt so that the points appear as computed. For computed stacks the button is 'open full report'.
How it works: the dashboard runs in an iframe; its buttons write '<kind>|<stack(s)>|<time>|<lat>|<lon>|<window m>' into a hidden
text widget of this app.
"""
import contextlib, html, importlib.util, io, os, pathlib, subprocess, sys, threading, uuid
from . import reference

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _script(name):
    """Import one of the package's top-level scripts (build_dashboard.py, stack_report.py) by path."""
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _existing(p):
    p = os.path.expanduser(p) if p else None
    return p if p and os.path.exists(p) else None


class App:
    def __init__(self, out="~/data/runs", ismn_geojson=reference.ISMN_GEOJSON_DEFAULT, ismn_dir=reference.ISMN_DIR_DEFAULT, height=780):
        import ipywidgets as W
        self.W, self.height, self.busy = W, height, False
        self.out = pathlib.Path(os.path.expanduser(out)); self.ismn, self.ismn_dir = _existing(ismn_geojson), _existing(ismn_dir)
        self.tag = "sm-req-" + uuid.uuid4().hex[:8]
        self.req = W.Text(value="", continuous_update=True, layout=W.Layout(display="none")); self.req.add_class(self.tag)
        self.req.observe(self._on_request, names="value")
        self.dash, self.status, self.report = W.HTML(), W.HTML(), W.HTML()
        self.log = W.Output(layout=W.Layout(max_height="320px", overflow="auto", border="1px solid #ddd"))
        self.logbox = W.Accordion(children=[self.log]); self.logbox.set_title(0, "run log"); self.logbox.selected_index = None
        self.rebuild = W.Button(description="rebuild dashboard", tooltip="re-read land cover and computed runs from disk")
        self.rebuild.on_click(lambda _: self._dashboard(force=True))

    # ---- dashboard -----------------------------------------------------------------------------------------------------
    def _dashboard(self, force=False):
        f = ROOT / "dashboard.html"
        if force or not f.exists():
            self.status.value = "building the dashboard (about 30 s) ..."
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _script("build_dashboard").build(str(ROOT / "l1c_stack_index.csv"), str(ROOT / "l1c_items_index.parquet"),
                                                 str(ROOT / "tile_landcover.csv"), str(self.out), str(f))
            self.log.append_stdout(buf.getvalue())
        doc = f.read_text().replace("<head>", f'<head><script>window.SM_TARGET="{self.tag}";</script>', 1)
        self.dash.value = f'<iframe srcdoc="{html.escape(doc, quote=True)}" style="width:100%;height:{self.height}px;border:1px solid #ccc"></iframe>'
        self.status.value = "Click a tile, then a stack, then <b>compute full report</b> (or <b>open full report</b> for computed stacks)."

    def show(self):
        from IPython.display import display
        self._dashboard()
        display(self.W.VBox([self.dash, self.W.HBox([self.status, self.rebuild]), self.req, self.logbox, self.report]))

    # ---- requests from the dashboard ----------------------------------------------------------------------------------------
    def _on_request(self, change):
        parts = str(change["new"]).split("|") + ["", "", "", "", ""]
        kind, key, lat, lon, win = parts[0], parts[1], parts[3], parts[4], parts[5]
        pt = (float(lat), float(lon)) if lat and lon else None
        half = float(win) / 2 if win else None                     # analysis window chosen in the dashboard [m] -> --half-m
        if kind == "run" and key: self.run(key, pt, half)
        elif kind == "runall" and key: self.run_many([k for k in key.split(",") if k], pt, half)
        elif kind == "show" and key: self.show_report(key)

    def run(self, key, point=None, half_m=None):
        """run_stack.py for one stack in a background thread (the notebook stays usable), then report + dashboard rebuild.
        point = (lat, lon) chosen in the dashboard, else the stack centre (if there is no station); half_m = half the analysis
        window chosen in the dashboard (default 300 m = 600 m window); ISMN stations inside the stack footprint are always added."""
        self.run_many([key], point, half_m)

    def run_many(self, keys, point=None, half_m=None):
        """Several stacks one after the other (e.g. all stacks that cover the chosen point = all dates there), same point.
        Each stack gets its own run folder and report; the dashboard is rebuilt at the end, the last report is shown."""
        if self.busy: self.status.value = "<span style='color:#D55E00'>a run is still going - wait until it has finished</span>"; return
        if not keys: return
        self.busy = True
        self.log.clear_output(); self.logbox.selected_index = 0; self.report.value = ""
        where = (f" at {point[0]:.4f}, {point[1]:.4f}" if point else " (stack centre + ISMN stations)") + (f", {2 * half_m:g} m window" if half_m else "")

        def cmd(key):
            return ([sys.executable, str(ROOT / "run_stack.py"), "--stack", key, "--out", str(self.out)] + (["--ismn", self.ismn] if self.ismn else [])
                    + (["--ismn-dir", self.ismn_dir] if self.ismn_dir else [])
                    + (["--lat", f"{point[0]:.5f}", "--lon", f"{point[1]:.5f}"] if point else [])
                    + (["--half-m", f"{half_m:g}"] if half_m else []))

        def work():
            done, failed = [], []
            try:
                for q, key in enumerate(keys):
                    self.status.value = f"computing <b>{key}</b>{where} ({q + 1} of {len(keys)}) - several minutes per stack; the log is below"
                    c = cmd(key); self.log.append_stdout(f"\n==== {q + 1}/{len(keys)} {key}\nrunning: " + " ".join(c) + "\n")
                    p = subprocess.Popen(c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT))
                    for line in p.stdout: self.log.append_stdout(line)
                    p.wait(); self.log.append_stdout(f"[exit {p.returncode}]\n")
                    ok = p.returncode == 0 and (self.out / key / "products.csv").exists()
                    (done if ok else failed).append(key)
                    if ok:
                        try: self.show_report(key, rebuild=True)
                        except Exception as ex: self.log.append_stdout(f"[report of {key} failed: {ex}]\n")
                if done: self._dashboard(force=True)
                msg = f"computed {len(done)} of {len(keys)} stack(s){where}" + (f"; report of <b>{done[-1]}</b> below" if done else "")
                if failed: msg += f"; <span style='color:#D55E00'>failed: {', '.join(failed)} (see the run log)</span>"
                self.status.value = msg
            except Exception as ex:
                self.status.value = f"<span style='color:#D55E00'>run failed: {html.escape(str(ex)[:200])}</span>"
            finally:
                self.busy = False
        threading.Thread(target=work, daemon=True).start()

    def show_report(self, key, rebuild=False):
        run = self.out / key; f = run / "report.html"
        if not (run / "products.csv").exists():
            self.status.value = f"{key} has not been computed in {self.out}"; return
        if rebuild or not f.exists():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf): _script("stack_report").build(run)
            self.log.append_stdout(buf.getvalue())
        self.report.value = f'<iframe srcdoc="{html.escape(f.read_text(), quote=True)}" style="width:100%;height:950px;border:1px solid #ccc"></iframe>'
        self.status.value = f"Full report of <b>{key}</b> below (file {f})."
