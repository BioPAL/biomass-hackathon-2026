"""Check the Python packages the pipeline needs and install the missing ones with pip (nothing happens if all are present).

Called at the start of run_stack.py (and therefore tests/replot_from_npz.py), run_batch.py, build_dashboard.py,
stack_report.py and in the first cell of select_tile.ipynb. Outside a virtualenv pip installs with --user, which survives a
restart of the MAAP server. PyPolSARpro (optional; polarimetric partitions and H/A/alpha) is installed from the pinned
GitHub commit with --no-deps, after its import dependencies dask and xarray; it needs Python >= 3.12.
A failed install is not retried for 12 h (file ~/.cache/sm_pipeline/pip_failed.json; delete it to retry at once).
Set the environment variable SM_PIPELINE_NO_INSTALL=1 to only report missing packages.
"""
import importlib, importlib.util, json, os, pathlib, site, subprocess, sys, time

# pip name -> import name (requirements.txt)
REQUIRED = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "matplotlib": "matplotlib", "rasterio": "rasterio",
            "netCDF4": "netCDF4", "xarray": "xarray", "requests": "requests", "folium": "folium", "ipywidgets": "ipywidgets",
            "pyarrow": "pyarrow", "Pillow": "PIL"}
POLSARPRO_DEPS = {"dask": "dask", "xarray": "xarray"}           # what PyPolSARpro imports beyond numpy/scipy/matplotlib
POLSARPRO_PIP = "git+https://github.com/satim-co/PolSARpro@bea8352fec10496f1a78d9b9992154006d8e0514"
FAILED_FILE = pathlib.Path.home() / ".cache" / "sm_pipeline" / "pip_failed.json"
RETRY_AFTER_S = 12 * 3600


def missing(pkgs=None):
    """pip names of the packages in `pkgs` (default REQUIRED) that cannot be imported."""
    return [p for p, mod in (pkgs or REQUIRED).items() if importlib.util.find_spec(mod) is None]


def _failed():
    try: return json.loads(FAILED_FILE.read_text())
    except Exception: return {}


def _pip(names, log, extra=()):
    """pip install `names`; returns True on success. Remembers failures so that batch runs do not retry every stack."""
    failed = _failed(); now = time.time()
    names = [n for n in names if now - failed.get(n, 0) > RETRY_AFTER_S]
    if not names: return False
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)        # --user is not allowed inside a virtualenv
    cmd = [sys.executable, "-m", "pip", "install", "-q", *([] if in_venv else ["--user"]), *extra, *names]
    log("[deps] installing with pip: " + " ".join([*extra, *names]))
    ok = subprocess.run(cmd).returncode == 0
    if not ok:
        failed.update({n: now for n in names})
        try: FAILED_FILE.parent.mkdir(parents=True, exist_ok=True); FAILED_FILE.write_text(json.dumps(failed))
        except Exception: pass
        log(f"[deps] pip failed for {names}; not retried for 12 h (delete {FAILED_FILE} to retry)")
    site.addsitedir(site.getusersitepackages()); importlib.invalidate_caches()   # new --user packages importable right away
    return ok


def ensure(polsarpro=True, log=print):
    """Install missing required packages (and PyPolSARpro if `polsarpro`). Returns the pip names still missing afterwards."""
    miss = missing()
    want_psp = polsarpro and importlib.util.find_spec("polsarpro") is None
    if want_psp and sys.version_info < (3, 12):
        want_psp = False
        if not os.environ.get("SM_PIPELINE_QUIET_PSP"):
            log(f"[deps] PyPolSARpro needs Python >= 3.12 (this kernel: {sys.version.split()[0]}): polarimetric partitions skipped")
            os.environ["SM_PIPELINE_QUIET_PSP"] = "1"
    if not miss and not want_psp: return []
    if os.environ.get("SM_PIPELINE_NO_INSTALL"):
        log(f"[deps] missing: {miss + (['PyPolSARpro'] if want_psp else [])} (auto-install off: SM_PIPELINE_NO_INSTALL is set)")
        return miss
    had_widgets = importlib.util.find_spec("ipywidgets") is not None
    if miss: _pip(miss, log)
    if want_psp:
        dmiss = missing(POLSARPRO_DEPS)
        if dmiss: _pip(dmiss, log)
        if not missing(POLSARPRO_DEPS): _pip([POLSARPRO_PIP], log, extra=("--no-deps",))
        if importlib.util.find_spec("polsarpro") is None: log("[deps] PyPolSARpro not available: polarimetric partitions skipped")
    left = missing()
    if left: log(f"[deps] still missing: {left} - install by hand: pip install --user {' '.join(left)}")
    if not had_widgets and importlib.util.find_spec("ipywidgets") is not None and "ipykernel" in sys.modules:
        log("[deps] ipywidgets was just installed: reload the browser page (F5), then run the cell again")
    return left
