"""Step 1: resolve a stack id to its products and asset URLs (public MAAP STAC) and load the user token."""
import os, json, urllib.request, collections
import pandas as pd
STAC = "https://catalog.maap.eo.esa.int/catalogue/collections/{col}/items"
# offline token: first match of ~/token.txt, ~/my-private-bucket/token.txt (one line, from portal.maap.eo.esa.int -> services -> auth -> token)
TOKEN_FILE = next((p for p in (os.path.expanduser("~/token.txt"), os.path.expanduser("~/my-private-bucket/token.txt")) if os.path.exists(p)),
                  os.path.expanduser("~/my-private-bucket/token.txt"))
ASSET_KEYS = {"abs_tiff": "enclosure_i_abs_tiff", "phase_tiff": "enclosure_i_phase_tiff", "lut_nc": "enclosure_nc", "vrt": "enclosure_vrt",
              "annot_xml_1": "enclosure_annot_xml_1", "annot_xml_2": "enclosure_annot_xml_2", "ql_pauli_png": "enclosure_ql_pauli_png",
              "ql_coh_png": "enclosure_coh_ql_png", "ql_itf_png": "enclosure_itf_ql_png", "thumbnail_public": "quicklook_jpg", "zip": "product"}

# MAAP auth: token.txt holds the long-lived OFFLINE token from portal.maap.eo.esa.int; data servers want a short-lived ACCESS
# token obtained by exchanging it at the ESA MAAP IAM (public 'offline-token' client, same constants as in the BioPAL hackathon notebooks).
IAM_URL = "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token"
IAM_CLIENT_ID = "offline-token"
IAM_CLIENT_SECRET = "p1eL7uonXs6MDxtGbgKdPVRAmnGxHpVE"
_ACCESS = {"token": None, "exp": 0}

def access_token(path=TOKEN_FILE, force=False):
    """Exchange the offline token for an access token (cached until ~1 min before expiry)."""
    import time, requests
    if not force and _ACCESS["token"] and time.time() < _ACCESS["exp"] - 60: return _ACCESS["token"]
    if not os.path.exists(os.path.expanduser(path)):
        raise RuntimeError("no MAAP offline token found: put your own token (one line, portal.maap.eo.esa.int -> services -> auth -> token)"
                           " into ~/my-private-bucket/token.txt or ~/token.txt. The dashboard and --no-sar runs work without it.")
    offline = open(os.path.expanduser(path)).read().strip().replace("\n", "")
    r = requests.post(IAM_URL, data={"client_id": IAM_CLIENT_ID, "client_secret": IAM_CLIENT_SECRET, "grant_type": "refresh_token",
                                     "refresh_token": offline, "scope": "offline_access openid"}, timeout=60)
    if r.status_code != 200: raise RuntimeError(f"IAM token exchange failed: HTTP {r.status_code} {r.text[:200]} -> offline token in {path} expired or mis-copied?")
    j = r.json(); _ACCESS.update(token=j["access_token"], exp=time.time() + int(j.get("expires_in", 300))); return _ACCESS["token"]

def token(path=TOKEN_FILE):
    """Bearer token for data access = the exchanged ACCESS token (not the offline token itself)."""
    return access_token(path)

def _get(url, tries=4):
    """GET JSON with retries; the MAAP catalogue sometimes stalls for a minute (read timeouts), so wait and retry."""
    import time
    last = None
    for i, to in enumerate([60, 120, 180, 240][:tries]):
        try:
            with urllib.request.urlopen(url, timeout=to) as r: return json.loads(r.read())
        except Exception as e:
            last = e; print(f"[catalog] attempt {i+1} failed ({type(e).__name__}); retrying in {10*(i+1)} s", flush=True); time.sleep(10 * (i + 1))
    raise last

def stack_key(pid):
    p = pid.split("_"); trk = [x for x in p if len(x) == 4 and x[0] == "T" and x[1:].isdigit()][-1]
    frm = [x for x in p if len(x) == 4 and x[0] == "F" and x[1:].isdigit()][-1]
    return "_".join([p[1], trk, frm, [x for x in p if x.startswith("G0")][0], [x for x in p if x.startswith("M0")][0]])

LOCAL_ITEMS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "l1c_items_index.parquet")

def stacks_at_local(lat, lon, path=LOCAL_ITEMS):
    """Offline fallback: stacks whose item bbox contains the point, from the local L1C item index (snapshot, L1C only)."""
    it = pd.read_parquet(path); hit = it[(it.w <= lon) & (it.e >= lon) & (it.s <= lat) & (it.n >= lat)]
    out = collections.defaultdict(dict)
    for _, r in hit.iterrows(): out[r["stack"]][r["dt"][:10]] = r["pid"]
    return dict(out)

def stacks_at(lat, lon, collection="BiomassLevel1c", halfwidth=0.05, allow_local=True):
    """All stacks (dict stack_key -> {date: product_id}) covering a point. Live catalogue first; if it keeps timing out and
    the collection is L1C, fall back to the local item index snapshot (built by build_tile_index.py)."""
    bb = f"{lon-halfwidth},{lat-halfwidth},{lon+halfwidth},{lat+halfwidth}"
    try:
        feats = _get(STAC.format(col=collection) + f"?limit=200&bbox={bb}").get("features", [])
    except Exception as e:
        if allow_local and collection == "BiomassLevel1c" and os.path.exists(LOCAL_ITEMS):
            print(f"[catalog] live query failed ({type(e).__name__}); using local index snapshot {LOCAL_ITEMS}"); return stacks_at_local(lat, lon)
        raise
    out = collections.defaultdict(dict)
    for f in feats: out[stack_key(f["id"])][f["properties"]["start_datetime"][:10]] = f["id"]
    return dict(out)

def stack_table(product_ids, collection="BiomassLevel1c"):
    """One row per product with datetime, primary flag, orbit and asset URLs (sorted by time)."""
    rows = []
    for pid in product_ids:
        it = _get(STAC.format(col=collection) + "/" + pid); A, p = it["assets"], it["properties"]
        rows.append(dict(stack=stack_key(pid), pid=pid, datetime=p["start_datetime"][:19], primary=p.get("eofeos:is_coregistration_primary", False),
                         orbit=p.get("sat:absolute_orbit"), orbit_state=p.get("sat:orbit_state"), stack_id=p.get("eofeos:stack_id"),
                         **{k: A.get(v, {}).get("href") for k, v in ASSET_KEYS.items()}))
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
