import xml.etree.ElementTree as ET
<<<<<<< HEAD

=======
import threading, time, json
from concurrent.futures import ThreadPoolExecutor
from aiohttp import ClientResponseError
import fsspec

# XML tag (namespace stripped) -> short key used in the output dict
_FM_REPORT = {"rfiIsolatedFMReport": "iso", "rfiPersistentFMReport": "pers"}
_FM_FIELD = {
    "avgPercentageAffectedBW": "avgBW",
    "maxPercentageAffectedBW": "maxBW",
    "percentageAffectedLines": "linesPct",
}


def _local(tag):
    """Tag name without its XML namespace, e.g. '{ns}foo' -> 'foo'."""
    return tag.split("}")[-1]


def rfi_fm_stats(root):
    """Extract frequency-mitigation (FM) statistics from th

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        Root of the parsed annotation XML.

    Returns
    -------
    dict
        Flat dict keyed '<kind>_<pol>_<stat>', e.g. {'iso_H
        'pers_HH_linesPct': 100.0}. kind is 'iso' (isolated) or 'pers' (persistent),
        pol is the polarisation, stat is 'avgBW', 'maxBW' o
    """
    out = {}
    for e in root.iter():
        kind = _FM_REPORT.get(_local(e.tag))
        if not kind:
            continue
        pol = e.get("polarisation")
        for c in e:
            key = _FM_FIELD.get(_local(c.tag))
            if key:
                out[f"{kind}_{pol}_{key}"] = float(c.text)
    return out


def read_annot_file(prod_id, url, token):
    """Stream a product annotation XML and report its RFI p

    Opens the file directly over HTTPS (no local download)
    then reads rfiDetectionFlag, rfiCorrectionFlag, the mitigation properties
    and the FM statistics.

    Parameters
    ----------
    prod_id : str
        Product name, e.g.
        'BIO_S1_SCS__1S_20260514T003311_20260514T003332_T_GE'.
    url : str
        HTTPS URL of the annotation XML file.
    token : str
        Bearer token for HTTP authentication.

    Returns
    -------
    list
        [] if no RFI was detected. Otherwise a one-element list with a dict holding
        'id', 'rfiCorrectionFlag', 'properties' (rfiMitigat
        rfiFMMitigationMethod, rfiMask) and 'fm' (see rfi_fm_stats).
    """
    fs = fsspec.filesystem("https", headers={"Authorization": f"Bearer {token}"})
    with fs.open(url, "rb") as f:
        root = ET.parse(f).getroot()

    def get(name):
        """Text of the first element called `name` (namespaces ignored), or None."""
        for e in root.iter():
            if _local(e.tag) == name:
                return (e.text or "").strip()
        return None

    detected = (get("rfiDetectionFlag") or "").lower() == "true"
    print(f"rfiDetectionFlag = {get('rfiDetectionFlag')}")
    print(f"rfiCorrectionFlag = {get('rfiCorrectionFlag')}")
    if not detected:
        return []

    return [
        {
            "id": prod_id,
            "rfiCorrectionFlag": get("rfiCorrectionFlag"),
            "properties": {
                "rfiMitigationMethod": get("rfiMitigationMethod"),
                "rfiFMMitigationMethod": get("rfiFMMitigationMethod"),
                "rfiMask": get("rfiMask"),  # 'Single': same for all pols; 'Multiple': one per pol
            },
            "fm": rfi_fm_stats(root),
        }
    ]
