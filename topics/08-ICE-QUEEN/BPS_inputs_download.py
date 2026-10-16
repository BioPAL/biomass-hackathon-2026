# BPS_inputs_download is a set of functions that allows users to download automatically all the necessary input files to be used within the ESA BPS, given one single Biomass L0S (RAW_0S) product id.
# DISCLAIMER: this is a first version with limited functionalities. Further improvements and other features will be implemented.

#########################################################################################################################
########################## For any information_______Joseph Melizza (joseph.melizza@serco.com) ##########################
#########################################################################################################################

# Version 0.1.3

# Fixes in v.0.1.1: 
# 1. output products to be searched also considering the RAW_0S input baseline (productVersion)
# 2. metadata values are now extracted from the ogc_17 file (metadata_ogc_17_003r2) instead of the ogc_10 file (metadata_ogc_10_157r4)
# 3. output searching criteria to be displayed after the script is launched (track number, frame number, global coverage, repeat cycle, major cycle and baseline)

# Fixes in v.0.1.2: 
# 1. added bps_l0_extraction_tool in function get_NetCDF

# Fixes in v.0.1.3: 
# 1. added functions to check duplicate products within the catalogue and to check RAW_0M products with same repeat cycle (C) with matching time interval:
#    - _decode_timestamp
#    - _group_key
#    - dedupe_most_recent
#    - filter_matching_products
#    - _parse_item
#    - check_major_cycle_0M



from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET
from pystac_client import Client
from pathlib import Path
from tqdm import tqdm
import urllib.request
import numpy as np
import requests
import pathlib
import fsspec
import json
import os
import io
import re



BASE_DATE = datetime(2000, 1, 1)


def _decode_timestamp(item_id: str) -> datetime:
    """Decode the trailing base-36 suffix of an item id into a datetime."""
    suffix = item_id.split('_')[-1]
    seconds = int(suffix, 36)
    return BASE_DATE + timedelta(seconds=seconds)


def _group_key(item_id: str) -> str:
    """Everything except the trailing timestamp suffix (used to identify duplicates)."""
    return item_id.rsplit('_', 1)[0]


def dedupe_most_recent(items):
    """
    Given a list of Item objects (each with an `.id` attribute),
    return a list containing only the most recent item per unique product
    (i.e. duplicates that differ only in the trailing timestamp suffix
    are collapsed to the latest one).
    """
    latest_by_group = {}

    for item in items:
        key = _group_key(item.id)
        ts = _decode_timestamp(item.id)

        if key not in latest_by_group or ts > latest_by_group[key][1]:
            latest_by_group[key] = (item, ts)

    return [item for item, _ in latest_by_group.values()]



def filter_matching_products(items, reference_datetime):
    """
    Given a list of Item objects (each with an `.id` attribute) and a
    reference datetime interval (e.g. ['2026-04-23T09:57:31.000Z',
    '2026-04-23T09:59:28.000Z']), return one item per cycle (C01, C02, ...)
    such that:

      1. the item's time-of-day interval contains the reference product's
         time-of-day interval (the date itself is ignored, since each
         cycle repeats the same ground track at a similar local time but
         on a different date), and
      2. if multiple items in the same cycle satisfy (1), only the one
         with the largest datetime interval (longest duration) is kept.
    """
    ref_start, ref_stop = _parse_reference_datetime(reference_datetime)
    ref_start_t, ref_stop_t = ref_start.time(), ref_stop.time()

    best_by_cycle = {}

    for item in items:
        start, stop, cycle = _parse_item(item.id)
        start_t, stop_t = start.time(), stop.time()

        # Condition 1: time-of-day containment
        if not (start_t <= ref_start_t and stop_t >= ref_stop_t):
            continue

        # Condition 2: keep the longest-duration item per cycle
        duration = stop - start
        if cycle not in best_by_cycle or duration > best_by_cycle[cycle][1]:
            best_by_cycle[cycle] = (item, duration)

    # Sort output by cycle number for readability (C01, C02, ...)
    return [item for _, (item, _) in sorted(best_by_cycle.items())]

def _parse_reference_datetime(reference_datetime):
    """
    Parse a [start, stop] datetime pair (ISO 8601 strings, e.g.
    '2026-04-23T09:57:31.000Z') into datetime objects.
    """
    start_str, stop_str = reference_datetime

    def _to_dt(s):
        return datetime.strptime(s.replace("Z", ""), "%Y-%m-%dT%H:%M:%S.%f")

    return _to_dt(start_str), _to_dt(stop_str)


def product_search(catalog, collections, prod_type, datetime, baseline):
    '''
    Searches a STAC catalog product matching a given product type, collection and datetime.

    Args:
        catalog (Client): pystac_client Client instance connected to the catalog.
        collections (list[str]): list of collection names to search in.
        prod_type (str): product type filter string (e.g. 'AUX_ATT___').
        datetime (list[str]): a [start_datetime, end_datetime] pair in ISO 8601 format.

    Returns:
        list[Item]: a list containing at most one matching STAC Item.
    '''

    search = catalog.search(
    collections = collections,
    filter = f"product:type='{prod_type}' and\
                version='{baseline}'",
    datetime = datetime,
    method = "GET",
    #max_items = 1,  
    )

    # Gives a list of items considering the most recent product in case there are two identical ids, based on the last 6 digits. For example:
    # BIO_S1_RAW__0M_20251121T094510_20251121T095758_T_G01_M01_C01_T006_F____01_DNGC54
    # BIO_S1_RAW__0M_20251121T094510_20251121T095758_T_G01_M01_C01_T006_F____01_DNPTVM
    # It gives only BIO_S1_RAW__0M_20251121T094510_20251121T095758_T_G01_M01_C01_T006_F____01_DNPTVM
    
    items = dedupe_most_recent(list(search.items()))

    return items


def product_search_majorcycle(catalog, collections, prod_type, global_id, major_cycle, track, frame, baseline):
    """
    Searches a STAC catalog product matching a given product type within a specific major cycle, track, and frame.

    Args:
        catalog (Client): a pystac_client Client instance connected to the catalog.
        collections (list[str]): list of collection names to search in.
        prod_type (str): product type filter string (e.g. 'AUX_ATT___').
        global_id (int): global coverage ID to filter by (eofeos:global_coverage_id).
        major_cycle (int): major cycle ID to filter by (eofeos:major_cycle_id).
        track (int): track number to filter by.
        frame (int): frame number to filter by.
        baseline (str): baseline number to filter by.

    Returns:
        list[Item]: a list of matching STAC Items, up to 10.
    """

    search = catalog.search(
    collections = collections,
    filter = f"product:type='{prod_type}' and\
            eofeos:global_coverage_id={global_id} and\
            eofeos:major_cycle_id={major_cycle} and\
            track={track} and\
            frame={frame} and\
            version='{baseline}'",
        
    method = "GET",
    #max_items = 10,  
    )

    items = dedupe_most_recent(list(search.items()))
    
    return items




def download_product(urls, output_path, disable_bar=False, subdir="Inputs"):
    """
    Downloads and unzips a list of products from the given URLs into the specified output path. 
    After the product is unzipped, the .zip file is deleted to save memory.

    The function authenticates using the MAAP_CATALOGUE_ACCESS_TOKEN environment variable.
    Each product is downloaded as a .zip file and then extracted into a subdirectory
    called 'Inputs' within output_path. Both directories are created if they do not exist.

    Args:
        urls (list[str]): list of URLs pointing to the products to download.
        output_path (str): local directory path where downloaded .zip files will be saved and extracted.
        disable_bar (bool): optional, if True, suppresses the tqdm progress bar and prints a completion message instead. Default is False.

    Raises:
        OSError: if the output directories cannot be created or if the deletion of the .zip file cannot be performed.
        requests.exceptions.RequestException: if the HTTP request fails (e.g. authentication error, connection timeout).
        Exception: if the unzipping step fails.
    """

    try:

        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    except OSError as e:
        print(f"Error creating output directories: {e}")

    
    for url in urls:

        try:

            filename = url.rsplit('/', 1)[-1] + '.zip'
            output_filename = os.path.join(output_path, filename)
            unzip_path = os.path.join(output_path, subdir)
            token = os.environ["MAAP_CATALOGUE_ACCESS_TOKEN"]
    
    
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
    
            file_size = int(response.headers.get('content-length', 0))
    
            chunk_size = 8 * 1024 * 1024 
            with open(output_filename, "wb") as f, tqdm(
                desc = url.rsplit('/', 1)[-1] + '.zip',
                total = file_size,
                unit = 'iB',
                unit_scale = True,
                unit_divisor = 1024,
                disable = disable_bar,
              ) as bar:
              for chunk in response.iter_content(chunk_size = chunk_size):
                read_size = f.write(chunk)
                bar.update(read_size)
    
            if (disable_bar): 
              print(f"File downloaded successfully to {output_filename}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")

        
        try:
            print(f"Unzipping {url.rsplit('/', 1)[-1]} into {unzip_path}")
            import subprocess
            subprocess.run(["unzip", "-q", "-o", output_filename, "-d", unzip_path], check=True)

        except Exception as e:
            print(f"Error unzipping file: {e}")

        
        try:
            print(f'Removing {url.rsplit('/', 1)[-1] + '.zip'} to save memory')
            os.remove(output_filename)
            print(f"Deleted {output_filename}")
            
        except OSError as e:
            print(f"Error deleting file: {e}")

            

def extract_xml_fields(url, keys=None):
    """
    Fetches an OWC/GeoJSON metadata document from a URL and extracts field values
    by recursively searching all nested dicts and lists under 'properties'.

    Args:
        url (str): URL of the OWC JSON document to fetch and parse.
        keys (list[str]): List of field names to extract. Each key is searched
                          recursively across the entire properties tree.
                          First match wins (depth-first).

    Returns:
        dict: Mapping each requested key to its extracted value (None if not found).
    """
    
    keys    = keys or []
    results = {k: None for k in keys}

    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))

    props = data.get("properties", {})

    def _find(obj, key):
        """Recursively search dicts and lists for a given key."""
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                found = _find(v, key)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _find(item, key)
                if found is not None:
                    return found
        return None

    for key in keys:
        results[key] = _find(props, key)

    return results



def get_raw_repeat_cycle(ID, output_path):
    """
    Downloads all raw input products needed for a single repeat cycle.

    Given a L0S product ID, this function retrieves and downloads the corresponding
    0S and 0M products along with the required auxiliary files (AUX_ATT, AUX_ORB, AUX_TEC)
    covering the same datetime interval. All products are downloaded and extracted into output_path.

    Args:
        ID (str): STAC item ID of the reference L0S product (e.g. 'BIO_S3_RAW__0S_...').
        output_path (str): local directory path where all products will be downloaded and extracted.

    Raises:
        IndexError: if no item matching the given ID is found in the catalog.
        requests.exceptions.RequestException: if any product download fails.
    """

    catalog_url = 'https://catalog.maap.eo.esa.int/catalogue/'
    catalog = Client.open(catalog_url)

    search = catalog.search(
    collections = ["BiomassLevel0","BiomassLevel0IOC"],
    ids=ID,
    method = "GET",
    #max_items = 1,  
    )

    items = list(search.items())
    L0_S = items

    start_date = L0_S[0].properties.get("start_datetime")
    end_date = L0_S[0].properties.get("end_datetime")
    prod_type_0S = L0_S[0].properties.get("product:type")
    prod_type_0M = prod_type_0S[:-1] + "M"
    metadata_url = L0_S[0].assets['metadata_ogc_17_003r2'].href

    datetime = [start_date, end_date]

    # Get the baseline from the product metadata file
    values = extract_xml_fields(
    url = metadata_url,
    keys = [
        "productVersion",
    ]
    )

    baseline = values['productVersion']

    
    # Get the 0M product
    collections = ["BiomassLevel0","BiomassLevel0IOC"]
    L0_M = product_search(catalog, collections, prod_type_0M, datetime, baseline)

    # Get the AUX_ATT
    collections = ["BiomassAuxIOC","BiomassAux","BiomassAuxRest"]
    BIO_AUX_ATT = product_search(catalog, collections, 'AUX_ATT___', datetime, baseline)

    # Get the AUX_ORB
    BIO_AUX_ORB = product_search(catalog, collections, 'AUX_ORB___', datetime, baseline)

    # Get the AUX_TEC
    BIO_AUX_TEC = product_search(catalog, collections, 'AUX_TEC___', datetime, baseline)


    total_prod = L0_S + L0_M + BIO_AUX_ATT + BIO_AUX_ORB + BIO_AUX_TEC

    
    # Print a table resuming the matching values
    labels={
    "productVersion":   "Baseline",
    }
    lines  = [f"{labels.get(k, k)}: {v}" for k, v in values.items()]
    max_len = max(len(line) for line in lines)
    width   = max_len + 4

    top    = "┌" + "─" * width + "┐"
    bottom = "└" + "─" * width + "┘"
    sep    = "├" + "─" * width + "┤"

    print(f'Matching parameters for {ID}')

    print(top)
    for i, line in enumerate(lines):
        print(f"│  {line:<{max_len}}  │")
        if i < len(lines) - 1:
            print(sep)
    print(bottom)

    print('\n')
    print(f'Products to be downloaded in {output_path} ({len(total_prod)}):')

    for item in total_prod:
        print(f"- {item.id}")
    print('\n')


    # Get the urls for the download
    L0_S_url = L0_S[0].assets['product'].href
    L0_M_url = L0_M[0].assets['product'].href
    AUX_ATT_url = BIO_AUX_ATT[0].assets['product'].href
    AUX_ORB_url = BIO_AUX_ORB[0].assets['product'].href
    AUX_TEC_url = BIO_AUX_TEC[0].assets['product'].href

    urls = [L0_S_url, L0_M_url, AUX_ATT_url, AUX_ORB_url, AUX_TEC_url]

    # Download and unzip products in the specified path
    download_product(urls, output_path, disable_bar=False)

    return L0_S, L0_M, BIO_AUX_ATT, BIO_AUX_ORB, BIO_AUX_TEC



_PATTERN = re.compile(
    r'_(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_T_G\d{2}_M\d{2}_(?P<cycle>C\d{2})_'
)

_DT_FMT = "%Y%m%dT%H%M%S"


def _parse_item(item_id: str):
    """Extract (start, stop, cycle) from an item id."""
    match = _PATTERN.search(item_id)
    if not match:
        raise ValueError(f"Could not parse item id: {item_id}")

    start = datetime.strptime(match.group("start"), _DT_FMT)
    stop = datetime.strptime(match.group("stop"), _DT_FMT)
    cycle = match.group("cycle")
    return start, stop, cycle


def check_major_cycle_0M(items):
    """
    Given a list of Item objects, return a
    reduced list containing only one item per cycle (C01, C02, ...): the
    one with the largest time interval (start-to-stop duration).
    """
    best_by_cycle = {}

    for item in items:
        start, stop, cycle = _parse_item(item.id)
        duration = stop - start

        if cycle not in best_by_cycle or duration > best_by_cycle[cycle][1]:
            best_by_cycle[cycle] = (item, duration)

    # Sort output by cycle number for readability (C01, C02, ...)
    return [item for _, (item, _) in sorted(best_by_cycle.items())]



def get_raws_major_cycle(ID, output_path):
    """
    Downloads all raw input products needed for a full major cycle.

    Given a L0S product ID, this function extracts the global coverage ID, major cycle ID,
    track, and frame from its metadata, then retrieves all 0S products sharing the same
    major cycle, track, and frame. For each 0S product, the corresponding auxiliary files
    (AUX_ATT, AUX_ORB, AUX_TEC) are fetched. All matching 0M products for the same major cycle
    and track are also retrieved. All products are downloaded and extracted into output_path.

    Args:
        ID (str): STAC item ID of the reference L0S product (e.g. 'BIO_S3_RAW__0S_...').
        output_path (str): local directory path where all products will be downloaded and extracted.

    Raises:
        IndexError: if no item matching the given ID is found in the catalog.
        KeyError: if required metadata fields are missing from the product XML.
        requests.exceptions.RequestException: if any product download fails.
    """

    catalog_url = 'https://catalog.maap.eo.esa.int/catalogue/'
    catalog = Client.open(catalog_url)

    search = catalog.search(
    collections = ["BiomassLevel0","BiomassLevel0IOC"],
    ids=ID,
    method = "GET",
    #max_items = 1,  
    )

    items = list(search.items())
    L0_S = items
    prod_type_0S = L0_S[0].properties.get("product:type")
    prod_type_0M = prod_type_0S[:-1] + "M"

    
    # Get the remaining 0S products
    collections = ["BiomassLevel0","BiomassLevel0IOC"]
    metadata_url = L0_S[0].assets['metadata_ogc_17_003r2'].href

    values = extract_xml_fields(
    url = metadata_url,
    keys = [
        "wrsLongitudeGrid",
        "wrsLatitudeGrid",
        "globalCoverageID",
        "repeatCycleID",
        "majorCycleID",
        "productVersion",
    ]
    )

    global_id = int(values['globalCoverageID'])
    major_cycle = int(values['majorCycleID'])
    track = int(values['wrsLongitudeGrid'])
    frame = int(values['wrsLatitudeGrid'])
    baseline = values['productVersion']
    
    L0S_items = product_search_majorcycle(catalog, collections, prod_type_0S, global_id, major_cycle, track, frame, baseline)

    L0M_items = []
    for item in L0S_items:
        start_date = item.properties.get("start_datetime")
        end_date = item.properties.get("end_datetime")
    
        datetime = [start_date, end_date]
        L0M_items.extend(product_search(catalog, collections, prod_type_0M, datetime, baseline))

    # Check if there are multiple products with same repeat cycle and take the one with the largest time interval
    L0M_items = check_major_cycle_0M(L0M_items)

    # Get the AUX products
    collections = ["BiomassAuxIOC","BiomassAux","BiomassAuxRest"]

    ATT_items = []
    ORB_items = []
    TEC_items = []
    for item in L0S_items:
        start_date = item.properties.get("start_datetime")
        end_date = item.properties.get("end_datetime")
    
        datetime = [start_date, end_date]
    
        ATT_items.extend(product_search(catalog, collections, 'AUX_ATT___', datetime, baseline))
        ORB_items.extend(product_search(catalog, collections, 'AUX_ORB___', datetime, baseline))
        TEC_items.extend(product_search(catalog, collections, 'AUX_TEC___', datetime, baseline))


    total_prod = L0S_items + L0M_items + ATT_items + ORB_items + TEC_items
    
    # Print a table resuming the matching values
    labels={
    "globalCoverageID": "Global Coverage",
    "majorCycleID":     "Major Cycle",
    "repeatCycleID":    "Repeat Cycle",
    "wrsLongitudeGrid": "Track",
    "wrsLatitudeGrid":  "Frame", 
    "productVersion":   "Baseline",
    }

    lines  = [f"{labels.get(k, k)}: {v}" for k, v in values.items()]
    max_len = max(len(line) for line in lines)
    width   = max_len + 4

    top    = "┌" + "─" * width + "┐"
    bottom = "└" + "─" * width + "┘"
    sep    = "├" + "─" * width + "┤"

    print(f'Matching parameters for {ID}')

    print(top)
    for i, line in enumerate(lines):
        print(f"│  {line:<{max_len}}  │")
        if i < len(lines) - 1:
            print(sep)
    print(bottom)

    print('\n')
    print(f'Products to be downloaded in {output_path} ({len(total_prod)}):')

    for item in total_prod:
        print(f"- {item.id}")
    print('\n')
    
    
    # Getting the urls of the products
    L0S_urls = [url.assets["product"].href for url in L0S_items]
    L0M_urls = [url.assets["product"].href for url in L0M_items]
    ATT_urls = [url.assets["product"].href for url in ATT_items]
    ORB_urls = [url.assets["product"].href for url in ORB_items]
    TEC_urls = [url.assets["product"].href for url in TEC_items]

    # Download and extract the products
    download_product(L0S_urls, output_path, disable_bar=False)
    download_product(L0M_urls, output_path, disable_bar=False)
    download_product(ATT_urls, output_path, disable_bar=False)
    download_product(ORB_urls, output_path, disable_bar=False)
    download_product(TEC_urls, output_path, disable_bar=False)


    
def get_NetCDF(input_path, output_path, start, stop):

    '''
    Generates a NetCDF file starting from RAW_0S, RAW_0M, AUX_ATT, AUX_orb and AUX_INS files.

    Args:
        input_path (Path): path to the input products (RAW_0S, RAW_0M, AUX_ATT, AUX_orb, AUX_INS).
        output_path (Path): path to the output NetCDF file.
        start (str): start time of interest, e.g. '09-APR-2026 03:59:50.000000'.
        stop (str): stop time of interest, e.g. '09-APR-2026 03:59:50.000000'.

    Returns:
        NetCDF file [.nc file]: single NetCDF file with the same name of input L0S product.
    '''

    from bps.l0_extraction_tool.main import run

    inputs_dir = Path(os.path.join(input_path, 'Inputs'))
    output_path = Path(output_path)
    # Map each variable to its identifying pattern
    patterns = {
        "l0s":     "RAW__0S",
        "l0m":     "RAW__0M",
        "aux_att": "AUX_ATT",
        "aux_orb": "AUX_ORB",
        "aux_ins": "AUX_INS",
    }
    
    # Search subfolders and match patterns
    found = {}
    for var_name, pattern in patterns.items():
        matches = [p for p in inputs_dir.iterdir() if p.is_dir() and pattern in p.name]
        if len(matches) == 1:
            found[var_name] = matches[0]
        elif len(matches) == 0:
            print(f"WARNING: No folder found for '{var_name}' (pattern: '{pattern}')")
            found[var_name] = None
        else:
            print(f"WARNING: Multiple matches for '{var_name}' (pattern: '{pattern}'): {matches}")
            found[var_name] = None
    
    # Unpack into individual variables
    l0s_product_path     = found["l0s"]
    l0m_product_path     = found["l0m"]
    aux_att_product_path = found["aux_att"]
    aux_orb_product_path = found["aux_orb"]
    aux_ins_product_path = found["aux_ins"]
    
    # Print results
    for name, path in found.items():
        print(f"{name}: {path}")


    # Run the l0_extraction_tool
    run(
    l0s_product_path,
    l0m_product_path,
    aux_orb_product_path,
    aux_att_product_path,
    aux_ins_product_path,
    start,
    stop,
    output_path,
    )

# ============================================================
# SCS / L1A download — added for L1A→L2A processing
# ============================================================

_SCS_ID_PATTERN = re.compile(
    r'_T_G(?P<global>\d{2})_M(?P<major>\d{2})_C(?P<cycle>\d{2})_T(?P<track>\d{3})_F(?P<frame>\d{3})_(?P<baseline>\d{2})_'
)


def _parse_scs_id(item_id):
    """Extract global coverage / major cycle / track / frame / baseline from an SCS product ID, e.g.
    BIO_S1_SCS__1S_20251122T005346_20251122T005407_T_G01_M01_C01_T016_F050_01_DKAYPO"""
    m = _SCS_ID_PATTERN.search(item_id)
    if not m:
        raise ValueError(f"Could not parse SCS ID: {item_id}")
    return m.groupdict()


def product_search_gmtf(catalog, collections, prod_type, global_id, major_cycle, track, frame, baseline):
    """
    Searches a STAC catalog for all SCS items sharing the same global coverage (G), major cycle (M),
    track (T), frame (F) and baseline as the reference product (i.e. every repeat cycle of the stack).
    """
    search = catalog.search(
        collections=collections,
        filter=f"product:type='{prod_type}' and \
                eofeos:global_coverage_id={global_id} and \
                eofeos:major_cycle_id={major_cycle} and \
                track={track} and\
                frame={frame} and\
                version='{baseline}'",
        method="GET",
    )
    items = dedupe_most_recent(list(search.items()))
    return items


def get_scs(ID, output_path):
    """
    Downloads all SCS/L1A products sharing the same G/M/T/F/baseline as the given reference ID.
    The stack itself is NOT downloaded here — it is generated later by the STA chain.

    Args:
        ID (str): STAC item ID of one reference SCS/L1A product (e.g. 'BIO_S1_SCS__1S_...').
        output_path (str): local directory path where all products will be downloaded and extracted.

    Returns:
        list[Item]: the STAC items that were downloaded.
    """

    catalog_url = 'https://catalog.maap.eo.esa.int/catalogue/'
    catalog = Client.open(catalog_url)
    collections = ["BiomassLevel1a", "BiomassLevel1aIOC"]   

    search = catalog.search(collections=collections, ids=ID, method="GET")
    items = list(search.items())
    if not items:
        raise IndexError(f"No item found for ID {ID} in {collections}")

    ref_item  = items[0]
    prod_type = ref_item.properties.get("product:type")
    parsed    = _parse_scs_id(ID)

    scs_items = product_search_gmtf(
        catalog, collections, prod_type,
        parsed["global"], parsed["major"], parsed["track"], parsed["frame"], parsed["baseline"],
    )

    print(f"Matching parameters for {ID}: G{parsed['global']} M{parsed['major']} "
          f"T{parsed['track']} F{parsed['frame']} baseline={parsed['baseline']}")
    print(f"Products to be downloaded in {output_path} ({len(scs_items)}):")
    for item in scs_items:
        print(f"- {item.id}")

    urls = [it.assets["product"].href for it in scs_items]
    download_product(urls, output_path, disable_bar=False, subdir="L1A") 

    return scs_items

    
def get_stack(ID, output_path):
    """
    Downloads all STA/L1C products sharing the same G/M/T/F/baseline as the given reference ID.
    The stack itself is NOT downloaded here — it is generated later by the STA chain.

    Args:
        ID (str): STAC item ID of one reference STA/L1C product (e.g. 'BIO_S1_STA__1S_...').
        output_path (str): local directory path where all products will be downloaded and extracted.

    Returns:
        list[Item]: the STAC items that were downloaded.
    """

    catalog_url = 'https://catalog.maap.eo.esa.int/catalogue/'
    catalog = Client.open(catalog_url)
    collections = ["BiomassLevel1c", "BiomassLevel1cIOC"]   

    search = catalog.search(collections=collections, ids=ID, method="GET")
    items = list(search.items())
    if not items:
        raise IndexError(f"No item found for ID {ID} in {collections}")

    ref_item  = items[0]
    prod_type = ref_item.properties.get("product:type")
    parsed    = _parse_scs_id(ID)

    sta_items = product_search_gmtf(
        catalog, collections, prod_type,
        parsed["global"], parsed["major"], parsed["track"], parsed["frame"], parsed["baseline"],
    )

    print(f"Matching parameters for {ID}: G{parsed['global']} M{parsed['major']} "
          f"T{parsed['track']} F{parsed['frame']} baseline={parsed['baseline']}")
    print(f"Products to be downloaded in {output_path} ({len(sta_items)}):")
    for item in sta_items:
        print(f"- {item.id}")

    urls = [it.assets["product"].href for it in sta_items]
    download_product(urls, output_path, disable_bar=False, subdir="L1C") 

    return sta_items






