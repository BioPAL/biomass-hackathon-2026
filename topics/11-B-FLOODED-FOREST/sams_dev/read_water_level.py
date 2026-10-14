#!/usr/init/python3
# -*- coding: utf-8 -*-
''' Version: 2.1 '''
''' Description: Find nearest DAHITI target to a ROI centroid and download water level time series '''

import os
import sys
import json
import pprint
import requests
from shapely import wkt

''' Download path were all water level time series will be stored '''
download_path = 'downloads/'
if not os.path.isdir(download_path):
    os.mkdir(download_path)

API_KEY = "589AD7A19BAE1CBA8FCE33756EBD90BD124ECB2A2F4E638BFADD12C6C4DE84C4"
OUTPUT_FORMAT = 'ascii'
BASE_URL = "https://dahiti.dgfi.tum.de/api/v2/"

# 1. Setup ROI and find Centroid
roi_wkt = "POLYGON((-62.0 -4.0, -59.0 -4.0, -59.0 -2.0, -62.0 -2.0, -62.0 -4.0))"
polygon = wkt.loads(roi_wkt)
centroid = polygon.centroid

# 2. List targets in the region
url = BASE_URL + 'list-targets/'
args = {
    'api_key': API_KEY,
    'basin': 'Amazon'
}

response = requests.post(url, data=args)
if response.status_code == 200:
    targets = json.loads(response.text)['data']
    print('Dataset(s) found:', len(targets))

    if not targets:
        print('No targets found in the specified basin.')
        sys.exit(0)

    # 3. Find the nearest target based on Euclidean distance to the centroid (lat/lon)
    nearest_target = min(
        targets,
        key=lambda t: (t['latitude'] - centroid.y)**2 + (t['longitude'] - centroid.x)**2
    )
    
    print('Nearest Target Selected:')
    pprint.pprint(nearest_target)

    # 4. Download water level time series for the nearest target
    url = BASE_URL + 'download-water-level/'
    args = {
        'api_key': API_KEY,
        'dahiti_id': nearest_target['dahiti_id'],
        'format': OUTPUT_FORMAT
    }

    ext_map = {'ascii': 'txt', 'json': 'json', 'netcdf': 'nc', 'csv': 'csv'}
    file_ext = ext_map.get(OUTPUT_FORMAT, 'txt')
    path_output = os.path.abspath(f"{download_path}/{nearest_target['dahiti_id']}.{file_ext}")

    print(f"Downloading ... {nearest_target['dahiti_id']} -> {nearest_target['target_name']} ({path_output})")
    
    response_download = requests.post(url, json=args)
    if response_download.status_code == 200:
        mode = 'wb' if OUTPUT_FORMAT == 'netcdf' else 'w'
        encoding = None if OUTPUT_FORMAT == 'netcdf' else 'utf-8'
        
        with open(path_output, mode, encoding=encoding) as f:
            if OUTPUT_FORMAT == 'netcdf':
                for chunk in response_download.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            else:
                f.write(response_download.text)
        print('Download completed successfully.')
    else:
        print('Error: `download-water-level` request failed!')
        pprint.pprint(json.loads(response_download.text))
        sys.exit(0)
else:
    print('Error: `list-targets` request failed!')
    pprint.pprint(json.loads(response.text))