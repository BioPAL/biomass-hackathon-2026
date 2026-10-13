import requests
import json
import pprint

url = "https://dahiti.dgfi.tum.de/api/v2/download-water-level/"

args = {}
args["api_key"] = "589AD7A19BAE1CBA8FCE33756EBD90BD124ECB2A2F4E638BFADD12C6C4DE84C4"
args["dahiti_id"] = 15307  # Rio Solimoes
args["format"] = "json"

response = requests.get(url, params=args)

if response.status_code == 200:
    if args["format"] == "ascii":
        data_ascii = response.text
        print(data_ascii)
    if args["format"] == "json":
        data_ascii = json.loads(response.text)
        pprint.pprint(data_ascii)
    if args["format"] == "netcdf":
        path_netcdf = "/tmp/" + str(args["dahiti_id"]) + "_water_level_altimetry.nc"
        print("writing ... " + path_netcdf)
        with open(path_netcdf, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("done!")
    if args["format"] == "csv":
        data_csv = response.text
        print(data_csv)
else:
    print(response.text)
    print(response.status_code)
