# Guide BIOMASS hackathon

## New environment: installation of packages
For windows with python 3.13.13

```
python --version
py -3.13 -v venv .venv_biomass
.\.venv_biomass\Scripts\Activate.ps1
```

Install Jupyter and the packages used by your notebook
```
python -m pip install --upgrade pip
python -m pip install jupyter ipykernel aiohttp requests shapely rasterio matplotlib pystac-client folium geopandas fiona tqdm pillow pandas numpy fsspec scipy

```

Register the environment as a Jupyter kernel:
```
python -m ipykernel install --user --name geo313 --display-name "Python 3.13 (geo313)"
```
Then, in VS Code:
1. Open your notebook and click Select Kernel in the top-right.
2. Choose Select Another Kernel → Python Environments.
3. Select the interpreter in your project’s .venv folder