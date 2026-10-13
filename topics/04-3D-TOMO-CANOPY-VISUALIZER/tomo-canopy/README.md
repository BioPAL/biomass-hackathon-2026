# tomo-canopy

Multi-scale LiDAR and BIOMASS tomogram processing and visualisation tools.

## Package layout

- `providers`: readers and adapters for TLS, ALS, GEDI, and tomogram data
- `processors`: processing pipelines and data preparation
- `cubegen`: generation and assembly of three-dimensional data cubes
- `methods`: scientific algorithms and analysis methods
- `transforms`: spatial, coordinate, and value transformations
- `visualisation`: interactive and static visualisation backends

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

