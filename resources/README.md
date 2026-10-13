# Shared resources

Code and helpers useful to **several** project groups live here. Group-specific notebooks and scripts stay under `topics/<GROUP-LABEL>/` (same name as the GitHub group label).

Do not commit large datasets or credentials. Prefer MAAP links or small download scripts. See [CONTRIBUTING.md](../CONTRIBUTING.md).

## Layout

Shared code is organised by layer (as in the hackathon kick-off):

```
resources/
├── README.md
├── load/          # read products → arrays / xarray (no plotting)
├── process/       # arrays in → arrays out (no plotting)
├── evaluate/      # metrics, validation, stats
├── plot/          # figures, maps
└── tests/         # pytest for shared code (add when ready)
```

| Layer | Role | Plotting? |
| --- | --- | --- |
| `load/` | Read BIOMASS / MAAP / other products into NumPy or xarray | No |
| `process/` | Transform arrays (filters, indices, resampling, …) | No |
| `evaluate/` | Metrics, validation against references, summary stats | Prefer returning numbers; leave figures to `plot/` |
| `plot/` | Figures, maps, RGB composites, and other visuals | Yes |

Keep I/O and computation free of `matplotlib` / plotting side effects so notebooks and batch jobs can reuse the same functions.

### Migration note

Before this layout, `resources/` only held this README. There was no existing Python package to move. New shared modules should go into the matching layer folder above rather than the `resources/` root.

## Example import from a topic notebook

From a notebook under `topics/...`, add the repo root to `sys.path` (or install the shared code locally if you prefer), then import by layer:

```python
import sys
from pathlib import Path

# notebook lives at topics/<topic>/... — go up to the repo root
ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT / "resources").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from resources.load import ...      # noqa: E402 — once helpers exist
from resources.process import ...   # noqa: E402
from resources.evaluate import ...  # noqa: E402
from resources.plot import ...      # noqa: E402
```

Replace the `...` imports with real function names as you add them. Run notebooks with the working directory set to the repo root when possible so relative imports stay simple.

## Dependencies

There is no root `requirements.txt` yet. Use the MAAP Python environment for the hackathon, and document any extra packages in your topic README or in a short note next to the shared module you add. Prefer Apache-2.0-compatible dependencies.

## Contributing shared code

1. Prototype in your `topics/...` folder.
2. If a second team needs the same helper, open a PR that moves or copies a cleaned version into the right `resources/` layer.
3. Add a test under `resources/tests/` when the function is non-trivial.
4. Follow the light conventions in [CONTRIBUTING.md](../CONTRIBUTING.md) (docstrings, type hints on exported APIs, no committed data).
