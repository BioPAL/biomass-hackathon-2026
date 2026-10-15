"""config.yaml loader and output layout. Relative paths resolve against the topic root."""

import json
from pathlib import Path

import yaml

TOPIC_ROOT = Path(__file__).resolve().parents[2]


class RoiPaths:
    """Output folders of one ROI: outputs/<roi>/{extraction,quicklooks,coherence,polsar}/."""

    def __init__(self, root):
        self.root = Path(root)
        self.extraction = self.root / "extraction"    # common area, summaries, one folder per stack
        self.quicklooks = self.root / "quicklooks"    # catalogue/ and one folder per stack
        self.coherence = self.root / "coherence"      # one folder per stack
        self.polsar = self.root / "polsar"            # one folder per stack
        self.water_level = self.root / "water_level"  # DAHITI series + info of the nearest target
        self.figures = self.extraction / "figures"    # ROI-level figures (time series)

    def stack_dirs(self):
        """Stack folders of the last extraction (same common area); every stack folder if no summary."""
        summary = self.extraction / "extraction_summary.json"
        if summary.exists():
            names = [st["folder"] for st in json.loads(summary.read_text(encoding="utf-8"))["stacks"]]
            return [self.extraction / n for n in names if (self.extraction / n / "stack_info.json").exists()]
        return sorted(p.parent for p in self.extraction.glob("*/stack_info.json"))


class Config:
    """User settings from config.yaml."""

    def __init__(self, path=None):
        self.path = Path(path) if path else TOPIC_ROOT / "config.yaml"
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}

        # paths (relative -> topic root)
        self.credentials_path = self._resolve(raw.get("credentials_path", "credentials.txt"))
        self.roi_json_path = self._resolve(raw.get("roi_json_path", "rois.json"))
        self.outputs_dir = self._resolve(raw.get("outputs_dir", "outputs"))

        # MAAP catalogue
        self.stac_api_url = raw.get("stac_api_url", "https://catalog.maap.eo.esa.int/catalogue/")
        self.token_url = raw.get(
            "token_url",
            "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token",
        )
        self.collection = raw.get("collection", "BiomassLevel1c")

        # STAC search
        self.min_overlap_fraction = float(raw.get("min_overlap_fraction", 0.0))
        self.max_items_per_roi = int(raw.get("max_items_per_roi", 1000))
        self.request_pause_seconds = float(raw.get("request_pause_seconds", 0.1))

        # stack extraction
        self.max_files = int(raw.get("max_files", 70))
        self.min_coverage = float(raw.get("min_coverage", 0.5))
        self.workers = int(raw.get("workers", 6))
        self.apply_skp = bool(raw.get("apply_skp", True))
        self.apply_flat = bool(raw.get("apply_flat", True))

        # spatial averaging shared by coherence and PolSAR
        self.averaging_window = float(raw.get("averaging_window", raw.get("coherence_window", 250)))

        # DAHITI water level (pipeline step 10, tools/water_level_roi.py)
        # str() also handles the date object that YAML makes of an unquoted 2024-01-01
        self.water_level_since = str(raw.get("water_level_since", "2024-01-01"))

        # PolSAR
        self.swath_incidence_deg = {
            str(k).upper(): float(v)
            for k, v in (raw.get("swath_incidence_deg") or {"S1": 25.4, "S2": 29.1, "S3": 32.2}).items()
        }

    def _resolve(self, value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else (TOPIC_ROOT / path).resolve()

    @property
    def catalogue_dir(self):
        return self.outputs_dir / "catalogue"

    def roi_paths(self, slug):
        return RoiPaths(self.outputs_dir / slug)

    def incidence_deg(self, swath):
        """Mean incidence angle of a swath ('S1', 'S2', 'S3')."""
        try:
            return self.swath_incidence_deg[swath.upper()]
        except KeyError:
            raise KeyError(f"no incidence angle for swath {swath} in config.yaml "
                           f"(swath_incidence_deg has {sorted(self.swath_incidence_deg)})") from None

    def ensure_dirs(self):
        self.catalogue_dir.mkdir(parents=True, exist_ok=True)


def load_config(path=None):
    return Config(path)
