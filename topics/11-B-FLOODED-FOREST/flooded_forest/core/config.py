"""config.yaml loader. Relative paths resolve against the topic root."""

from pathlib import Path

import yaml

TOPIC_ROOT = Path(__file__).resolve().parent.parent


class Config:
    """User settings from config.yaml."""

    def __init__(self, path=None):
        self.path = Path(path) if path else TOPIC_ROOT / "config.yaml"
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}

        # paths (relative -> topic root)
        self.credentials_path = self._resolve(raw.get("credentials_path", "credentials.txt"))
        self.roi_json_path = self._resolve(raw.get("roi_json_path", "rois.json"))
        self.assets_dir = self._resolve(raw.get("assets_dir", "assets"))

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

        # stack extraction defaults (passed to marcos_dev/1_data_extraction/extract_l1c_roi.py)
        self.max_files = raw.get("max_files")
        self.min_coverage = raw.get("min_coverage")
        self.workers = int(raw.get("workers", 6))

        # MBMP coherence (passed to marcos_dev/1_data_extraction/compute_coherence.py)
        self.coherence_window = float(raw.get("coherence_window", 250))

    def _resolve(self, value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else (TOPIC_ROOT / path).resolve()

    @property
    def catalogue_dir(self):
        return self.assets_dir / "catalogue"

    @property
    def stacks_dir(self):
        return self.assets_dir / "stacks"

    def roi_dir(self, slug):
        """Folder of one ROI under assets/stacks (figures, quicklooks, stacks)."""
        return self.stacks_dir / slug

    def ensure_dirs(self):
        for folder in (self.catalogue_dir, self.stacks_dir):
            folder.mkdir(parents=True, exist_ok=True)


def load_config(path=None):
    return Config(path)
