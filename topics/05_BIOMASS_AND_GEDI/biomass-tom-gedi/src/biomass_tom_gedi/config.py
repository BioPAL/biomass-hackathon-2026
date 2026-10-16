"""YAML configuration with `~` / `${ENV}` expansion and a few derived directories."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


class Config:
    """Thin wrapper around the parsed YAML. Use `cfg["section"]["key"]` or the helpers below."""

    def __init__(self, data: dict[str, Any], source: Path | None = None):
        self.data = _expand(data)
        self.source = source

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Config":
        path = Path(path)
        with open(path) as fh:
            return cls(yaml.safe_load(fh) or {}, source=path)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.data
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    # ---- directories -------------------------------------------------------
    @property
    def interim_dir(self) -> Path:
        d = Path(self.get("paths", "data_dir", default="data")) / "interim"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def results_dir(self) -> Path:
        d = Path(self.get("paths", "results_dir", default="results"))
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- the files each step reads / writes ---------------------------------
    @property
    def shots_path(self) -> Path:
        return self.interim_dir / "shots.parquet"          # GEDI shots (+ row/col, slope after steps 2-3)

    @property
    def profiles_path(self) -> Path:
        return self.interim_dir / "tom_profiles.npz"       # TOM profile under every shot

    @property
    def cells_path(self) -> Path:
        return self.interim_dir / "cells.parquet"          # 500 m cell table

    @property
    def cell_profiles_path(self) -> Path:
        return self.interim_dir / "cell_profiles.npz"      # mean TOM profile per cell
