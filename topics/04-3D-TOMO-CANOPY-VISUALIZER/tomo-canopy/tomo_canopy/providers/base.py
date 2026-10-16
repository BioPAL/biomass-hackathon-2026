"""Shared interfaces for data providers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataProvider(Protocol):
    """Contract implemented by data-source adapters.

    Concrete providers may return an xarray object, point cloud, or another
    domain-specific representation. The exact data model will be refined as
    the ingestion APIs are implemented.
    """

    def open(self, source: str | Path, **kwargs: Any) -> Any:
        """Open *source* and return its in-memory or lazy representation."""
        ...

