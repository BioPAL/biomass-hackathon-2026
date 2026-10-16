"""Make the topic folder importable as the root of the 'src' package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
