"""Pytest path setup for isolated VL-P1 research package."""
from __future__ import annotations

from pathlib import Path
import sys

VL_P1_ROOT = Path(__file__).resolve().parent.parent
if str(VL_P1_ROOT) not in sys.path:
    sys.path.insert(0, str(VL_P1_ROOT))
