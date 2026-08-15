from __future__ import annotations
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent


def reference_topology(name: str) -> Path:
    """Path to a packaged reference topology JSON, e.g. reference_topology("german_17")."""
    return _DATA_DIR / f"{name}.json"


def reference_state(name: str) -> Path:
    """Path to a packaged prebuilt operating-network state file, e.g. reference_state("german_17")."""
    return _DATA_DIR / f"{name}_state.json"
