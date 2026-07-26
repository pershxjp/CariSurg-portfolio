"""
src/utils.py

Shared utility functions for the CariSurg ESI triage project.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible operations."""

    random.seed(seed)
    np.random.seed(seed)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return its Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)

    return directory
