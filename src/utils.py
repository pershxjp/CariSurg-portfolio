"""
src/utils.py
============

Shared helpers used across the pipeline: config loading, model
persistence, and lightweight logging. Nothing model-specific lives here
-- see src/model.py for that.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML config file (e.g. config.yaml) into a plain dict.

    Raises FileNotFoundError with a clear message if the path doesn't
    exist, rather than a raw YAML parser traceback.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_logger(name: str = "carisurg_triage", level: int = logging.INFO) -> logging.Logger:
    """
    A single, simple logger configuration used across scripts/train.py
    and any future entry points, so log formatting is consistent.
    Safe to call repeatedly (won't add duplicate handlers).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory of `path` if it doesn't already exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model: Any, path: str | Path) -> Path:
    """Save a fitted model/pipeline to disk with joblib, creating parent dirs as needed."""
    path = ensure_parent_dir(path)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path) -> Any:
    """Load a model/pipeline previously saved with save_model()."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
