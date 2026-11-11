"""Configuration loading (config/settings.yaml + environment overrides)."""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def _load_file() -> dict:
    with SETTINGS_FILE.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_settings() -> dict:
    s = copy.deepcopy(_load_file())
    if os.environ.get("SUNSIGHT_MODEL_PATH"):
        s["model"]["path"] = os.environ["SUNSIGHT_MODEL_PATH"]
    if os.environ.get("SUNSIGHT_DATASET_DIR"):
        s["dataset"]["dir"] = os.environ["SUNSIGHT_DATASET_DIR"]
    return s


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p
