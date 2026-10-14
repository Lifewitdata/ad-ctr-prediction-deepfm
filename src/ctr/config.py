"""Load the single project-wide configuration file (config.yaml)."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["project_root"] = PROJECT_ROOT
    return cfg


def data_file(cfg: dict, key: str) -> Path:
    return PROJECT_ROOT / cfg["data"][key]


def artifacts_dir(cfg: dict) -> Path:
    return PROJECT_ROOT / cfg["artifacts_dir"]


def reports_dir(cfg: dict) -> Path:
    return PROJECT_ROOT / cfg["reports_dir"]
