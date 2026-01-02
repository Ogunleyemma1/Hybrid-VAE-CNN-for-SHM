# src/utils/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping/dict, got {type(data)}")
    return data


def get_section(cfg: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    out: Dict[str, Any] = cfg
    for k in keys:
        v = out.get(k, {})
        if v is None:
            v = {}
        if not isinstance(v, dict):
            raise ValueError(f"Config section {'.'.join(keys)} must be a dict")
        out = v
    return out
