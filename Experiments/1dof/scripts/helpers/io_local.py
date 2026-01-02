from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_npz(path: Path, **arrays) -> None:
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> dict:
    return dict(np.load(path))


def save_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
