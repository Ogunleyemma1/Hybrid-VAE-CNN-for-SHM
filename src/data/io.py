# src/data/io.py
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create directory if it does not exist. Returns Path(path)."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: PathLike, obj: Any) -> None:
    """Save a JSON artifact with readable indentation."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: PathLike) -> Any:
    """Load a JSON artifact."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(path: PathLike, df: pd.DataFrame) -> None:
    """Save a CSV artifact (index suppressed)."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)


def load_csv(path: PathLike) -> pd.DataFrame:
    """Load a CSV artifact."""
    return pd.read_csv(Path(path))


def save_npy(path: PathLike, arr: np.ndarray) -> None:
    """Save a NumPy array to .npy (parent directories created)."""
    p = Path(path)
    ensure_dir(p.parent)
    np.save(p, arr)


def load_npy(path: PathLike, allow_pickle: bool = False) -> np.ndarray:
    """Load a NumPy array from .npy."""
    return np.load(Path(path), allow_pickle=allow_pickle)


def run_id_from_path(file_path: PathLike) -> str:
    """Extract run_id from a Catman file path (filename without extension)."""
    p = Path(file_path)
    return p.stem


def read_openlab_catman(file_path: PathLike, encoding: str = "cp1252") -> pd.DataFrame:
    """
    Read openLAB catman MD_*.txt format and return a DataFrame with an absolute time column.

    This is a direct, cleaned equivalent of your current import routine. :contentReference[oaicite:1]{index=1}

    Important:
    - Parses T0 from header line 13 (index 12), as observed in your files.
    - Reads tab-separated values, decimal comma.
    - Adds:
        df["time"] as datetime objects computed from Time_1 seconds offset.

    Returns:
        pd.DataFrame with standardized columns.
    """
    file_path = Path(file_path)

    with file_path.open("r", encoding=encoding) as f:
        lines = f.readlines()
    if len(lines) < 13:
        raise ValueError(f"Unexpected header length in {file_path}")

    # Header line 13 contains T0
    line = lines[12]
    m = re.search(r"T0\s*=\s*(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2}):(\d{2})", line)
    if not m:
        raise ValueError(f"T0 not found in {file_path}")

    start_time = _dt.datetime.strptime(
        f"{m.group(3)}-{m.group(2)}-{m.group(1)} {m.group(4)}:{m.group(5)}:{m.group(6)}",
        "%Y-%m-%d %H:%M:%S",
    )

    df = pd.read_csv(
        file_path,
        sep="\t",
        decimal=",",
        encoding=encoding,
        skiprows=36,
        on_bad_lines="skip",
    )

    # Standardize columns (matches your current mapping)
    df.columns = [
        "Time_1", "DMS_1",
        "Time_2", "Force_N", "Force_A",
        "IWA", "Temp_Bridge", "Temp_Ambient",
        "Time_3", "LWA_1", "LWA_2", "LWA_3",
        "Time_4", "LWA_4", "LWA_5", "NMA_5",
        "F_total", "Comment",
    ]

    df["Time_1"] = pd.to_numeric(df["Time_1"], errors="coerce")
    df["time"] = df["Time_1"].apply(
        lambda s: start_time + _dt.timedelta(seconds=float(s)) if pd.notnull(s) else pd.NaT
    )
    return df
