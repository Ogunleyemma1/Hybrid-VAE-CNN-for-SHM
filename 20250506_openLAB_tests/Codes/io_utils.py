# utils_io.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

PathLike = Union[str, os.PathLike]


def ensure_dir(path: PathLike) -> str:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path : str | os.PathLike
        Directory path.

    Returns
    -------
    str
        The directory path as a string.
    """
    p = os.fspath(path)
    os.makedirs(p, exist_ok=True)
    return p


def save_json(path: PathLike, obj: Any, *, indent: int = 2) -> None:
    """
    Save a Python object to a JSON file.

    Notes
    -----
    Uses UTF-8 encoding and stable indentation for reproducible diffs.
    """
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def load_json(path: PathLike) -> Any:
    """Load a JSON file and return its decoded Python object."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(path: PathLike, arr: np.ndarray) -> None:
    """
    Save a NumPy array to .npy.

    Notes
    -----
    Uses NumPy's default safe format (no pickle).
    """
    p = Path(path)
    ensure_dir(p.parent)
    np.save(p, arr, allow_pickle=False)


def load_npy(path: PathLike, *, mmap_mode: str | None = None) -> np.ndarray:
    """
    Load a NumPy array from .npy.

    Parameters
    ----------
    path : str | os.PathLike
        Path to .npy file.
    mmap_mode : str | None
        Optional memory mapping mode (e.g., 'r') for large arrays.

    Returns
    -------
    np.ndarray
    """
    p = Path(path)
    return np.load(p, allow_pickle=False, mmap_mode=mmap_mode)


def save_csv(path: PathLike, df: pd.DataFrame) -> None:
    """
    Save a pandas DataFrame to CSV (no index).

    Notes
    -----
    Uses UTF-8 and consistent line endings for portability.
    """
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False, encoding="utf-8")


def load_csv(path: PathLike, **kwargs: Any) -> pd.DataFrame:
    """
    Load a CSV into a pandas DataFrame.

    Parameters
    ----------
    path : str | os.PathLike
        Path to CSV.
    **kwargs
        Forwarded to pandas.read_csv (e.g., dtype=..., parse_dates=...).

    Returns
    -------
    pd.DataFrame
    """
    p = Path(path)
    return pd.read_csv(p, **kwargs)
