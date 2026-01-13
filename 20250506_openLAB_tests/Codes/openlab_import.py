# utils_openlab_import.py
from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, os.PathLike]

# Expected number of header lines before the tabular data begins.
_CATMAN_SKIPROWS = 36

# Expected location of the T0 line in openLAB catman MD_*.txt exports.
# The original implementation assumes line 13 (index 12).
_T0_LINE_INDEX = 12

# Regex for extracting T0 = DD.MM.YYYY HH:MM:SS
_T0_PATTERN = re.compile(
    r"T0\s*=\s*(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2}):(\d{2})"
)

# Canonical column schema for the openLAB catman export used in this project.
_CATMAN_COLUMNS = [
    "Time_1", "DMS_1", "Time_2", "Force_N", "Force_A", "IWA", "Temp_Bridge", "Temp_Ambient",
    "Time_3", "LWA_1", "LWA_2", "LWA_3", "Time_4", "LWA_4", "LWA_5", "NMA_5", "F_total", "Comment",
]


def import_catman_file(file_path: PathLike) -> pd.DataFrame:
    """
    Read an openLAB catman MD_*.txt export file.

    This reader:
    - extracts the experiment start timestamp T0 from the header (line index 12),
    - loads the tabular data (tab-separated, decimal comma),
    - assigns a fixed column schema used in this project,
    - creates an absolute datetime column 'time' from T0 + Time_1 seconds.

    Parameters
    ----------
    file_path : str | os.PathLike
        Path to a catman MD_*.txt file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the catman data, with an additional 'time' column.
    """
    file_path = os.fspath(file_path)

    with open(file_path, encoding="cp1252") as f:
        lines = f.readlines()

    if len(lines) <= _T0_LINE_INDEX:
        raise ValueError(f"Unexpected header length in {file_path!r} (need > {_T0_LINE_INDEX} lines).")

    m = _T0_PATTERN.search(lines[_T0_LINE_INDEX])
    if m is None:
        raise ValueError(f"T0 not found in header of {file_path!r} (expected pattern: 'T0 = DD.MM.YYYY HH:MM:SS').")

    start_time = dt.datetime.strptime(
        f"{m.group(3)}-{m.group(2)}-{m.group(1)} {m.group(4)}:{m.group(5)}:{m.group(6)}",
        "%Y-%m-%d %H:%M:%S",
    )

    df = pd.read_csv(
        file_path,
        sep="\t",
        decimal=",",
        encoding="cp1252",
        skiprows=_CATMAN_SKIPROWS,
        on_bad_lines="skip",
    )

    df.columns = _CATMAN_COLUMNS

    df["Time_1"] = pd.to_numeric(df["Time_1"], errors="coerce")
    df["time"] = df["Time_1"].apply(
        lambda s: start_time + dt.timedelta(seconds=float(s)) if pd.notnull(s) else pd.NaT
    )
    return df


def run_id_from_path(file_path: PathLike) -> str:
    """
    Extract a run identifier from a file path.

    Parameters
    ----------
    file_path : str | os.PathLike
        Path to a data file.

    Returns
    -------
    str
        Basename without extension (e.g., 'MD_00123').
    """
    p = Path(file_path)
    return p.stem
