# experiments/openlab/scripts/00_import_raw_openlab.py
"""
Step 00: Import raw openLAB Catman measurement files into a normalized intermediate format.

Input:
  experiments/openlab/datasets/raw/MD_*.txt

Output:
  experiments/openlab/datasets/processed/runs/<run_id>.npz
  experiments/openlab/datasets/processed/runs_manifest.json

The NPZ contains:
  - time_s: (M,)
  - force_total_kN: (M,)  (if present)
  - disp_mm: (M, K) for selected displacement channels
  - disp_cols: list[str] channel names
  - file: original filename

This step is intentionally separate from windowing (Step 01) for transparency/reproducibility.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils import configure_logging, default_experiment_dirs, find_repo_root, resolve_under_root


# ---------------------------
# Catman parser (robust)
# ---------------------------
_CATMAN_TIME_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2}).*?(\d{2}):(\d{2}):(\d{2})")


def run_id_from_path(p: Path) -> str:
    return p.stem


def _parse_start_time_from_header(lines: List[str]) -> dt.datetime | None:
    """
    Many Catman exports include an absolute start time in the header.
    Your legacy parser used line 13 (index 12). We keep a robust fallback.

    Returns None if not found.
    """
    for line in lines[:40]:
        m = _CATMAN_TIME_RE.search(line)
        if m:
            y, mo, d, hh, mm, ss = map(int, m.groups())
            return dt.datetime(y, mo, d, hh, mm, ss)
    return None


def import_catman_file(file_path: Path, encoding: str = "cp1252") -> pd.DataFrame:
    """
    Read openLAB MD_*.txt Catman format into a DataFrame with stable column names.

    The raw files typically have a header then a CSV-like numeric section.
    We:
      1) read the entire file lines
      2) detect the start of numeric data by finding the first line that begins with a number
      3) read the numeric portion with pandas
      4) attach 'time_abs' if start time exists

    IMPORTANT:
      Channel naming depends on the exported file.
      In Step 00 we normalize by renaming known channel patterns to a stable set.
    """
    lines = file_path.read_text(encoding=encoding, errors="ignore").splitlines()
    start_time = _parse_start_time_from_header(lines)

    # Find first numeric line index
    data_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if s[0].isdigit() or (s[0] == "-" and len(s) > 1 and s[1].isdigit()):
            data_start = i
            break
    if data_start is None:
        raise ValueError(f"Could not locate numeric data section in: {file_path}")

    # Load numeric section (Catman typically uses comma separator, sometimes semicolon)
    raw_text = "\n".join(lines[data_start:])
    # Try comma first; if too few columns, retry semicolon.
    df = pd.read_csv(pd.io.common.StringIO(raw_text), header=None)
    if df.shape[1] <= 3:
        df = pd.read_csv(pd.io.common.StringIO(raw_text), header=None, sep=";")

    # If your dataset matches the common openLAB export, it has 18 columns (incl comment)
    # We keep the “expected” schema but tolerate differences.
    expected_cols = [
        "Time_1", "DMS_1", "Time_2", "Force_N", "Force_A", "IWA", "Temp_Bridge", "Temp_Ambient",
        "Time_3", "LWA_1", "LWA_2", "LWA_3", "Time_4", "LWA_4", "LWA_5", "NMA_5", "F_total", "Comment"
    ]
    if df.shape[1] >= len(expected_cols):
        df = df.iloc[:, : len(expected_cols)]
        df.columns = expected_cols
    else:
        # Fallback: name generically
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

    # Coerce primary time column
    if "Time_1" in df.columns:
        df["Time_1"] = pd.to_numeric(df["Time_1"], errors="coerce")
        if start_time is not None:
            df["time_abs"] = df["Time_1"].apply(
                lambda s: start_time + dt.timedelta(seconds=float(s)) if pd.notnull(s) else pd.NaT
            )

    return df


# ---------------------------
# Channel selection
# ---------------------------
DEFAULT_DISP_COLS = ["LWA_1", "LWA_2", "LWA_3", "LWA_4"]  # your pipeline uses 4 for CNN


def extract_arrays(df: pd.DataFrame, disp_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      time_s: (M,)
      force_total_kN: (M,) (nan if absent)
      disp_mm: (M, K)
    """
    # time
    if "Time_1" in df.columns:
        time_s = pd.to_numeric(df["Time_1"], errors="coerce").to_numpy(dtype=np.float32)
    else:
        time_s = np.arange(len(df), dtype=np.float32)

    # force
    if "F_total" in df.columns:
        force_total = pd.to_numeric(df["F_total"], errors="coerce").to_numpy(dtype=np.float32)
    else:
        force_total = np.full(len(df), np.nan, dtype=np.float32)

    # displacement columns
    disp = []
    used = []
    for c in disp_cols:
        if c in df.columns:
            disp.append(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32))
            used.append(c)
    if len(disp) == 0:
        raise ValueError(f"No displacement columns found. Requested={disp_cols}, df_cols={list(df.columns)}")

    disp_mm = np.stack(disp, axis=1)  # (M,K)
    return time_s, force_total, disp_mm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="experiments/openlab/datasets/raw")
    ap.add_argument("--processed_dir", type=str, default="experiments/openlab/datasets/processed")
    ap.add_argument("--pattern", type=str, default="MD_*.txt")
    ap.add_argument("--disp_cols", type=str, nargs="*", default=DEFAULT_DISP_COLS)
    args = ap.parse_args()

    root = find_repo_root()
    raw_dir = resolve_under_root(args.raw_dir, root=root)
    processed_dir = resolve_under_root(args.processed_dir, root=root)

    dirs = default_experiment_dirs("experiments/openlab")
    logger = configure_logging(name="openlab", log_file=dirs["logs"] / "00_import_raw_openlab.log")

    runs_dir = processed_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No raw files found in {raw_dir} matching '{args.pattern}'")

    manifest: Dict[str, Dict] = {"raw_dir": str(raw_dir), "pattern": args.pattern, "runs": {}}

    logger.info(f"Found {len(files)} raw files in: {raw_dir}")

    for fp in files:
        run_id = run_id_from_path(fp)
        logger.info(f"Importing: {fp.name}  -> run_id={run_id}")

        df = import_catman_file(fp)
        time_s, force_total, disp_mm = extract_arrays(df, args.disp_cols)

        out_npz = runs_dir / f"{run_id}.npz"
        np.savez_compressed(
            out_npz,
            time_s=time_s,
            force_total_kN=force_total,
            disp_mm=disp_mm,
            disp_cols=np.array(args.disp_cols, dtype=object),
            file=str(fp.name),
        )

        manifest["runs"][run_id] = {
            "file": fp.name,
            "npz": str(out_npz.relative_to(processed_dir)),
            "n_samples": int(len(time_s)),
            "n_disp_channels": int(disp_mm.shape[1]),
            "disp_cols": args.disp_cols,
        }

    manifest_path = processed_dir / "runs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Saved manifest: {manifest_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
