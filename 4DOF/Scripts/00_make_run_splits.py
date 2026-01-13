from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

SEED = 42

# 40/30/30 (must sum to 1.0)
TRAIN_FRAC = 0.40
VAL_FRAC = 0.30
TEST_FRAC = 0.30

SEQ_LEN = 100  # MUST match Scripts/Models/cnn_model.py
STRIDE = 1     # MUST match all windowing scripts

NORMAL_GLOB = "Data/raw/normal/*.csv"
SENSOR_GLOB = "Data/raw/faults/sensor_fault/**/*.csv"
STRUCT_GLOB = "Data/raw/faults/structural_fault/**/*.csv"

OUT_PATH = Path("Data/processed/run_splits.json")


def _stable_int(s: str) -> int:
    """Stable across OS/runs: convert string to int via md5."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _n_windows(n_rows: int, win_len: int, stride: int) -> int:
    if n_rows < win_len:
        return 0
    return 1 + (n_rows - win_len) // stride


def _count_rows_csv(path: str) -> int:
    # Fast line count; assumes 1 header row
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        n_lines = sum(1 for _ in f)
    return max(n_lines - 1, 0)


def _relpaths(pattern: str) -> List[str]:
    return [p.replace("\\", "/") for p in sorted(glob.glob(pattern, recursive=True))]


def _split_indices_contiguous(n: int) -> Dict[str, List[int]]:
    """
    Option A (OLD CONCEPT): contiguous time-block split of window indices [0..n-1].
      train = [0 .. n_tr-1]
      val   = [n_tr .. n_tr+n_va-1]
      test  = [n_tr+n_va .. n-1]

    Rounding is handled so totals sum exactly to n.
    """
    if n <= 0:
        return {"train": [], "val": [], "test": []}

    # Use floor for train/val, remainder to test (stable, no shuffle)
    n_tr = int(TRAIN_FRAC * n)
    n_va = int(VAL_FRAC * n)
    n_te = n - n_tr - n_va

    # Safety: if tiny n causes empty splits, still keep totals consistent
    if n_te < 0:
        n_te = 0
    if n_tr + n_va + n_te != n:
        n_te = n - n_tr - n_va

    tr = list(range(0, n_tr))
    va = list(range(n_tr, n_tr + n_va))
    te = list(range(n_tr + n_va, n_tr + n_va + n_te))

    assert len(tr) + len(va) + len(te) == n
    return {"train": tr, "val": va, "test": te}


def build_group(file_list: List[str], seed_offset: int) -> Tuple[Dict[str, object], int, int, int]:
    """
    Returns:
      group_dict = { "files": [...], "window_indices": {file: {train:[], val:[], test:[]}} }
      totals for train/val/test windows (across files)
    """
    files: List[str] = []
    win_map: Dict[str, Dict[str, List[int]]] = {}
    tr_tot = va_tot = te_tot = 0

    for fp in file_list:
        if not fp.lower().endswith(".csv"):
            continue

        n_rows = _count_rows_csv(fp)
        n_win = _n_windows(n_rows, SEQ_LEN, STRIDE)
        if n_win <= 0:
            continue

        files.append(fp)

        # kept for traceability (even though split is contiguous)
        _ = SEED + seed_offset + _stable_int(fp)

        split = _split_indices_contiguous(n_win)
        win_map[fp] = split

        tr_tot += len(split["train"])
        va_tot += len(split["val"])
        te_tot += len(split["test"])

    group = {"files": files, "window_indices": win_map}
    return group, tr_tot, va_tot, te_tot


def main() -> None:
    normal_files = _relpaths(NORMAL_GLOB)
    sensor_files = _relpaths(SENSOR_GLOB)
    struct_files = _relpaths(STRUCT_GLOB)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    normal_group, ntr, nva, nte = build_group(normal_files, seed_offset=0)
    sensor_group, str_, sva, ste = build_group(sensor_files, seed_offset=10_000)
    struct_group, ttr, tva, tte = build_group(struct_files, seed_offset=20_000)

    out = {
        "mode": "window_level_per_file",
        "seed": SEED,
        "fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
        "normal": normal_group,
        "sensor_fault": sensor_group,
        "structural_fault": struct_group,
        "totals": {
            "normal": {"train": ntr, "val": nva, "test": nte},
            "sensor_fault": {"train": str_, "val": sva, "test": ste},
            "structural_fault": {"train": ttr, "val": tva, "test": tte},
        },
        "note": "Option A contiguous time-block split per file (no shuffle).",
    }

    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] wrote: {OUT_PATH.as_posix()}")
    print("[OK] mode=window_level_per_file | split=contiguous blocks (Option A)")


if __name__ == "__main__":
    main()
