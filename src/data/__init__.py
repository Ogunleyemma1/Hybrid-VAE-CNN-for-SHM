# src/data/__init__.py
"""
Data subpackage for the OpenLAB hybrid SHM repository.

Design principles (reviewer-facing):
- Deterministic preprocessing: split-by-run, saved manifests, seed-controlled behavior.
- Clear shape conventions:
    * Continuous signals: (T, C)
    * Windowed signals:    (N, L, C)  where L = seq_len
- Separation of concerns:
    * io.py            -> filesystem + structured artifact I/O
    * windowing.py     -> segmentation into fixed-length windows
    * normalization.py -> train-fit statistics + apply to splits
    * splits.py        -> deterministic split generation and persistence
"""

from .io import (
    ensure_dir,
    save_json,
    load_json,
    save_csv,
    load_csv,
    save_npy,
    load_npy,
    read_openlab_catman,
    run_id_from_path,
)
from .windowing import windowize_2d, windowize_1d
from .splits import make_run_split, apply_forced_runs
from .normalization import (
    fit_standard_scaler,
    apply_standard_scaler,
    fit_robust_scaler,
    apply_robust_scaler,
    save_norm_stats,
    load_norm_stats,
)

__all__ = [
    # io
    "ensure_dir",
    "save_json",
    "load_json",
    "save_csv",
    "load_csv",
    "save_npy",
    "load_npy",
    "read_openlab_catman",
    "run_id_from_path",
    # windowing
    "windowize_2d",
    "windowize_1d",
    # splits
    "make_run_split",
    "apply_forced_runs",
    # normalization
    "fit_standard_scaler",
    "apply_standard_scaler",
    "fit_robust_scaler",
    "apply_robust_scaler",
    "save_norm_stats",
    "load_norm_stats",
]
