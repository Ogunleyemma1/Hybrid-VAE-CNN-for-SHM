# config.py
"""
Central configuration for the openLAB Hybrid VAE–CNN/ML pipeline.

This file is intentionally simple and import-safe:
- Pure constants (no side effects)
- Paths are resolved relative to this file (portable across machines)
"""

from __future__ import annotations

import os

# =============================================================================
# Project paths (resolved relative to this Codes/ folder)
# =============================================================================
CODES_DIR: str = os.path.dirname(os.path.abspath(__file__))  # .../Codes
PROJECT_DIR: str = os.path.dirname(CODES_DIR)               # .../20250506_openLAB_tests
DATA_DIR: str = os.path.join(PROJECT_DIR, "Data")           # .../Data

RAW_DIR: str = os.path.join(DATA_DIR, "raw")                # .../Data/raw  (MD_*.txt here)
OUT_DIR: str = os.path.join(DATA_DIR, "extracted")          # .../Data/extracted

# =============================================================================
# Windowing
# =============================================================================
SEQ_LEN: int = 200
STRIDE: int = 20

# =============================================================================
# Structural-fault thresholding (mm)
# =============================================================================
# Structural Fault is defined as: u_max > ALLOW_MAX
# NOTE: ALLOW_MIN is intentionally permissive / unused for labeling because
# normal displacement signals can be negative (oscillations around 0).
ALLOW_MIN: float = -1e9
ALLOW_MAX: float = 20.0

# =============================================================================
# Cleaning thresholds (openLAB-style)
# =============================================================================
OBSTRUCTION_SENTINEL: float = -1e5
CLEAN_MAX_JUMP_MM: float = 1.0
CLEAN_MAX_ABS_MM: float = 65.0
MOVING_AVG_WINDOW: int = 5

# =============================================================================
# Sensor-fault rules (RAW integrity)
# =============================================================================
RAW_DIFF_TH_MM: float = 1.0
RAW_ABS_TH_MM: float = 65.0
RAW_INVALID_RATIO_FAULT: float = 0.05  # 5% invalid -> SF window

# Optional load-aware flatline proxy (uses DMS range as load change indicator)
FLAT_VAR_EPS: float = 1e-6
FORCE_RANGE_FOR_FLATLINE: float = 5.0  # units consistent with DMS proxy thresholding logic

# =============================================================================
# Splits (run-based)
# =============================================================================
SEED: int = 42
TRAIN_FRAC: float = 0.40
VAL_FRAC: float = 0.30
TEST_FRAC: float = 0.30

# =============================================================================
# Artifact names (filenames only; directories are defined above)
# =============================================================================
ARTIFACTS: dict[str, str] = {
    "windows_clean": "X_clean.npy",
    "windows_raw": "X_raw.npy",
    "meta": "window_labels.csv",
    "splits": "run_split.json",
}

# Optional: explicit public API for linters and reviewers
__all__ = [
    "CODES_DIR",
    "PROJECT_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "OUT_DIR",
    "SEQ_LEN",
    "STRIDE",
    "ALLOW_MIN",
    "ALLOW_MAX",
    "OBSTRUCTION_SENTINEL",
    "CLEAN_MAX_JUMP_MM",
    "CLEAN_MAX_ABS_MM",
    "MOVING_AVG_WINDOW",
    "RAW_DIFF_TH_MM",
    "RAW_ABS_TH_MM",
    "RAW_INVALID_RATIO_FAULT",
    "FLAT_VAR_EPS",
    "FORCE_RANGE_FOR_FLATLINE",
    "SEED",
    "TRAIN_FRAC",
    "VAL_FRAC",
    "TEST_FRAC",
    "ARTIFACTS",
]
