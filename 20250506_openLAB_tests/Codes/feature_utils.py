# utils_features.py
"""
Feature and rule utilities for the openLAB Hybrid VAE–CNN/ML pipeline.

This module provides:
- basic smoothing utility (moving average)
- provider-aligned cleaning rules for displacement channels
- windowization utilities (1D and 2D)
- simple slope statistics
- window-level weak-supervision helpers for sensor-fault (SF) flags/metrics

All functions are deterministic and side-effect free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# Basic utilities
# =============================================================================
def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """
    Apply a centered moving average of width `w`.

    Parameters
    ----------
    x : np.ndarray
        1D array.
    w : int
        Window length. If w <= 1, returns input unchanged.

    Returns
    -------
    np.ndarray
        Smoothed array (same length).
    """
    if w is None or w <= 1:
        return x
    kern = np.ones(int(w), dtype=float) / float(w)
    return np.convolve(x, kern, mode="same")


# =============================================================================
# Provider-aligned cleaning for openLAB displacement channels
# =============================================================================
def clean_openlab_and_rule(x: np.ndarray, max_jump: float = 1.0, max_abs: float = 65.0, ma_window: int = 5):
    """
    Provider-aligned AND-rule cleaning for displacement signals.

    A sample is removed if:
      (|Δx| > max_jump) AND (|x| > max_abs)

    Invalid values (NaN/inf) are removed as well. Removed samples are linearly
    interpolated, followed by optional moving-average smoothing.

    Parameters
    ----------
    x : np.ndarray
        1D signal.
    max_jump : float
        Jump threshold on absolute difference between consecutive samples.
    max_abs : float
        Absolute magnitude threshold.
    ma_window : int
        Moving average window length.

    Returns
    -------
    cleaned : np.ndarray
        Cleaned signal (float32).
    removed_mask : np.ndarray
        Mask of removed/invalid samples (float32; 1.0 indicates removed).
    """
    x = np.asarray(x, dtype=float)
    removed = np.zeros_like(x, dtype=bool)

    # Invalid values -> removed
    bad = ~np.isfinite(x)
    x2 = x.copy()
    x2[bad] = np.nan
    removed[bad] = True

    for i in range(1, len(x2)):
        if np.isfinite(x2[i]) and np.isfinite(x2[i - 1]):
            if (abs(x2[i] - x2[i - 1]) > float(max_jump)) and (abs(x2[i]) > float(max_abs)):
                x2[i] = np.nan
                removed[i] = True
        else:
            x2[i] = np.nan
            removed[i] = True

    s = pd.Series(x2)
    xi = s.interpolate(limit_direction="both").to_numpy()
    xi = moving_average(xi, ma_window)

    return xi.astype(np.float32), removed.astype(np.float32)


def provider_raw_outlier_mask(x_raw: np.ndarray, diff_th: float = 1.0, abs_th: float = 65.0):
    """
    Provider-style AND-rule outlier mask for RAW displacement (per documentation).

    A sample is flagged if:
      (|Δx| >= diff_th) AND (|x| >= abs_th)
    plus invalid samples (NaN/inf).

    Returns mask float32 (1.0 indicates outlier/invalid).
    """
    x = np.asarray(x_raw, dtype=float)
    n = x.size
    m = np.zeros((n,), dtype=bool)

    # invalid always outlier
    m |= ~np.isfinite(x)

    if n > 1:
        dx = np.abs(np.diff(x))
        # flag index i based on dx at i-1 and abs(x[i])
        m[1:] |= (dx >= float(diff_th)) & (np.abs(x[1:]) >= float(abs_th))

    return m.astype(np.float32)


# =============================================================================
# Windowization utilities
# =============================================================================
def windowize_2d(A: np.ndarray, seq_len: int, stride: int):
    """
    Convert a 2D array A (N, K) into overlapping windows (W, seq_len, K).

    Returns empty arrays if N < seq_len.

    Returns
    -------
    X : np.ndarray
        Windowed array (float32).
    idx0 : np.ndarray
        Starting indices for each window (int).
    """
    n = A.shape[0]
    if n < seq_len:
        return np.empty((0, seq_len, A.shape[1]), dtype=np.float32), np.empty((0,), dtype=int)

    X, idx0 = [], []
    for i in range(0, n - seq_len + 1, stride):
        X.append(A[i : i + seq_len])
        idx0.append(i)

    return np.asarray(X, dtype=np.float32), np.asarray(idx0, dtype=int)


def windowize_1d(x: np.ndarray, seq_len: int, stride: int):
    """
    Convert a 1D array x (N,) into overlapping windows (W, seq_len).

    Returns empty arrays if N < seq_len.

    Returns
    -------
    W : np.ndarray
        Windowed array (float32).
    idx0 : np.ndarray
        Starting indices for each window (int).
    """
    n = x.shape[0]
    if n < seq_len:
        return np.empty((0, seq_len), dtype=np.float32), np.empty((0,), dtype=int)

    W, idx0 = [], []
    for i in range(0, n - seq_len + 1, stride):
        W.append(x[i : i + seq_len])
        idx0.append(i)

    return np.asarray(W, dtype=np.float32), np.asarray(idx0, dtype=int)


def slope_stats(x: np.ndarray):
    """
    Simple local slope statistics (per window).

    Returns
    -------
    rms_abs_dx : float
        RMS of dx (not absolute; corresponds to RMS magnitude).
    max_abs_dx : float
        Max absolute dx.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0, 0.0

    dx = np.diff(x)
    rms = float(np.sqrt(np.mean(dx * dx)))
    mx = float(np.max(np.abs(dx)))
    return rms, mx


# =============================================================================
# Weak-supervision / silver-label rule helpers (window-level)
# =============================================================================
def invalid_ratio_1d(x: np.ndarray) -> float:
    """Fraction of invalid samples (NaN/inf) in a 1D array."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(~np.isfinite(x))) if x.size else 0.0


def jump_ratio_1d(x: np.ndarray, delta: float) -> float:
    """
    Fraction of time steps where |x[t] - x[t-1]| >= delta among valid consecutive samples.

    Robust to NaNs: only diffs where both points are finite are considered.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0

    x0 = x[:-1]
    x1 = x[1:]
    ok = np.isfinite(x0) & np.isfinite(x1)
    if not np.any(ok):
        return 0.0

    dx = np.abs(x1[ok] - x0[ok])
    return float(np.mean(dx >= float(delta)))


def range_violation_ratio_1d(x: np.ndarray, abs_th: float) -> float:
    """Fraction of samples where |x| >= abs_th among finite samples."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if not np.any(ok):
        return 0.0
    return float(np.mean(np.abs(x[ok]) >= float(abs_th)))


def is_stuck_1d(x: np.ndarray, var_eps: float) -> bool:
    """Flatline / stuck detection based on low variance."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if np.sum(ok) < 5:
        return False
    return bool(np.var(x[ok]) < float(var_eps))


def is_stuck_force_aware(u: np.ndarray, f: np.ndarray, var_eps: float, force_rng_min: float) -> bool:
    """
    Force-aware flatline detection.

    A window is marked stuck if:
      - displacement variance is tiny, AND
      - force changes meaningfully within the window
    """
    u = np.asarray(u, dtype=float)
    f = np.asarray(f, dtype=float)

    u_ok = np.isfinite(u)
    f_ok = np.isfinite(f)
    if np.sum(u_ok) < 5 or np.sum(f_ok) < 5:
        return False

    u_var = float(np.var(u[u_ok]))
    f_rng = float(np.max(f[f_ok]) - np.min(f[f_ok]))
    return bool((u_var < float(var_eps)) and (f_rng > float(force_rng_min)))


def channel_inconsistency_score(U: np.ndarray, zthr: float = 4.0) -> float:
    """
    Generic multi-channel inconsistency score for a window.

    Parameters
    ----------
    U : np.ndarray
        Window data with shape (T, K).
    zthr : float
        Robust z-score threshold across channels.

    Returns
    -------
    float
        Ratio of time steps with any channel exceeding zthr.
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or U.shape[0] < 2 or U.shape[1] < 2:
        return 0.0

    ok = np.all(np.isfinite(U), axis=1)
    if np.sum(ok) < 5:
        return 0.0
    V = U[ok]

    med = np.median(V, axis=1, keepdims=True)
    mad = np.median(np.abs(V - med), axis=1, keepdims=True) + 1e-9
    z = np.abs((V - med) / (1.4826 * mad))

    extreme = np.any(z >= float(zthr), axis=1)
    return float(np.mean(extreme))


def sensor_fault_silver_flags(
    u_raw: np.ndarray,
    u_clean: np.ndarray | None = None,
    f: np.ndarray | None = None,
    *,
    jump_th: float = 1.0,
    abs_th: float = 65.0,
    invalid_ratio_th: float = 0.05,
    var_eps: float = 1e-6,
    force_rng_min: float = 0.0,
    use_plain_stuck: bool = True,
) -> dict:
    """
    Compute interpretable window-level silver-rule metrics/flags for sensor fault (SF).

    Policy for sf_any:
    - flags if invalid_ratio >= invalid_ratio_th
    - flags if any jumps are present (jump_ratio > 0)
    - flags if any range violations are present (range_violation_ratio > 0)
    - flags stuck when:
        * force-aware stuck is available (f provided and force_rng_min > 0), OR
        * plain stuck is enabled (use_plain_stuck=True)

    Returns
    -------
    dict
        Keys:
          invalid_ratio, jump_ratio, range_violation_ratio,
          stuck, stuck_forceaware, sf_any
    """
    u_raw = np.asarray(u_raw, dtype=float)

    inv_ratio = invalid_ratio_1d(u_raw)
    jr = jump_ratio_1d(u_raw, jump_th)
    rr = range_violation_ratio_1d(u_raw, abs_th)

    u_for_stuck = np.asarray(u_clean, dtype=float) if u_clean is not None else u_raw
    stuck = is_stuck_1d(u_for_stuck, var_eps)

    stuck_fa = False
    if (f is not None) and (force_rng_min > 0.0):
        stuck_fa = is_stuck_force_aware(u_for_stuck, f, var_eps, force_rng_min)

    stuck_term = stuck_fa or (use_plain_stuck and stuck)

    sf_any = (
        (inv_ratio >= float(invalid_ratio_th))
        or (jr > 0.0)
        or (rr > 0.0)
        or bool(stuck_term)
    )

    return {
        "invalid_ratio": float(inv_ratio),
        "jump_ratio": float(jr),
        "range_violation_ratio": float(rr),
        "stuck": int(stuck),
        "stuck_forceaware": int(stuck_fa),
        "sf_any": int(sf_any),
    }
