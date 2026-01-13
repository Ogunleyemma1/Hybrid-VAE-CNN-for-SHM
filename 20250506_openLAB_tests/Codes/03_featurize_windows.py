"""
03_featurize_windows.py

Compute fixed-length feature vectors from windowed signals (X_raw.npy) for ML baselines.

Inputs
------
- X_raw.npy : window tensor (N, T, C)
- window_labels_augmented.csv (preferred if present) OR window_labels.csv

Outputs
-------
- ML_Features/X_feat.npy
- ML_Features/y.npy
- ML_Features/meta_used.csv
- ML_Features/feat_names.json
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import config as C
from io_utils import ensure_dir


# =============================================================================
# Paths
# =============================================================================
DATA_DIR = C.OUT_DIR
X_RAW_PATH = os.path.join(DATA_DIR, C.ARTIFACTS["windows_raw"])  # (N, T, C)

# Prefer augmented if present; otherwise use standard meta produced by extractor
META_AUG_PATH = os.path.join(DATA_DIR, "window_labels_augmented.csv")
META_STD_PATH = os.path.join(DATA_DIR, C.ARTIFACTS["meta"])

FEATURE_DIR = os.path.join(C.CODES_DIR, "ML_Features")
ensure_dir(FEATURE_DIR)

X_FEAT_PATH = os.path.join(FEATURE_DIR, "X_feat.npy")
Y_PATH = os.path.join(FEATURE_DIR, "y.npy")
META_USED_PATH = os.path.join(FEATURE_DIR, "meta_used.csv")
FEAT_NAMES_PATH = os.path.join(FEATURE_DIR, "feat_names.json")


# =============================================================================
# Label mapping
# =============================================================================
LABEL_MAP = {
    "Normal": 0,
    "Structural Fault": 1,
    "Sensor Fault": 2,
}


# =============================================================================
# Numeric helpers (NaN-safe)
# =============================================================================
def _finite(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)


def _nanmean(x: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    ok = _finite(x)
    return float(np.mean(x[ok])) if np.any(ok) else 0.0


def _nanstd(x: np.ndarray, eps: float = 1e-12) -> float:
    x = x.astype(np.float64, copy=False)
    ok = _finite(x)
    if not np.any(ok):
        return 1.0
    s = float(np.std(x[ok], dtype=np.float64))
    return float(s if s > eps else 1.0)


def _nanmin(x: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    ok = _finite(x)
    return float(np.min(x[ok])) if np.any(ok) else 0.0


def _nanmax(x: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    ok = _finite(x)
    return float(np.max(x[ok])) if np.any(ok) else 0.0


def _nanrms(x: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    ok = _finite(x)
    if not np.any(ok):
        return 0.0
    v = float(np.mean(x[ok] * x[ok], dtype=np.float64))
    return float(np.sqrt(max(v, 0.0)))


def _nanskew(x: np.ndarray, eps: float = 1e-12) -> float:
    x = x.astype(np.float64, copy=False)
    ok = _finite(x)
    if np.sum(ok) < 3:
        return 0.0
    xv = x[ok]
    mu = float(np.mean(xv))
    sd = float(np.std(xv))
    if sd < eps:
        return 0.0
    z = (xv - mu) / sd
    z = np.clip(z, -50.0, 50.0)
    return float(np.mean(z**3))


def _nankurtosis(x: np.ndarray, eps: float = 1e-12) -> float:
    x = x.astype(np.float64, copy=False)
    ok = _finite(x)
    if np.sum(ok) < 4:
        return 0.0
    xv = x[ok]
    mu = float(np.mean(xv))
    sd = float(np.std(xv))
    if sd < eps:
        return 0.0
    z = (xv - mu) / sd
    z = np.clip(z, -50.0, 50.0)
    return float(np.mean(z**4) - 3.0)  # excess kurtosis


def _crest_factor(x: np.ndarray, eps: float = 1e-12) -> float:
    rms = _nanrms(x)
    if rms < eps:
        return 0.0
    mx = _nanmax(np.abs(x))
    return float(mx / rms)


def _snr_db(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    SNR-like proxy: signal power / noise power where noise = x - mean(x).
    NaN-safe.
    """
    x = x.astype(np.float64, copy=False)
    ok = _finite(x)
    if np.sum(ok) < 5:
        return 0.0
    xv = x[ok]
    mu = float(np.mean(xv))
    sig = float(np.mean(xv * xv))
    if sig < eps:
        return 0.0
    noise = xv - mu
    p_noise = float(np.mean(noise * noise))
    if p_noise < eps:
        return 60.0
    return float(10.0 * np.log10(sig / p_noise))


def _bandpower_features(x: np.ndarray, n_bands: int = 5) -> list[float]:
    """
    Simple frequency-domain descriptors from rFFT power spectrum.

    Returns:
      [total_power, centroid, rolloff_85, band1..bandN]

    Notes:
    - NaN-safe: uses finite samples only; if too few, returns zeros.
    - Frequency axis is in FFT-bin units (no sampling rate assumed).
    """
    x = x.astype(np.float64, copy=False)
    ok = _finite(x)
    if np.sum(ok) < 8:
        return [0.0, 0.0, 0.0] + [0.0] * int(n_bands)

    xv = x[ok]
    xv = xv - np.mean(xv)
    T = xv.shape[0]

    X = np.fft.rfft(xv)
    P = (np.abs(X) ** 2) / max(T, 1)
    if P.size > 0:
        P[0] = 0.0  # remove DC
    freqs = np.arange(P.size, dtype=np.float64)

    total = float(np.sum(P))
    if total <= 1e-18:
        return [total, 0.0, 0.0] + [0.0] * int(n_bands)

    centroid = float(np.sum(freqs * P) / total)

    cumsum = np.cumsum(P)
    roll_idx = int(np.searchsorted(cumsum, 0.85 * total))
    rolloff = float(min(roll_idx, P.size - 1))

    edges = np.linspace(0, P.size, int(n_bands) + 1).astype(int)
    bands: list[float] = []
    for i in range(int(n_bands)):
        a, b = edges[i], edges[i + 1]
        bands.append(float(np.sum(P[a:b]) / total))

    return [total, centroid, rolloff] + bands


# =============================================================================
# Featurization
# =============================================================================
def featurize_channel(x: np.ndarray, *, include_freq: bool = True) -> list[float]:
    """
    Extract NaN-safe features for a single channel.

    Parameters
    ----------
    x : np.ndarray
        Shape (T,)
    include_freq : bool
        Whether to include FFT-based bandpower descriptors.

    Returns
    -------
    list[float]
        Feature vector for the channel.
    """
    x = x.astype(np.float64, copy=False)

    mu = _nanmean(x)
    sd = _nanstd(x)
    v = float(sd * sd)
    mn = _nanmin(x)
    mx = _nanmax(x)
    ptp = float(mx - mn)
    rms = _nanrms(x)
    cf = _crest_factor(x)
    sk = _nanskew(x)
    ku = _nankurtosis(x)
    snr = _snr_db(x)

    feats = [mu, sd, v, mn, mx, ptp, rms, cf, sk, ku, snr]

    if include_freq:
        feats += _bandpower_features(x, n_bands=5)

    return feats


def main(*, include_freq: bool = True, drop_sensor_fault: bool = False) -> None:
    if not os.path.isfile(X_RAW_PATH):
        raise FileNotFoundError(f"Missing: {X_RAW_PATH}")

    meta_path = META_AUG_PATH if os.path.isfile(META_AUG_PATH) else META_STD_PATH
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing: {meta_path}")

    print(f"[meta] Using: {meta_path}")

    X = np.load(X_RAW_PATH).astype(np.float32)  # (N, T, C)
    meta = pd.read_csv(meta_path)

    if X.ndim != 3:
        raise ValueError(f"X_raw must be (N,T,C). Got {X.shape}")
    if len(meta) != X.shape[0]:
        raise ValueError(f"Meta rows ({len(meta)}) must match windows N ({X.shape[0]})")
    if "label" not in meta.columns:
        raise KeyError(f"Meta file must include a 'label' column. Columns: {list(meta.columns)}")

    # Optional: drop sensor fault windows (common for 2-class ML baselines)
    if drop_sensor_fault:
        keep = (meta["label"].astype(str) != "Sensor Fault").to_numpy()
        X = X[keep]
        meta = meta.loc[keep].reset_index(drop=True)
        print(f"[filter] Dropped Sensor Fault windows. Remaining N={len(meta)}")

    # Encode labels
    y = meta["label"].astype(str).map(LABEL_MAP)
    if y.isna().any():
        bad = meta.loc[y.isna(), "label"].value_counts()
        raise ValueError(f"Unknown labels found in meta: {bad.to_dict()}")
    y = y.to_numpy(dtype=np.int64)

    N, T, Cn = X.shape

    # Feature names
    base = ["mean", "std", "var", "min", "max", "ptp", "rms", "crest", "skew", "kurt_excess", "snr_db"]
    freq = ["fft_total", "fft_centroid", "fft_roll85", "fft_band1", "fft_band2", "fft_band3", "fft_band4", "fft_band5"]
    per_ch = base + (freq if include_freq else [])

    feat_names: list[str] = []
    for ci in range(Cn):
        for n in per_ch:
            feat_names.append(f"ch{ci}_{n}")

    X_feat = np.zeros((N, len(feat_names)), dtype=np.float32)

    for i in range(N):
        if (i + 1) % 500 == 0 or (i + 1) == N:
            print(f"[featurize] {i+1}/{N}")

        row: list[float] = []
        for ci in range(Cn):
            row += featurize_channel(X[i, :, ci], include_freq=include_freq)
        X_feat[i, :] = np.asarray(row, dtype=np.float32)

    # Save outputs
    np.save(X_FEAT_PATH, X_feat)
    np.save(Y_PATH, y)

    # Save meta used (after optional filtering) to keep alignment with X_feat/y
    meta.to_csv(META_USED_PATH, index=False, encoding="utf-8")

    with open(FEAT_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_samples": int(N),
                "n_features": int(X_feat.shape[1]),
                "n_channels": int(Cn),
                "seq_len": int(T),
                "label_map": LABEL_MAP,
                "meta_source": meta_path,
                "drop_sensor_fault": bool(drop_sensor_fault),
                "include_freq": bool(include_freq),
                "feature_names": feat_names,
            },
            f,
            indent=2,
        )

    print(f"Saved: {X_FEAT_PATH}  shape={X_feat.shape}")
    print(f"Saved: {Y_PATH}       shape={y.shape}")
    print(f"Saved: {META_USED_PATH}")
    print(f"Saved: {FEAT_NAMES_PATH} n_features={len(feat_names)}")


if __name__ == "__main__":
    # Set drop_sensor_fault=True if you want only Normal vs Structural Fault for baseline ML.
    main(include_freq=True, drop_sensor_fault=False)
