from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NewmarkParams:
    beta: float = 1 / 4
    gamma: float = 1 / 2


@dataclass(frozen=True)
class SDOFParams:
    m: float = 100.0
    k: float = 1000.0
    c: float = 0.0
    x0: float = 0.01
    v0: float = 0.0
    t_total: float = 30.0
    dt: float = 0.01


def simulate_free_vibration(
    p: SDOFParams,
    nm: NewmarkParams = NewmarkParams(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Free vibration with Newmark-beta. No external forcing.
    Returns (t, x, v, a).
    """
    t = np.arange(0.0, p.t_total + p.dt, p.dt)
    n = len(t)

    x = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)
    a = np.zeros(n, dtype=float)

    x[0] = p.x0
    v[0] = p.v0
    a[0] = (-p.k * x[0] - p.c * v[0]) / p.m

    beta, gamma = nm.beta, nm.gamma
    k_eff = p.m / (beta * p.dt**2) + gamma * p.c / (beta * p.dt) + p.k

    for i in range(1, n):
        b = (
            p.m
            * (
                (1 / (beta * p.dt**2)) * x[i - 1]
                + (1 / (beta * p.dt)) * v[i - 1]
                + ((1 / (2 * beta)) - 1) * a[i - 1]
            )
            - p.c * (v[i - 1] + (1 - gamma) * p.dt * a[i - 1])
        )
        x[i] = b / k_eff
        a[i] = (
            (1 / (beta * p.dt**2)) * (x[i] - x[i - 1])
            - (1 / (beta * p.dt)) * v[i - 1]
            - ((1 / (2 * beta)) - 1) * a[i - 1]
        )
        v[i] = v[i - 1] + p.dt * ((1 - gamma) * a[i - 1] + gamma * a[i])

    return t, x, v, a


def make_clean_variants(
    t: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    drift_rate: float = 0.001,
    amp_scale: float = 1.5,
    lowfreq_factor: float = 0.6,
) -> dict[str, np.ndarray]:
    """
    Creates four variants per channel:
      - original
      - drifted (linear bias over time)
      - amplitude scaled (amplitude upscaled)
      - lowfreq (frequency-reduced using time-stretching)

    Parameters
    ----------
    drift_rate:
        Linear drift slope added to each signal.
    amp_scale:
        Multiplicative factor applied to each signal (amplitude upscaling when > 1).
    lowfreq_factor:
        Time-scaling factor alpha in x_low(t) = x(alpha * t).
        alpha < 1 reduces frequency (f_low = alpha * f_original).
        Recommended range: 0.4 to 0.8.
    """
    # Drift (linear bias)
    x_drift = x + drift_rate * t
    v_drift = v + drift_rate * t
    a_drift = a + drift_rate * t

    # Amplitude upscaling (keep key name for compatibility)
    x_amp = x * amp_scale
    v_amp = v * amp_scale
    a_amp = a * amp_scale

    # Low-frequency: frequency reduction via time-stretching (not smoothing)
    # x_low(t) = x(alpha t), alpha < 1 -> fewer cycles over same time window
    alpha = float(lowfreq_factor)
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"lowfreq_factor must be in (0, 1], got {alpha}")

    t_scaled = t * alpha
    x_low = np.interp(t_scaled, t, x)
    v_low = np.interp(t_scaled, t, v)
    a_low = np.interp(t_scaled, t, a)

    return {
        "x_original": x,
        "x_drift": x_drift,
        "x_amplitude_scaled": x_amp,
        "x_lowfreq": x_low,
        "v_original": v,
        "v_drift": v_drift,
        "v_amplitude_scaled": v_amp,
        "v_lowfreq": v_low,
        "a_original": a,
        "a_drift": a_drift,
        "a_amplitude_scaled": a_amp,
        "a_lowfreq": a_low,
    }


def _triangle_wave(t: np.ndarray, f: float) -> np.ndarray:
    # Triangle wave in [-1,1] without scipy
    return (2.0 / np.pi) * np.arcsin(np.sin(2.0 * np.pi * f * t))


def _square_wave(t: np.ndarray, f: float) -> np.ndarray:
    # Square wave in {-1,1} without scipy
    return np.sign(np.sin(2.0 * np.pi * f * t))


def make_unseen_variants(
    t: np.ndarray,
    amplitude: float = 0.01,
    base_freq_hz: float = 0.33,
) -> dict[str, np.ndarray]:
    """
    Unseen signals (still 1DOF-like) with same amplitude scale:
      - original sinusoid
      - envelope-modulated sinusoid
      - triangle wave
      - square wave
    Returns x,v,a for each.
    """
    w = 2.0 * np.pi * base_freq_hz

    x_ori = amplitude * np.sin(w * t)

    # Envelope-modulated sinusoid
    env = 0.5 * (1.0 + np.sin(0.2 * w * t))  # slow modulation in [0,1]
    x_env = amplitude * env * np.sin(w * t)

    x_tri = amplitude * _triangle_wave(t, base_freq_hz)
    x_sqr = amplitude * _square_wave(t, base_freq_hz)

    # Numerical derivatives for v and a
    dt = t[1] - t[0]

    def deriv(y: np.ndarray) -> np.ndarray:
        return np.gradient(y, dt)

    v_ori, a_ori = deriv(x_ori), deriv(deriv(x_ori))
    v_env, a_env = deriv(x_env), deriv(deriv(x_env))
    v_tri, a_tri = deriv(x_tri), deriv(deriv(x_tri))
    v_sqr, a_sqr = deriv(x_sqr), deriv(deriv(x_sqr))

    return {
        "x_original": x_ori,
        "x_envelope": x_env,
        "x_triangle": x_tri,
        "x_square": x_sqr,
        "v_original": v_ori,
        "v_envelope": v_env,
        "v_triangle": v_tri,
        "v_square": v_sqr,
        "a_original": a_ori,
        "a_envelope": a_env,
        "a_triangle": a_tri,
        "a_square": a_sqr,
    }
