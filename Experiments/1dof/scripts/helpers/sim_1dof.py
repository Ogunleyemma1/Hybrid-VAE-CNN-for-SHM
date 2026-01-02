from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np


@dataclass(frozen=True)
class SimConfig:
    """
    Linear 1DOF oscillator used for VAE interpretability experiments.

    Notes
    -----
    This module is scoped to experiments/1dof to keep src/ clean.
    """
    m: float = 1.0
    c: float = 0.02
    k: float = 25.0
    dt: float = 0.01
    t_total: float = 30.0
    forcing_amp: float = 1.0
    forcing_freq_hz: float = 1.0


def newmark_beta_1dof(
    sim: SimConfig,
    force: np.ndarray,
    beta: float = 0.25,
    gamma: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Newmark-beta integration for a linear 1DOF system:

        m x¨ + c x˙ + k x = f(t)

    Returns
    -------
    t : (N,) time vector
    x : (N,) displacement
    v : (N,) velocity
    a : (N,) acceleration
    """
    n = int(force.shape[0])
    t = np.arange(n, dtype=np.float64) * sim.dt

    x = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)
    a = np.zeros(n, dtype=np.float64)

    # initial acceleration from equilibrium
    a[0] = (force[0] - sim.c * v[0] - sim.k * x[0]) / sim.m

    k_eff = sim.k + gamma * sim.c / (beta * sim.dt) + sim.m / (beta * sim.dt**2)

    for i in range(1, n):
        p_eff = (
            force[i]
            + sim.m
            * (
                x[i - 1] / (beta * sim.dt**2)
                + v[i - 1] / (beta * sim.dt)
                + (1.0 / (2.0 * beta) - 1.0) * a[i - 1]
            )
            + sim.c
            * (
                gamma * x[i - 1] / (beta * sim.dt)
                + (gamma / beta - 1.0) * v[i - 1]
                + sim.dt * (gamma / (2.0 * beta) - 1.0) * a[i - 1]
            )
        )

        x[i] = p_eff / k_eff
        a[i] = (
            (x[i] - x[i - 1]) / (beta * sim.dt**2)
            - v[i - 1] / (beta * sim.dt)
            - (1.0 / (2.0 * beta) - 1.0) * a[i - 1]
        )
        v[i] = v[i - 1] + sim.dt * ((1.0 - gamma) * a[i - 1] + gamma * a[i])

    return t, x, v, a


def forcing_sine(sim: SimConfig, n_steps: int, amp: float | None = None, freq_hz: float | None = None) -> np.ndarray:
    """Sine forcing used for both seen and unseen experiments."""
    amp = sim.forcing_amp if amp is None else float(amp)
    freq_hz = sim.forcing_freq_hz if freq_hz is None else float(freq_hz)
    t = np.arange(n_steps, dtype=np.float64) * sim.dt
    return amp * np.sin(2.0 * np.pi * freq_hz * t)


def make_seen_signal(sim: SimConfig, n_steps: int, label: str, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Seen families (used during training/val/test):
    - normal
    - drifted
    - noisy
    """
    f = forcing_sine(sim, n_steps)
    t, x, v, a = newmark_beta_1dof(sim, f)

    if label == "normal":
        pass
    elif label == "drifted":
        x = x + 1e-4 * t
    elif label == "noisy":
        x = x + rng.normal(0.0, 2e-4, size=x.shape)
    else:
        raise ValueError(f"Unknown seen label: {label}")

    return {
        "t": t.astype(np.float32),
        "x": x.astype(np.float32),
        "v": v.astype(np.float32),
        "a": a.astype(np.float32),
    }


def make_unseen_signal(sim: SimConfig, n_steps: int, variant: str, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Unseen variants (not used during training).

    Adjust magnitudes/variants here to exactly match your thesis/manuscript.
    Current defaults are conservative and reviewer-friendly.
    """
    t = np.arange(n_steps, dtype=np.float64) * sim.dt

    if variant == "amp_shifted":
        f = forcing_sine(sim, n_steps, amp=1.5 * sim.forcing_amp)
    elif variant == "freq_shifted":
        f = forcing_sine(sim, n_steps, freq_hz=1.4 * sim.forcing_freq_hz)
    else:
        f = forcing_sine(sim, n_steps)

    t, x, v, a = newmark_beta_1dof(sim, f)

    if variant == "noise_heavy":
        x = x + rng.normal(0.0, 6e-4, size=x.shape)
    elif variant == "drift_heavy":
        x = x + 5e-4 * t
    elif variant in ("amp_shifted", "freq_shifted"):
        pass
    else:
        raise ValueError(f"Unknown unseen variant: {variant}")

    return {
        "t": t.astype(np.float32),
        "x": x.astype(np.float32),
        "v": v.astype(np.float32),
        "a": a.astype(np.float32),
    }
