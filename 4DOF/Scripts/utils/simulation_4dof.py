# Scripts/utils/simulation_4dof.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch


@dataclass
class SystemConfig:
    mass: List[float]
    stiffness: List[float]
    damping_ratio: float
    beta: float
    gamma: float
    num_dofs: int
    dt: float
    T_total: float


def init_force(T_total: float, dt: float, num_dofs: int, rms: float, seed: int) -> torch.Tensor:
    """
    Smoothed Gaussian excitation. Returns tensor shape [steps, num_dofs].
    """
    np.random.seed(seed)
    steps = int(T_total / dt) + 1
    base = np.random.randn(steps, num_dofs) * rms

    # smooth each DOF force
    window = int(0.5 / dt)
    if window < 1:
        window = 1
    for j in range(num_dofs):
        s = pd.Series(base[:, j])
        base[:, j] = s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()

    return torch.tensor(base, dtype=torch.float32)


def compute_matrices(m: np.ndarray, k: np.ndarray, zeta: float, num_dofs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chain-like stiffness matrix and Rayleigh damping fitted from first two modes.
    """
    M = np.diag(m)
    K = np.zeros((num_dofs, num_dofs), dtype=float)

    for i in range(num_dofs):
        if i == 0:
            K[i, i] = k[i] + k[i + 1]
            K[i, i + 1] = -k[i + 1]
        elif i == num_dofs - 1:
            K[i, i] = k[i]
            K[i, i - 1] = -k[i]
        else:
            K[i, i] = k[i] + k[i + 1]
            K[i, i - 1] = -k[i]
            K[i, i + 1] = -k[i + 1]

    eigvals = np.linalg.eigvals(np.linalg.inv(M) @ K)
    omegas = np.sqrt(np.sort(eigvals.real[eigvals.real > 0]))

    if len(omegas) < 2:
        alpha, beta = 0.1, 0.001
    else:
        o1, o2 = float(omegas[0]), float(omegas[1])
        A = np.array([[1.0 / (2 * o1), o1 / 2], [1.0 / (2 * o2), o2 / 2]], dtype=float)
        z = np.array([zeta, zeta], dtype=float)
        alpha, beta = np.linalg.solve(A, z)
        alpha = max(alpha, 0.0)
        beta = max(beta, 1e-4)

    C = alpha * M + beta * K
    return M, C, K


def run_simulation(cfg: SystemConfig, force: torch.Tensor) -> pd.DataFrame:
    """
    Newmark-beta integration for nd DOFs.
    Returns DataFrame columns: x1..xN, v1..vN, a1..aN
    """
    nd = cfg.num_dofs
    dt = cfg.dt
    T_total = cfg.T_total
    beta = cfg.beta
    gamma = cfg.gamma

    steps = int(T_total / dt) + 1
    assert force.shape[0] == steps, "Force length must match steps."

    t = np.linspace(0.0, T_total, steps)

    m = np.array(cfg.mass, dtype=float)
    k = np.array(cfg.stiffness, dtype=float)

    M, C, K = compute_matrices(m, k, cfg.damping_ratio, nd)
    M_inv = np.linalg.inv(M)

    x = np.zeros((nd, steps), dtype=float)
    v = np.zeros((nd, steps), dtype=float)
    a = np.zeros((nd, steps), dtype=float)

    F0 = force[0].numpy()
    a[:, 0] = np.nan_to_num(M_inv @ (F0 - C @ v[:, 0] - K @ x[:, 0]), nan=0.0)

    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = (1.0 / (2 * beta)) - 1.0
    a4 = (gamma / beta) - 1.0
    a5 = (dt / 2.0) * ((gamma / beta) - 2.0)

    K_eff = a0 * M + a1 * C + K
    K_eff_inv = np.linalg.inv(K_eff)

    for i in range(1, steps):
        Ft = force[i].numpy()

        x_prev = x[:, i - 1]
        v_prev = v[:, i - 1]
        a_prev = a[:, i - 1]

        P_eff = Ft + (M @ (a0 * x_prev + a2 * v_prev + a3 * a_prev)) + (C @ (a1 * x_prev + a4 * v_prev + a5 * a_prev))
        x_curr = K_eff_inv @ P_eff
        a_curr = a0 * (x_curr - x_prev) - a2 * v_prev - a3 * a_prev
        v_curr = v_prev + dt * ((1.0 - gamma) * a_prev + gamma * a_curr)

        x[:, i] = np.clip(x_curr, -1e5, 1e5)
        v[:, i] = np.clip(v_curr, -1e5, 1e5)
        a[:, i] = np.clip(a_curr, -1e5, 1e5)

    data = np.vstack((x, v, a)).T
    cols = [f"x{j+1}" for j in range(nd)] + [f"v{j+1}" for j in range(nd)] + [f"a{j+1}" for j in range(nd)]
    return pd.DataFrame(data, columns=cols)


def default_system_config() -> SystemConfig:
    BASE_MASS = 50.0
    BASE_STIFFNESS = 200000.0
    return SystemConfig(
        mass=[BASE_MASS * 1.2, BASE_MASS, BASE_MASS, BASE_MASS * 0.8],
        stiffness=[BASE_STIFFNESS * 1.5, BASE_STIFFNESS * 1.2, BASE_STIFFNESS, BASE_STIFFNESS * 0.8],
        damping_ratio=0.02,
        beta=0.25,
        gamma=0.5,
        num_dofs=4,
        dt=0.01,
        T_total=10.0,
    )
