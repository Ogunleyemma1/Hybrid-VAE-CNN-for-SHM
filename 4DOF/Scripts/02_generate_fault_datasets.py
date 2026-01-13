# Scripts/02_generate_fault_datasets.py
from __future__ import annotations

from pathlib import Path
import sys
import re
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Scripts.utils.simulation_4dof import default_system_config, init_force, run_simulation

OUT_FAULTS_DIR = Path("Data/raw/faults")
STRUCT_DIR = OUT_FAULTS_DIR / "structural_fault"
SENSOR_DIR = OUT_FAULTS_DIR / "sensor_fault"

FIG_DIR = Path("Output/figures/faults")

FORCE_RMS = 200.0
FORCE_SEED = 42

DOF_COLORS = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}

NORMAL_ALPHA = 0.70
NORMAL_LW = 1.9
FAULT_COLOR = "0.25"
FAULT_LW = 2.6
FAULT_LS = "--"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_name(text: str, maxlen: int = 60) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "", text)[:maxlen]


def configure_axis(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=16, labelpad=14)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.set_facecolor("none")


def save_figure(fig, out_dir: Path, file_stem: str) -> None:
    ensure_dir(out_dir)
    for ext, dpi in (("pdf", None), ("png", 300), ("svg", None)):
        fig.savefig(out_dir / f"{file_stem}.{ext}", bbox_inches="tight", transparent=True, dpi=dpi)
    print(f"[OK] saved: {file_stem}.(pdf/png/svg)")


def _legend_structural_fault() -> Tuple[list, list]:
    normal_handles = [
        Line2D([0], [0], color=DOF_COLORS[j], lw=2.2, linestyle="-", alpha=1.0)
        for j in range(1, 5)
    ]
    normal_labels = [f"Normal DOF {j}" for j in range(1, 5)]
    fault_handle = Line2D([0], [0], color=FAULT_COLOR, lw=FAULT_LW, linestyle=FAULT_LS)
    return normal_handles + [fault_handle], normal_labels + ["Structural Fault"]


def _legend_sensor_fault(corrupt_dof: int, fault_name: str) -> Tuple[list, list]:
    normal_handles = [
        Line2D([0], [0], color=DOF_COLORS[j], lw=2.2, linestyle="-", alpha=1.0)
        for j in range(1, 5)
    ]
    normal_labels = [f"Normal DOF {j}" for j in range(1, 5)]
    fault_handle = Line2D([0], [0], color=FAULT_COLOR, lw=FAULT_LW, linestyle=FAULT_LS, alpha=1.0)
    fault_label = f"Sensor Fault ({fault_name}) DOF {corrupt_dof}"
    return normal_handles + [fault_handle], normal_labels + [fault_label]


def plot_structural_fault_stacked(normal_df: pd.DataFrame, fault_df: pd.DataFrame, dt: float, out_dir: Path, file_stem: str) -> None:
    n = min(len(normal_df), len(fault_df))
    t = np.arange(n, dtype=float) * float(dt)
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 8), sharex=True)

    for ax, j in zip(axes, range(1, 5)):
        col = f"x{j}"
        ax.plot(t, normal_df[col].to_numpy()[:n], linewidth=NORMAL_LW, color=DOF_COLORS[j], alpha=NORMAL_ALPHA)
        ax.plot(t, fault_df[col].to_numpy()[:n], linewidth=FAULT_LW, color=FAULT_COLOR, linestyle=FAULT_LS)
        configure_axis(ax, ylabel=f"x{j} (m)")

    axes[-1].set_xlabel("Time (s)", fontsize=16)
    handles, labels = _legend_structural_fault()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0.02, 0.06, 1.0, 1.0))
    save_figure(fig, out_dir, file_stem=file_stem)
    plt.close(fig)


def plot_sensor_fault_stacked(normal_df: pd.DataFrame, fault_df: pd.DataFrame, dt: float, out_dir: Path, file_stem: str, corrupt_dof: int, fault_name: str) -> None:
    n = min(len(normal_df), len(fault_df))
    t = np.arange(n, dtype=float) * float(dt)
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 8), sharex=True)

    for ax, j in zip(axes, range(1, 5)):
        col = f"x{j}"
        ax.plot(t, normal_df[col].to_numpy()[:n], linewidth=NORMAL_LW, color=DOF_COLORS[j], alpha=NORMAL_ALPHA)

        if j == corrupt_dof:
            ax.plot(t, fault_df[col].to_numpy()[:n], linewidth=FAULT_LW, color=FAULT_COLOR, linestyle=FAULT_LS)

        configure_axis(ax, ylabel=f"x{j} (m)")

    axes[-1].set_xlabel("Time (s)", fontsize=16)
    handles, labels = _legend_sensor_fault(corrupt_dof=corrupt_dof, fault_name=fault_name)
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0.02, 0.06, 1.0, 1.0))
    save_figure(fig, out_dir, file_stem=file_stem)
    plt.close(fig)


# ---------------- Sensor fault injections ----------------
def inject_noise(x: np.ndarray, magnitude: float) -> np.ndarray:
    return x + np.random.normal(0.0, magnitude, size=len(x))


def inject_spikes(x: np.ndarray, magnitude: float, freq: float = 0.01) -> np.ndarray:
    n = len(x)
    spikes = np.zeros_like(x)
    idx = np.random.choice(n, int(n * freq), replace=False)
    spikes[idx] = np.random.normal(magnitude, magnitude / 4.0, size=len(idx))
    return x + spikes


def inject_drift(x: np.ndarray, magnitude: float) -> np.ndarray:
    return x + np.linspace(0.0, magnitude, len(x))


def inject_bias(x: np.ndarray, magnitude: float) -> np.ndarray:
    return x + magnitude


def dof_triplet(dof: int) -> List[str]:
    return [f"x{dof}", f"v{dof}", f"a{dof}"]


def generate_structural_faults(normal_df: pd.DataFrame, base_cfg, force, dt: float) -> None:
    reductions = [0.9, 0.8, 0.70, 0.60]  # 10%, 20%, 30%, 40% reduction (match your log)
    ensure_dir(STRUCT_DIR)

    for perc in reductions:
        label = f"stiff_red_{int((1.0 - perc) * 100)}pct"
        case_dir = ensure_dir(STRUCT_DIR / label)

        cfg = default_system_config()
        cfg.mass = list(base_cfg.mass)
        cfg.stiffness = (np.array(base_cfg.stiffness) * perc).tolist()
        cfg.damping_ratio = base_cfg.damping_ratio

        fault_df = run_simulation(cfg, force)

        csv_path = case_dir / f"{label}.csv"
        fault_df.to_csv(csv_path, index=False)
        print(f"[OK] structural fault saved: {label}")

        fig_out = ensure_dir(FIG_DIR / "structural_fault" / label)
        plot_structural_fault_stacked(
            normal_df=normal_df,
            fault_df=fault_df,
            dt=dt,
            out_dir=fig_out,
            file_stem=f"{label}_normal_vs_structural_fault_displacement_stacked",
        )


def generate_sensor_faults(normal_df: pd.DataFrame, dt: float) -> None:
    ensure_dir(SENSOR_DIR)

    # exactly 4 datasets, each corrupting one DOF triplet only
    faults: Dict[str, Dict] = {
        "noise_x4":   {"func": inject_noise,  "dof": 4, "rel_mag": 0.50},
        "spikes_x1":  {"func": inject_spikes, "dof": 1, "rel_mag": 5.00},
        "drift_x2":   {"func": inject_drift,  "dof": 2, "rel_mag": 10.0},
        "bias_x3":    {"func": inject_bias,   "dof": 3, "rel_mag": 2.00},
    }

    for name, cfg in faults.items():
        case_dir = ensure_dir(SENSOR_DIR / safe_name(name))
        fault_df = normal_df.copy()

        dof = int(cfg["dof"])
        cols = dof_triplet(dof)  # [xk, vk, ak]
        func = cfg["func"]
        rel = float(cfg["rel_mag"])

        for c in cols:
            std = float(normal_df[c].std())
            mag = (std if std > 0 else 1.0) * rel
            fault_df[c] = func(normal_df[c].to_numpy(), magnitude=mag)

        csv_path = case_dir / f"{name}.csv"
        fault_df.to_csv(csv_path, index=False)
        print(f"[OK] sensor fault saved: {name} (target=x{dof} -> x{dof},v{dof},a{dof})")

        # plot only displacement overlay (as before) so the figure stays clean
        fault_kind = name.split("_")[0]
        fig_out = ensure_dir(FIG_DIR / "sensor_fault" / safe_name(name))
        plot_sensor_fault_stacked(
            normal_df=normal_df,
            fault_df=fault_df,
            dt=dt,
            out_dir=fig_out,
            file_stem=f"{safe_name(name)}_normal_vs_sensor_fault_displacement_stacked",
            corrupt_dof=dof,
            fault_name=fault_kind,
        )


def main() -> None:
    base_cfg = default_system_config()
    dt = float(base_cfg.dt)

    force = init_force(base_cfg.T_total, base_cfg.dt, base_cfg.num_dofs, FORCE_RMS, FORCE_SEED)
    normal_df = run_simulation(base_cfg, force)

    generate_structural_faults(normal_df, base_cfg, force, dt=dt)
    generate_sensor_faults(normal_df, dt=dt)

    print("[SUCCESS] Fault datasets + comparison plots generated.")


if __name__ == "__main__":
    main()
