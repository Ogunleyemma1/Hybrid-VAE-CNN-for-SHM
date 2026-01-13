# Scripts/01_generate_normal_runs.py
from __future__ import annotations

from pathlib import Path
import sys

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Ensure local imports work regardless of current working directory
_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent  # 4DOF/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Scripts.utils.simulation_4dof import default_system_config, init_force, run_simulation
# -----------------------------------------------------------------------------


N_NORMAL_RUNS = 10
OUT_DIR = Path("Data/raw/normal")
FORCE_RMS = 50.0
BASE_SEED = 2025

# Plot settings
PLOT_REP_SEED = BASE_SEED
FIG_DIR = Path("Output/figures/normal_runs")

# Consistent per-DOF colors (publication-friendly and stable across plots)
DOF_COLORS = {
    1: "C0",
    2: "C1",
    3: "C2",
    4: "C3",
}


def configure_axis(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=16, labelpad=14)
    ax.tick_params(axis="both", which="major", labelsize=13)

    # No internal gridlines
    ax.grid(False)

    # Border box (spines)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    # Transparent axes background
    ax.set_facecolor("none")


def save_figure(fig, out_dir: Path, file_stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / f"{file_stem}.pdf"
    png_path = out_dir / f"{file_stem}.png"
    svg_path = out_dir / f"{file_stem}.svg"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", transparent=True)
    fig.savefig(png_path, format="png", bbox_inches="tight", transparent=True, dpi=300)
    fig.savefig(svg_path, format="svg", bbox_inches="tight", transparent=True)

    print(f"[OK] saved: {pdf_path.name}, {png_path.name}, {svg_path.name}")


def plot_stacked_displacement(df: pd.DataFrame, dt: float, out_dir: Path, file_stem: str) -> None:
    """
    Four stacked axes: one per DOF displacement, with consistent colors.
    Uses time axis in seconds.
    """
    # reconstruct time (seconds)
    t = np.arange(len(df), dtype=float) * float(dt)

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 8), sharex=True)

    handles, labels = [], []
    for ax, j in zip(axes, range(1, 5)):
        col = f"x{j}"
        color = DOF_COLORS[j]
        (ln,) = ax.plot(t, df[col].to_numpy(), linewidth=1.5, color=color)
        configure_axis(ax, ylabel=f"x{j} (m)")
        handles.append(ln)
        labels.append(f"DOF {j}")

    axes[-1].set_xlabel("Time (s)", fontsize=16)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=13,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0.02, 0.06, 1.0, 1.0))
    save_figure(fig, out_dir, file_stem=file_stem)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = default_system_config()
    rep_df: pd.DataFrame | None = None
    rep_dt: float | None = None

    for i in range(N_NORMAL_RUNS):
        seed = BASE_SEED + i

        # jitter per run (reviewer-friendly: mild variability around baseline)
        cfg = default_system_config()
        cfg.mass = (np.array(base_cfg.mass) * np.random.uniform(0.98, 1.02, size=len(base_cfg.mass))).tolist()
        cfg.stiffness = (np.array(base_cfg.stiffness) * np.random.uniform(0.98, 1.02, size=len(base_cfg.stiffness))).tolist()
        cfg.damping_ratio = float(np.random.uniform(0.015, 0.025))

        force = init_force(cfg.T_total, cfg.dt, cfg.num_dofs, FORCE_RMS, seed)
        df = run_simulation(cfg, force)

        out_csv = OUT_DIR / f"normal_seed{seed}.csv"
        df.to_csv(out_csv, index=False)
        print(f"[OK] normal run saved: {out_csv}")

        if seed == PLOT_REP_SEED:
            rep_df = df
            rep_dt = float(cfg.dt)

    # --- stacked figure from representative run (no overlay) ---
    if rep_df is not None and rep_dt is not None:
        plot_stacked_displacement(
            rep_df,
            dt=rep_dt,
            out_dir=FIG_DIR,
            file_stem=f"normal_run_seed{PLOT_REP_SEED}_displacement_stacked",
        )
    else:
        print(f"[WARN] Representative seed {PLOT_REP_SEED} not generated; no plots produced.")

    print("[DONE] Normal run generation + stacked plot complete.")


if __name__ == "__main__":
    main()
