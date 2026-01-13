from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Ensure local imports work regardless of current working directory
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from signals_1dof import make_unseen_variants


def configure_axis(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=16, labelpad=16)
    ax.tick_params(axis="both", which="major", labelsize=14)

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


def plot_stacked_variants(
    time: pd.Series,
    df: pd.DataFrame,
    prefix: str,
    y_label_short: str,
    out_dir: Path,
    file_stem: str,
) -> None:
    # Unseen variants defined by make_unseen_variants()
    # Keep fixed colors (blue/orange/green/red)
    variant_specs = [
        ("Original", f"{prefix}original", "C0"),
        ("Envelope", f"{prefix}envelope", "C1"),
        ("Triangle", f"{prefix}triangle", "C2"),
        ("Square", f"{prefix}square", "C3"),
    ]

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 8), sharex=True)

    legend_handles = []
    legend_labels = []

    for ax, (label, col, color) in zip(axes, variant_specs):
        (line,) = ax.plot(time, df[col], linewidth=1.5, color=color)
        configure_axis(ax, ylabel=y_label_short)
        legend_handles.append(line)
        legend_labels.append(label)

    axes[-1].set_xlabel("Time (s)", fontsize=16)

    # Single legend at the bottom
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        fontsize=14,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    # Make room for legend
    fig.tight_layout(rect=(0.02, 0.06, 1.0, 1.0))

    save_figure(fig, out_dir, file_stem=file_stem)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]  # 1_DOF

    data_dir = root / "Data" / "raw"
    fig_dir = root / "Output" / "figures" / "variants_unseen"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    t_total = 30.0
    dt = 0.01
    t = np.arange(0.0, t_total + dt, dt)

    variants = make_unseen_variants(t, amplitude=0.01, base_freq_hz=0.33)
    df = pd.DataFrame({"time": t, **variants})

    csv_path = data_dir / "1dof_unseen_variants.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] wrote {csv_path}")

    plot_stacked_variants(
        time=df["time"],
        df=df,
        prefix="x_",
        y_label_short="x (m)",
        out_dir=fig_dir,
        file_stem="unseen_variants_displacement_stacked",
    )

    plot_stacked_variants(
        time=df["time"],
        df=df,
        prefix="v_",
        y_label_short="v (m/s)",
        out_dir=fig_dir,
        file_stem="unseen_variants_velocity_stacked",
    )

    plot_stacked_variants(
        time=df["time"],
        df=df,
        prefix="a_",
        y_label_short=r"a (m/s$^2$)",
        out_dir=fig_dir,
        file_stem="unseen_variants_acceleration_stacked",
    )


if __name__ == "__main__":
    main()
