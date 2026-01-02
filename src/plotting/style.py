# src/plotting/style.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt


PathLike = Union[str, Path]


@dataclass(frozen=True)
class PlotSettings:
    """
    Centralized figure settings for consistent, print-ready plots.
    """
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 300
    font_size: int = 12
    label_size: int = 12
    tick_size: int = 11
    legend_size: int = 11

    line_width: float = 1.2          # thin (journal-friendly)
    axis_line_width: float = 1.0     # visible spines/borders

    grid: bool = False               # grid off by default for journals
    style: Optional[str] = None      # e.g. "fivethirtyeight" if you want it


def set_style(style: Optional[str] = None) -> None:
    """
    Apply an optional Matplotlib style.
    Use this sparingly in publication figures (defaults are already print-ready).

    Example:
        set_style("fivethirtyeight")
    """
    if style:
        plt.style.use(style)


def apply_plot_defaults(cfg: PlotSettings = PlotSettings()) -> None:
    """
    Apply rcParams for consistent publication-ready output:
    - thin line widths
    - visible axis spines (borders)
    - clean fonts and tick sizes
    """
    mpl.rcParams.update({
        # Fonts
        "font.size": cfg.font_size,
        "axes.labelsize": cfg.label_size,
        "xtick.labelsize": cfg.tick_size,
        "ytick.labelsize": cfg.tick_size,
        "legend.fontsize": cfg.legend_size,

        # Lines
        "lines.linewidth": cfg.line_width,
        "axes.linewidth": cfg.axis_line_width,

        # Spines visible
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": cfg.axis_line_width,
        "ytick.major.width": cfg.axis_line_width,
        "xtick.minor.width": cfg.axis_line_width * 0.8,
        "ytick.minor.width": cfg.axis_line_width * 0.8,

        # Grid
        "axes.grid": cfg.grid,

        # Output
        "savefig.dpi": cfg.dpi,
        "figure.dpi": cfg.dpi,
    })

    # Optional external style
    set_style(cfg.style)


def save_figure(
    fig: plt.Figure,
    outpath_no_ext: PathLike,
    save_pdf: bool = True,
    save_svg: bool = True,
    tight: bool = True,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Save a figure as PDF and/or SVG using a single base path.

    Args:
        fig: matplotlib Figure
        outpath_no_ext: path without extension, e.g. "outputs/figures/vae/mse_hist"
        save_pdf: if True, save PDF
        save_svg: if True, save SVG
        tight: if True, use bbox_inches="tight"

    Returns:
        (pdf_path, svg_path)
    """
    outpath_no_ext = Path(outpath_no_ext)
    outpath_no_ext.parent.mkdir(parents=True, exist_ok=True)

    bbox = "tight" if tight else None
    pdf_path = None
    svg_path = None

    if save_pdf:
        pdf_path = outpath_no_ext.with_suffix(".pdf")
        fig.savefig(pdf_path, format="pdf", bbox_inches=bbox)

    if save_svg:
        svg_path = outpath_no_ext.with_suffix(".svg")
        fig.savefig(svg_path, format="svg", bbox_inches=bbox)

    return pdf_path, svg_path
