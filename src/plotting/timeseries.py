# src/plotting/timeseries.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from .style import PlotSettings, apply_plot_defaults, save_figure


PathLike = Union[str, Path]


def plot_series(
    y: Sequence[float],
    x: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
    xlabel: str = "Index",
    ylabel: str = "Value",
    label: Optional[str] = None,
    outpath_no_ext: Optional[PathLike] = None,
    cfg: PlotSettings = PlotSettings(),
    show: bool = False,
) -> plt.Figure:
    """
    Plot a single series with publication-ready defaults.

    Args:
        y: series values
        x: optional x-values
        label: legend label
        outpath_no_ext: if provided, saves PDF+SVG to this base path
    """
    apply_plot_defaults(cfg)

    y = np.asarray(y, dtype=float)
    x = np.arange(len(y)) if x is None else np.asarray(x, dtype=float)

    fig = plt.figure(figsize=cfg.figsize)
    ax = fig.gca()

    ax.plot(x, y, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if label:
        ax.legend(loc="best")

    fig.tight_layout()

    if outpath_no_ext is not None:
        save_figure(fig, outpath_no_ext)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_series_with_band(
    x: Sequence[float],
    center: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    title: Optional[str] = None,
    xlabel: str = "Index",
    ylabel: str = "Value",
    line_label: str = "Median",
    band_label: str = "IQR",
    outpath_no_ext: Optional[PathLike] = None,
    cfg: PlotSettings = PlotSettings(),
    show: bool = False,
) -> plt.Figure:
    """
    Plot a central tendency curve with an uncertainty band (e.g., median + IQR).

    This is the clean functional equivalent of your template plotting code,
    but with print-ready thin lines and visible spines.

    Args:
        x: (N,) x-axis values
        center: (N,) median/mean curve
        lower/upper: (N,) band bounds (e.g., Q1 and Q3)
    """
    apply_plot_defaults(cfg)

    x = np.asarray(x, dtype=float)
    center = np.asarray(center, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    fig = plt.figure(figsize=cfg.figsize)
    ax = fig.gca()

    ax.plot(x, center, label=line_label)
    ax.fill_between(x, lower, upper, alpha=0.25, label=band_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.legend(loc="best")
    fig.tight_layout()

    if outpath_no_ext is not None:
        save_figure(fig, outpath_no_ext)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
