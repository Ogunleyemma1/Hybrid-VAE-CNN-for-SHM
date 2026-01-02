# src/plotting/distributions.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt

from .style import PlotSettings, apply_plot_defaults, save_figure


PathLike = Union[str, Path]


def plot_histogram(
    data: Sequence[float],
    bins: int = 50,
    density: bool = False,
    title: Optional[str] = None,
    xlabel: str = "Value",
    ylabel: str = "Count",
    outpath_no_ext: Optional[PathLike] = None,
    cfg: PlotSettings = PlotSettings(),
    show: bool = False,
) -> plt.Figure:
    """
    Histogram plot for scalar distributions (e.g., VAE MSE, latent norms).
    """
    apply_plot_defaults(cfg)

    data = np.asarray(data, dtype=float)

    fig = plt.figure(figsize=cfg.figsize)
    ax = fig.gca()

    ax.hist(data, bins=bins, density=density, edgecolor="black", linewidth=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else ylabel)
    if title:
        ax.set_title(title)

    fig.tight_layout()

    if outpath_no_ext is not None:
        save_figure(fig, outpath_no_ext)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_kde_like_hist(
    data: Sequence[float],
    bins: int = 80,
    title: Optional[str] = None,
    xlabel: str = "Value",
    ylabel: str = "Density (approx.)",
    outpath_no_ext: Optional[PathLike] = None,
    cfg: PlotSettings = PlotSettings(),
    show: bool = False,
) -> plt.Figure:
    """
    A KDE-like visualization without external dependencies:
    histogram with density=True and thin edges.
    """
    return plot_histogram(
        data=data,
        bins=bins,
        density=True,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        outpath_no_ext=outpath_no_ext,
        cfg=cfg,
        show=show,
    )


def plot_boxplot(
    groups: Sequence[Sequence[float]],
    group_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    ylabel: str = "Value",
    outpath_no_ext: Optional[PathLike] = None,
    cfg: PlotSettings = PlotSettings(),
    show: bool = False,
) -> plt.Figure:
    """
    Boxplot for comparing distributions across groups (e.g., MSE across classes).

    Args:
        groups: list of arrays
        group_labels: list of strings matching groups
    """
    apply_plot_defaults(cfg)

    fig = plt.figure(figsize=cfg.figsize)
    ax = fig.gca()

    ax.boxplot(groups, labels=group_labels, showfliers=False)

    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    fig.tight_layout()

    if outpath_no_ext is not None:
        save_figure(fig, outpath_no_ext)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
