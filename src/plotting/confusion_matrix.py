# src/plotting/confusion_matrix.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt

from .style import PlotSettings, apply_plot_defaults, save_figure


PathLike = Union[str, Path]


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[int],
    label_names: Optional[Dict[int, str]] = None,
    normalize: Optional[str] = None,
    title: Optional[str] = None,
    outpath_no_ext: Optional[PathLike] = None,
    cfg: PlotSettings = PlotSettings(),
    show: bool = False,
) -> plt.Figure:
    """
    Plot a confusion matrix with print-ready styling.

    Args:
        cm: (K,K) confusion matrix
        labels: label ids in order
        label_names: mapping id -> display name (e.g., {0:"Normal",1:"SF",2:"E"})
        normalize: None | "true" | "pred" | "all"
    """
    apply_plot_defaults(cfg)

    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be square (K,K).")

    names = [label_names.get(l, str(l)) if label_names else str(l) for l in labels]

    plot_cm = cm.copy()
    if normalize:
        eps = 1e-12
        if normalize == "true":
            plot_cm = plot_cm / (plot_cm.sum(axis=1, keepdims=True) + eps)
        elif normalize == "pred":
            plot_cm = plot_cm / (plot_cm.sum(axis=0, keepdims=True) + eps)
        elif normalize == "all":
            plot_cm = plot_cm / (plot_cm.sum() + eps)
        else:
            raise ValueError("normalize must be one of: None, 'true', 'pred', 'all'.")

    fig = plt.figure(figsize=cfg.figsize)
    ax = fig.gca()

    im = ax.imshow(plot_cm, interpolation="nearest")
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if title:
        ax.set_title(title)

    # Annotate cells
    fmt = ".2f" if normalize else "d"
    for i in range(plot_cm.shape[0]):
        for j in range(plot_cm.shape[1]):
            val = plot_cm[i, j]
            text = format(val, fmt) if normalize else str(int(val))
            ax.text(j, i, text, ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if outpath_no_ext is not None:
        save_figure(fig, outpath_no_ext)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
