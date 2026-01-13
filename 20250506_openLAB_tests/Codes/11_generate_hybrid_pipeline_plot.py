from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


SPLIT_TO_EVAL = "test"   # "val" or "test"
LINE_W = 1.5

CM_CMAPS = [
    "Greens",
    "Purples",
    "Greys",
    "PuRd",
    "Reds",
    "Blues",
    "Oranges",
    "YlGnBu",
]

MODEL_BAR_COLORS = [
    "#4C78A8",  # CNN
    "#F58518",  # CART
    "#54A24B",  # RF
    "#E45756",  # GB
    "#72B7B2",  # HGB
    "#B279A2",  # SVM_RBF
]


def resolve_project_root() -> str:
    cwd = os.path.abspath(os.getcwd())
    parts = cwd.split(os.sep)
    if "Codes" in parts:
        idx = parts.index("Codes")
        return os.sep.join(parts[:idx])
    return cwd


PROJECT_DIR = resolve_project_root()
OUTPUT_ROOT = os.path.join(PROJECT_DIR, "Output")
EXP_DIR = os.path.join(OUTPUT_ROOT, "Full_Pipeline_Test", SPLIT_TO_EVAL)
PLOTS_DIR = os.path.join(EXP_DIR, "plots")
REPORTS_DIR = os.path.join(EXP_DIR, "reports")
os.makedirs(PLOTS_DIR, exist_ok=True)


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_stage2_metrics(path: str) -> Dict:
    return np.load(path, allow_pickle=True).item()


def apply_global_style() -> None:
    plt.style.use("fivethirtyeight")
    plt.rcParams["axes.linewidth"] = LINE_W
    plt.rcParams["lines.linewidth"] = LINE_W


def make_transparent(fig: plt.Figure) -> None:
    fig.patch.set_alpha(0.0)


def apply_bounding_box(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(LINE_W)
        spine.set_color("black")


def save_three(fig: plt.Figure, out_base: str) -> None:
    fig.savefig(out_base + ".pdf", bbox_inches="tight", transparent=True)
    fig.savefig(out_base + ".svg", bbox_inches="tight", transparent=True)
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight", transparent=True)


def row_normalize(cm_counts: np.ndarray) -> np.ndarray:
    cm = cm_counts.astype(np.float64)
    s = cm.sum(axis=1, keepdims=True)
    s[s == 0.0] = 1.0
    return cm / s


def plot_cm_grid_row_normalized(summary: Dict) -> None:
    labels = summary["labels_order"]
    models = summary["models"]

    cms = [np.array(m["confusion_matrix_counts_3class"], dtype=int) for m in models]
    titles = [f"({chr(ord('a') + i)}) VAE + {models[i]['name']}" for i in range(len(models))]

    n = len(cms)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols))

    apply_global_style()
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.7 * ncols, 6.0 * nrows))
    make_transparent(fig)

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue

        cm_norm = row_normalize(cms[i])
        cmap = CM_CMAPS[i % len(CM_CMAPS)]

        im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap=cmap, interpolation="nearest")
        ax.set_title(titles[i], fontsize=16)

        ax.set_xlabel("Predicted label", fontsize=13, labelpad=6)
        ax.set_ylabel("True label", fontsize=13, labelpad=18)

        ticks = np.arange(len(labels))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=0, fontsize=12)
        ax.set_yticklabels(labels, fontsize=12)

        apply_bounding_box(ax)

        for r in range(cm_norm.shape[0]):
            for c in range(cm_norm.shape[1]):
                v = cm_norm[r, c]
                ax.text(
                    c, r, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=12,
                    color=("white" if v >= 0.55 else "black"),
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.09)
        cbar.ax.tick_params(labelsize=11)
        cbar.outline.set_linewidth(LINE_W)

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.06, top=0.95, wspace=0.55, hspace=0.28)
    save_three(fig, os.path.join(PLOTS_DIR, "cm_grid_row_normalized"))
    plt.close(fig)


def plot_stage2_metrics_bar(metrics_obj: Dict) -> None:
    model_names = list(metrics_obj["model_names"])
    keys = ["Accuracy", "Precision", "Recall", "F1", "AUROC"]

    x = np.arange(len(keys))
    width = 0.14

    apply_global_style()
    fig = plt.figure(figsize=(15, 7.2))
    make_transparent(fig)
    ax = plt.gca()
    ax.grid(False)
    ax.set_facecolor("none")

    all_vals = []
    for i, name in enumerate(model_names):
        vals = [float(metrics_obj[k][i]) for k in keys]
        all_vals.extend([v for v in vals if np.isfinite(v)])

        color = MODEL_BAR_COLORS[i % len(MODEL_BAR_COLORS)]
        bars = ax.bar(
            x + (i - (len(model_names) - 1) / 2) * width,
            vals,
            width,
            label=name,
            color=color,
        )

        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    float(v) + 0.02,
                    f"{float(v):.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    rotation=90,
                )

    label_pad = 0.02
    top_pad = 0.10

    max_bar = max(all_vals) if len(all_vals) else 1.0
    ymax = min(1.20, max_bar + label_pad + top_pad)
    ax.set_ylim(0.0, ymax)

    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=20)
    ax.set_ylabel("Score", fontsize=20)
    ax.tick_params(axis="y", labelsize=20)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(LINE_W)
        spine.set_color("black")

    ax.legend(
        fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncols=len(model_names),
        frameon=True,
        columnspacing=1.2,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.99999])
    save_three(fig, os.path.join(PLOTS_DIR, "stage2_metrics_bar"))
    plt.close(fig)


def main() -> None:
    summary_path = os.path.join(REPORTS_DIR, "comparison_summary.json")
    metrics_path = os.path.join(REPORTS_DIR, "stage2_metrics.npy")

    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Missing: {summary_path}")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Missing: {metrics_path}")

    summary = read_json(summary_path)
    metrics_obj = load_stage2_metrics(metrics_path)

    plot_cm_grid_row_normalized(summary)
    plot_stage2_metrics_bar(metrics_obj)

    print(f"Saved plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
