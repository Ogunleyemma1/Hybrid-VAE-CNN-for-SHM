from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


LINE_W = 1.5


def configure_axis(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    ax.set_facecolor("none")


def save_figure(fig, out_dir: Path, file_stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{file_stem}.pdf", format="pdf", bbox_inches="tight", transparent=True)
    fig.savefig(out_dir / f"{file_stem}.png", format="png", bbox_inches="tight", transparent=True, dpi=300)
    fig.savefig(out_dir / f"{file_stem}.svg", format="svg", bbox_inches="tight", transparent=True)


def _read_rmse_csv(candidates: list[Path]) -> tuple[pd.DataFrame, Path]:
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            return df, p
    raise FileNotFoundError("RMSE CSV not found. Tried:\n" + "\n".join(str(p) for p in candidates))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either:
      - segment_index / rmse
      - segment / rmse
    Returns standardized columns: segment_index, rmse
    """
    cols = set(df.columns)

    if "rmse" not in cols:
        raise ValueError(f"RMSE column not found. Available columns: {list(df.columns)}")

    if "segment_index" in cols:
        df = df.rename(columns={"segment_index": "segment_index"})
    elif "segment" in cols:
        df = df.rename(columns={"segment": "segment_index"})
    else:
        raise ValueError(f"Segment column not found. Available columns: {list(df.columns)}")

    df = df[["segment_index", "rmse"]].copy()
    df["segment_index"] = pd.to_numeric(df["segment_index"], errors="coerce")
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df = df.dropna().sort_values("segment_index").reset_index(drop=True)
    df["segment_index"] = df["segment_index"].astype(int)
    return df


def main() -> None:
    root = Path(__file__).resolve().parents[1]  # 1_DOF

    # New (current) locations produced by updated scripts
    seen_candidates = [
        root / "Output" / "tables" / "reconstruction_seen" / "segment_rmse.csv",
        root / "output" / "tables" / "reconstruction_seen" / "segment_rmse.csv",
    ]
    unseen_candidates = [
        root / "Output" / "tables" / "reconstruction_unseen" / "segment_rmse.csv",
        root / "output" / "tables" / "reconstruction_unseen" / "segment_rmse.csv",
    ]

    # Legacy (older) locations, if you still have them
    seen_candidates += [
        root / "Output" / "tables" / "seen" / "segment_rmse_stats_seen.csv",
        root / "output" / "tables" / "seen" / "segment_rmse_stats_seen.csv",
    ]
    unseen_candidates += [
        root / "Output" / "tables" / "unseen" / "segment_rmse_stats_unseen.csv",
        root / "output" / "tables" / "unseen" / "segment_rmse_stats_unseen.csv",
    ]

    seen_raw, seen_path = _read_rmse_csv(seen_candidates)
    unseen_raw, unseen_path = _read_rmse_csv(unseen_candidates)

    seen = _normalize_columns(seen_raw)
    unseen = _normalize_columns(unseen_raw)

    out_dir = root / "Output" / "figures" / "rmse_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[OK] using seen RMSE:   {seen_path}")
    print(f"[OK] using unseen RMSE: {unseen_path}")

    # -----------------------------
    # Line comparison (legend below)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 8))

    h1 = ax.plot(
        seen["segment_index"],
        seen["rmse"],
        linewidth=LINE_W,
        marker="o",
        markersize=4.5,
        label="Seen",
    )[0]
    h2 = ax.plot(
        unseen["segment_index"],
        unseen["rmse"],
        linewidth=LINE_W,
        marker="o",
        markersize=4.5,
        label="Unseen",
    )[0]

    configure_axis(ax, xlabel="Segment index", ylabel="RMSE")

    fig.legend(
        handles=[h1, h2],
        labels=["Seen", "Unseen"],
        loc="lower center",
        ncol=2,
        fontsize=18,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=2.0,
        columnspacing=1.6,
    )

    fig.tight_layout(rect=(0.02, 0.07, 1.0, 1.0))
    save_figure(fig, out_dir, "rmse_seen_vs_unseen")
    plt.close(fig)

    # -----------------------------
    # Boxplot (distribution)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    bp = ax.boxplot(
        [seen["rmse"].values, unseen["rmse"].values],
        labels=["Seen", "Unseen"],
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )

    configure_axis(ax, xlabel="", ylabel="RMSE")

    for element in ["boxes", "whiskers", "caps", "medians"]:
        for artist in bp[element]:
            try:
                artist.set_linewidth(1.2)
            except Exception:
                pass

    fig.tight_layout()
    save_figure(fig, out_dir, "rmse_boxplot_seen_vs_unseen")
    plt.close(fig)

    # -----------------------------
    # Summary stats table
    # -----------------------------
    summary = pd.DataFrame(
        {
            "Set": ["Seen", "Unseen"],
            "Mean": [seen["rmse"].mean(), unseen["rmse"].mean()],
            "Median": [seen["rmse"].median(), unseen["rmse"].median()],
            "Std": [seen["rmse"].std(ddof=1), unseen["rmse"].std(ddof=1)],
            "Min": [seen["rmse"].min(), unseen["rmse"].min()],
            "Max": [seen["rmse"].max(), unseen["rmse"].max()],
        }
    )
    summary_path = out_dir / "rmse_summary_stats.csv"
    summary.to_csv(summary_path, index=False)

    print(f"[OK] wrote figures -> {out_dir}")
    print(f"[OK] wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
