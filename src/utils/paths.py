# src/utils/paths.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union


PathLike = Union[str, Path]


def as_path(p: PathLike) -> Path:
    """Normalize to a pathlib.Path."""
    return p if isinstance(p, Path) else Path(p)


def ensure_dir(path: PathLike) -> Path:
    """Create a directory (parents ok). Returns the Path."""
    p = as_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_repo_root(start: Optional[PathLike] = None) -> Path:
    """
    Find repository root by walking upward until a marker is found.

    Markers:
    - pyproject.toml
    - .git
    - README.md

    This avoids hard-coded paths and makes scripts runnable from any working dir.
    """
    if start is None:
        start_path = Path.cwd()
    else:
        start_path = as_path(start).resolve()

    markers = ["pyproject.toml", ".git", "README.md"]

    p = start_path
    for _ in range(50):  # safety guard
        for m in markers:
            if (p / m).exists():
                return p
        if p.parent == p:
            break
        p = p.parent

    raise FileNotFoundError(
        "Repository root not found. Expected one of: pyproject.toml, .git, README.md"
    )


def resolve_under_root(relative_path: PathLike, root: Optional[PathLike] = None) -> Path:
    """
    Resolve a path relative to the repository root.

    Args:
        relative_path: e.g. 'experiments/openlab/outputs'
        root: optional explicit repo root; if None, auto-detected

    Returns:
        absolute Path
    """
    root = find_repo_root() if root is None else as_path(root)
    return (root / as_path(relative_path)).resolve()


def default_experiment_dirs(experiment_dir: PathLike) -> Dict[str, Path]:
    """
    Standardize output directories for an experiment.

    Args:
        experiment_dir: e.g. 'experiments/openlab'

    Returns:
        dict with common subdirectories created:
            - outputs
            - figures
            - models
            - metrics
            - logs
    """
    exp = as_path(experiment_dir)
    out = exp / "outputs"
    dirs = {
        "outputs": ensure_dir(out),
        "figures": ensure_dir(out / "figures"),
        "models": ensure_dir(out / "models"),
        "metrics": ensure_dir(out / "metrics"),
        "logs": ensure_dir(out / "logs"),
    }
    return dirs
