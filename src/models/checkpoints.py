# src/models/checkpoints.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


PathLike = Union[str, Path]


def _atomic_torch_save(obj: Any, path: Path) -> None:
    """
    Atomically write a PyTorch object to disk to reduce risk of partial writes.

    Implementation: write to a temporary file in the same directory and then
    replace the target path using os.replace (atomic on most filesystems).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# -----------------------------------------------------------------------------
# Backward-compatible helpers (state_dict only)
# -----------------------------------------------------------------------------

def save_model_state(path: PathLike, model: nn.Module) -> Path:
    """
    Save ONLY model.state_dict().

    This is intentionally maintained for backward compatibility with patterns like:
        model.load_state_dict(torch.load(PATH))

    Parameters
    ----------
    path : str | Path
        Destination file path (typically *.pt).
    model : nn.Module
        Model to save.

    Returns
    -------
    Path
        The saved path.
    """
    p = Path(path)
    _atomic_torch_save(model.state_dict(), p)
    return p


def load_model_state(
    path: PathLike,
    model: nn.Module,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> None:
    """
    Load a state_dict saved by save_model_state().

    Parameters
    ----------
    path : str | Path
        Checkpoint file path.
    model : nn.Module
        Model to load into.
    map_location : Optional[str | torch.device]
        torch.load map_location.
    strict : bool
        Passed to model.load_state_dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model state file not found: {p}")

    state = torch.load(p, map_location=map_location)
    if not isinstance(state, dict):
        raise TypeError("Expected a state_dict (dict). Got a non-dict object.")

    model.load_state_dict(state, strict=strict)


# -----------------------------------------------------------------------------
# Full training checkpoint (model + optimizer + metadata)
# -----------------------------------------------------------------------------

def save_checkpoint(
    path: PathLike,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a full training checkpoint for reproducibility and resuming.

    Stored keys
    ----------
    - model_state_dict (required)
    - optimizer_state_dict (optional)
    - scheduler_state_dict (optional)
    - epoch, step (optional)
    - metrics (optional)
    - extra (optional): e.g., resolved config, dataset manifest hashes

    Parameters
    ----------
    path : str | Path
        Destination checkpoint file path (typically *.pt).
    model : nn.Module
        Model to save.
    optimizer : Optional[torch.optim.Optimizer]
        Optimizer to save state for.
    scheduler : Optional[Any]
        Scheduler with state_dict/load_state_dict (optional).
    epoch : Optional[int]
        Epoch index (optional).
    step : Optional[int]
        Global step index (optional).
    metrics : Optional[dict]
        Numeric training/validation metrics (optional).
    extra : Optional[dict]
        Additional metadata (optional).

    Returns
    -------
    Path
        The saved checkpoint path.
    """
    p = Path(path)
    ckpt: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch) if epoch is not None else None,
        "step": int(step) if step is not None else None,
        "metrics": dict(metrics) if metrics is not None else {},
        "extra": dict(extra) if extra is not None else {},
    }

    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    _atomic_torch_save(ckpt, p)
    return p


def load_checkpoint(
    path: PathLike,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a full training checkpoint created by save_checkpoint().

    Parameters
    ----------
    path : str | Path
        Checkpoint file path.
    model : nn.Module
        Model to load into.
    optimizer : Optional[torch.optim.Optimizer]
        Optimizer to restore (if present in checkpoint).
    scheduler : Optional[Any]
        Scheduler to restore (if present in checkpoint).
    map_location : Optional[str | torch.device]
        torch.load map_location.
    strict : bool
        Passed to model.load_state_dict.

    Returns
    -------
    dict
        Full checkpoint dictionary (includes epoch/step/metrics/extra if present).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    ckpt = torch.load(p, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise TypeError("Invalid checkpoint: expected dict payload.")

    if "model_state_dict" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'. "
            "If this is a state_dict-only file, use load_model_state()."
        )

    model.load_state_dict(ckpt["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt


# -----------------------------------------------------------------------------
# Convenience: find latest checkpoint in a directory
# -----------------------------------------------------------------------------

def find_latest_checkpoint(
    directory: PathLike,
    prefix: str = "checkpoint",
    suffix: str = ".pt",
) -> Optional[Path]:
    """
    Return the checkpoint with the highest epoch number in a directory.

    Expected filename pattern:
        {prefix}_epoch####{suffix}
    Example:
        checkpoint_epoch0007.pt

    Parameters
    ----------
    directory : str | Path
        Directory containing checkpoint files.
    prefix : str
        Filename prefix (default: "checkpoint").
    suffix : str
        Filename suffix/extension (default: ".pt").

    Returns
    -------
    Optional[Path]
        Path to the latest checkpoint, or None if none match.
    """
    d = Path(directory)
    if not d.exists():
        return None

    pattern = re.compile(rf"^{re.escape(prefix)}_epoch(\d+){re.escape(suffix)}$")
    best_epoch: Optional[int] = None
    best_path: Optional[Path] = None

    for p in d.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        epoch = int(m.group(1))
        if best_epoch is None or epoch > best_epoch:
            best_epoch = epoch
            best_path = p

    return best_path
