# src/models/checkpoints.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


PathLike = Union[str, Path]


def _atomic_torch_save(obj: Any, path: Path) -> None:
    """
    Atomic save to reduce risk of partial checkpoint writes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_model_state(path: PathLike, model: nn.Module) -> None:
    """
    Backward-compatible save: stores ONLY model.state_dict().
    This matches your current usage:
        model.load_state_dict(torch.load(PATH))
    """
    _atomic_torch_save(model.state_dict(), Path(path))


def load_model_state(path: PathLike, model: nn.Module, map_location: Optional[Union[str, torch.device]] = None, strict: bool = True) -> None:
    """
    Backward-compatible load for save_model_state().
    """
    state = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(state, strict=strict)


def save_checkpoint(
    path: PathLike,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Full training checkpoint for reproducibility and resuming.

    Stores:
        - model_state_dict
        - optimizer_state_dict (optional)
        - scheduler_state_dict (optional)
        - epoch/step
        - metrics (optional)
        - extra (optional): e.g., resolved config, git commit, dataset manifest hashes
    """
    ckpt: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    _atomic_torch_save(ckpt, Path(path))


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
    Returns the full checkpoint dict for access to metrics/extra.
    """
    ckpt = torch.load(Path(path), map_location=map_location)

    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint does not contain 'model_state_dict' (did you save a state_dict-only file?)")

    model.load_state_dict(ckpt["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt
