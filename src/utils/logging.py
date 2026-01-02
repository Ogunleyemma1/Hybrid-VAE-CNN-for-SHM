# src/utils/logging.py
from __future__ import annotations

import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union


PathLike = Union[str, Path]


def get_logger(name: str = "openlab") -> logging.Logger:
    """
    Get a named logger. Use configure_logging() once at program start.
    """
    return logging.getLogger(name)


def configure_logging(
    name: str = "openlab",
    level: int = logging.INFO,
    log_file: Optional[PathLike] = None,
    overwrite: bool = True,
) -> logging.Logger:
    """
    Configure logging to console and optionally a file.

    Args:
        name: logger name
        level: logging.INFO / DEBUG / etc.
        log_file: optional path to write logs
        overwrite: if True, truncates the log file; else appends

    Returns:
        configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent duplicate logs when root logger is configured elsewhere

    # Clear existing handlers (important for notebooks / repeated runs)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "w" if overwrite else "a"
        fh = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_experiment_header(
    logger: logging.Logger,
    title: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a standardized run header. Reviewers appreciate consistent run metadata.
    """
    logger.info("=" * 80)
    logger.info(title)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python:   {sys.version.split()[0]}")
    try:
        import numpy as np  # local import to avoid hard dependency at import time
        logger.info(f"NumPy:    {np.__version__}")
    except Exception:
        pass
    try:
        import torch
        logger.info(f"PyTorch:  {torch.__version__}")
        logger.info(f"CUDA:     {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU:      {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    if extra:
        for k, v in extra.items():
            logger.info(f"{k}: {v}")
    logger.info("=" * 80)


def dump_config_text(
    config: Dict[str, Any],
    out_path: PathLike,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Persist a resolved config dict as JSON for reproducibility.

    Args:
        config: resolved configuration dictionary
        out_path: destination file (e.g. outputs/logs/config_resolved.json)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if logger is not None:
        logger.info(f"Saved resolved config to: {out_path}")
