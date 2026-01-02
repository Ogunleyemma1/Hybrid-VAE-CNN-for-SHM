# src/utils/__init__.py
"""
Utility subpackage for the OpenLAB hybrid SHM repository.

Includes:
- seed: reproducibility and deterministic behavior controls
- logging: consistent logging to console and optionally to file
- paths: repo-root aware path helpers and output directory conventions
"""

from .seed import set_seed, set_torch_determinism
from .logging import get_logger, configure_logging, log_experiment_header, dump_config_text
from .paths import (
    find_repo_root,
    as_path,
    ensure_dir,
    resolve_under_root,
    default_experiment_dirs,
)

__all__ = [
    "set_seed",
    "set_torch_determinism",
    "get_logger",
    "configure_logging",
    "log_experiment_header",
    "dump_config_text",
    "find_repo_root",
    "as_path",
    "ensure_dir",
    "resolve_under_root",
    "default_experiment_dirs",
]
