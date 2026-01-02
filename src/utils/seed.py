# src/utils/seed.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def set_torch_determinism(deterministic: bool = True) -> None:
    """
    Configure PyTorch deterministic settings.

    Notes:
    - Deterministic GPU kernels can be slower.
    - Some operations remain nondeterministic depending on hardware and versions.
    """
    if torch is None:
        return

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms can raise if unsupported ops are used
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # We keep training functional even if strict determinism is unavailable.
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> int:
    """
    Set random seeds across Python, NumPy, and PyTorch.

    Args:
        seed: seed value used everywhere
        deterministic_torch: if True, configures PyTorch deterministic behavior

    Returns:
        The seed used (int).
    """
    seed = int(seed)

    # Python / OS
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        set_torch_determinism(deterministic_torch)

    return seed
