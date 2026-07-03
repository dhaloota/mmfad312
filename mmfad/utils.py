"""General utilities for deterministic execution, device selection, and saving."""

from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_deterministic_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch.

    The exact bitwise reproducibility of sparse GPU operations may still depend on
    hardware and library versions. This function nevertheless activates the most
    important reproducibility controls for reviewer-facing execution.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(requested: str = "auto") -> torch.device:
    """Return CPU/GPU device from the configuration setting."""

    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested, but CUDA is not available.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of {'auto', 'cpu', 'cuda'}.")
    return torch.device(requested)


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    """Save a JSON object with readable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def package_versions() -> Dict[str, str]:
    """Return package and platform versions for reproducibility records."""

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    try:
        import pandas as pd

        versions["pandas"] = pd.__version__
    except Exception:
        versions["pandas"] = "unavailable"
    try:
        import sklearn

        versions["scikit_learn"] = sklearn.__version__
    except Exception:
        versions["scikit_learn"] = "unavailable"
    try:
        import matplotlib

        versions["matplotlib"] = matplotlib.__version__
    except Exception:
        versions["matplotlib"] = "unavailable"
    return versions
