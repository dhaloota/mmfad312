"""Plotting functions for saved reviewer-facing figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_value: float,
    output_dir: str | Path,
    save_png: bool = True,
    save_pdf: bool = True,
) -> None:
    """Save the ROC curve as PNG and/or PDF."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.5, 5.0))
    label = f"MMFAD (AUC = {auc_value:.4f})" if np.isfinite(auc_value) else "MMFAD (AUC unavailable)"
    plt.plot(fpr, tpr, label=label, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random ranking")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_png:
        fig.savefig(output_dir / "roc_curve.png", dpi=300)
    if save_pdf:
        fig.savefig(output_dir / "roc_curve.pdf")
    plt.close(fig)
