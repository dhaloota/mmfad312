"""Node-level MMFAD anomaly scoring."""

from __future__ import annotations

import torch

from .graph_utils import graph_propagate


def compute_anomaly_scores(
    fused: torch.Tensor,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    adj_norm: torch.Tensor,
    lambda_reconstruction: float = 1.0,
) -> torch.Tensor:
    """Compute z(v_i) = ||f_i - f_i^exp||_2 + lambda_r ||x_i - xhat_i||_2.

    In this unsupervised implementation, the expected normal embedding
    f_i^exp is the normalized-neighborhood prototype A_norm F. This makes the
    score directly computable without using anomaly labels during training.
    """

    expected = graph_propagate(fused, adj_norm)
    deviation = torch.linalg.vector_norm(fused - expected, ord=2, dim=1)
    reconstruction = torch.linalg.vector_norm(x - x_hat, ord=2, dim=1)
    score = deviation + lambda_reconstruction * reconstruction
    return score
