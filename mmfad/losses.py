"""Loss functions for the MMFAD objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from .graph_utils import graph_propagate
from .models import GAEBranch, MMFADForwardOutput


@dataclass
class LossBreakdown:
    """Named scalar losses saved for transparent auditing."""

    total: torch.Tensor
    attribute_reconstruction: torch.Tensor
    edge_reconstruction: torch.Tensor
    decorrelation: torch.Tensor
    structural_regularization: torch.Tensor

    def to_float_dict(self) -> Dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_attribute_reconstruction": float(self.attribute_reconstruction.detach().cpu()),
            "loss_edge_reconstruction": float(self.edge_reconstruction.detach().cpu()),
            "loss_decorrelation": float(self.decorrelation.detach().cpu()),
            "loss_structural_regularization": float(self.structural_regularization.detach().cpu()),
        }


def attribute_reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean-squared reconstruction loss for node attributes."""

    return F.mse_loss(x_hat, x)


def edge_reconstruction_loss(
    z_gae: torch.Tensor,
    positive_edge_index: torch.Tensor,
    negative_edge_index: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy edge reconstruction loss for a GAE branch."""

    pos_logits = GAEBranch.edge_logits(z_gae, positive_edge_index)
    neg_logits = GAEBranch.edge_logits(z_gae, negative_edge_index)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels)


def decorrelation_loss(branch_embeddings: List[torch.Tensor]) -> torch.Tensor:
    """Inter-encoder dependence-control penalty.

    This implements the manuscript's pairwise cosine-based dependence term in a
    stable reviewer-facing form: pairwise node-wise cosine similarities are
    averaged and squared, penalizing collapsed highly similar branch outputs.
    """

    if len(branch_embeddings) < 2:
        return torch.tensor(0.0, device=branch_embeddings[0].device)
    loss = torch.tensor(0.0, device=branch_embeddings[0].device)
    pairs = 0
    for i in range(len(branch_embeddings)):
        zi = F.normalize(branch_embeddings[i], p=2, dim=1, eps=1e-12)
        for j in range(i + 1, len(branch_embeddings)):
            zj = F.normalize(branch_embeddings[j], p=2, dim=1, eps=1e-12)
            mean_cosine = (zi * zj).sum(dim=1).mean()
            loss = loss + mean_cosine.pow(2)
            pairs += 1
    return loss / max(pairs, 1)


def structural_regularization_loss(z: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
    """Encourage structurally smoothed embeddings without using labels."""

    z_expected = graph_propagate(z, adj_norm)
    return F.mse_loss(z, z_expected)


def total_mmfad_loss(
    output: MMFADForwardOutput,
    x: torch.Tensor,
    adj_norm: torch.Tensor,
    positive_edge_index: torch.Tensor,
    negative_edge_index: torch.Tensor,
    lambda_decorr: float,
    beta_reg: float,
    lambda_reconstruction: float,
    lambda_edge_reconstruction: float,
) -> LossBreakdown:
    """Compute the complete unsupervised MMFAD training objective.

    L_total = lambda_rec * L_attr + lambda_edge * L_edge
              + lambda_decorr * L_decorr + beta_reg * L_reg
    """

    attr_loss = attribute_reconstruction_loss(output.x_hat, x)
    edge_loss = edge_reconstruction_loss(output.z_gae, positive_edge_index, negative_edge_index)
    decorr = decorrelation_loss(output.branch_embeddings)
    reg = structural_regularization_loss(output.fused, adj_norm)
    total = (
        lambda_reconstruction * attr_loss
        + lambda_edge_reconstruction * edge_loss
        + lambda_decorr * decorr
        + beta_reg * reg
    )
    return LossBreakdown(
        total=total,
        attribute_reconstruction=attr_loss,
        edge_reconstruction=edge_loss,
        decorrelation=decorr,
        structural_regularization=reg,
    )
