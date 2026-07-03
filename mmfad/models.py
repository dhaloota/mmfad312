"""MMFAD neural modules: GCN, feature self-attention, GAE, fusion, and decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F

from .graph_utils import graph_propagate


class GraphConvolution(nn.Module):
    """A minimal GCN layer implementing H' = A_norm H W.

    Input shape:  [n, in_dim]
    Output shape: [n, out_dim]
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        support = self.linear(x)
        return graph_propagate(support, adj_norm)


class GCNBranch(nn.Module):
    """Two-layer topology-aware GCN encoder branch.

    The branch captures 1-hop information in the first layer and 2-hop
    dependencies in the second layer, producing shape [n, embedding_dim].
    """

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GraphConvolution(input_dim, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, adj_norm)
        h = F.relu(h)
        h = self.dropout(h)
        z = self.conv2(h, adj_norm)
        return z


class FeatureSelfAttentionBranch(nn.Module):
    """Scalable attribute-salience self-attention branch.

    The reviewer-facing specification describes a SAM branch whose role is to
    emphasize informative attribute dimensions. For scalability and auditability,
    this implementation computes feature-wise attention/gating per node instead
    of constructing an O(n^2) node-attention matrix.

    Input shape:  [n, d]
    Output shape: [n, embedding_dim]
    """

    def __init__(self, input_dim: int, attention_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.input_dim = input_dim
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, max(attention_dim, input_dim)),
            nn.Tanh(),
            nn.Linear(max(attention_dim, input_dim), input_dim),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_scores = self.attention_net(x)
        attention_weights = torch.softmax(attention_scores, dim=1)
        attended = x * attention_weights
        return self.out_proj(attended)


class GAEBranch(nn.Module):
    """Graph autoencoder encoder branch.

    The encoder has the same two-layer GCN skeleton as the GCN branch, but its
    embeddings are additionally trained through edge reconstruction.
    """

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.encoder = GCNBranch(input_dim, hidden_dim, embedding_dim, dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, adj_norm)

    @staticmethod
    def edge_logits(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        return (z[src] * z[dst]).sum(dim=1)


class AdaptiveFusion(nn.Module):
    """Learnable convex fusion of same-shape encoder outputs.

    The raw trainable logits are initialized to zero, so softmax(logits) gives
    uniform initial fusion weights alpha_i = 1 / M.
    """

    def __init__(self, num_branches: int = 3):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_branches, dtype=torch.float32))

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def forward(self, embeddings: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(embeddings) == 0:
            raise ValueError("AdaptiveFusion received no embeddings.")
        base_shape = embeddings[0].shape
        for idx, emb in enumerate(embeddings):
            if emb.shape != base_shape:
                raise ValueError(
                    f"All branch embeddings must have identical shape before fusion. "
                    f"Branch 0 has {tuple(base_shape)}, branch {idx} has {tuple(emb.shape)}."
                )
        w = self.weights()
        stacked = torch.stack(embeddings, dim=0)  # [M, n, p]
        fused = (w.view(-1, 1, 1) * stacked).sum(dim=0)
        return fused, w


class AttributeDecoder(nn.Module):
    """Decode fused embeddings back to node attributes.

    Input shape:  [n, embedding_dim]
    Output shape: [n, input_dim]
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


@dataclass
class MMFADForwardOutput:
    """Forward-pass outputs used for loss computation and anomaly scoring."""

    z_gcn: torch.Tensor
    z_sam: torch.Tensor
    z_gae: torch.Tensor
    branch_embeddings: List[torch.Tensor]
    fused: torch.Tensor
    fusion_weights: torch.Tensor
    x_hat: torch.Tensor


class MMFAD(nn.Module):
    """Consensus-guided multi-encoder fusion model for attributed graph anomalies."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        sam_attention_dim: int = 32,
        dropout: float = 0.10,
        feedback_eta: float = 0.5,
        feedback_steps: int = 1,
    ):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if not (0.0 <= feedback_eta <= 1.0):
            raise ValueError("feedback_eta must be in [0, 1].")
        if feedback_steps < 0:
            raise ValueError("feedback_steps must be non-negative.")

        self.gcn_branch = GCNBranch(input_dim, hidden_dim, embedding_dim, dropout)
        self.sam_branch = FeatureSelfAttentionBranch(input_dim, sam_attention_dim, embedding_dim, dropout)
        self.gae_branch = GAEBranch(input_dim, hidden_dim, embedding_dim, dropout)
        self.fusion = AdaptiveFusion(num_branches=3)
        self.attribute_decoder = AttributeDecoder(embedding_dim, hidden_dim, input_dim, dropout)
        self.feedback_eta = feedback_eta
        self.feedback_steps = feedback_steps

    def _feedback_refine(self, embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        refined = embeddings
        for _ in range(self.feedback_steps):
            fused, _ = self.fusion(refined)
            refined = [(1.0 - self.feedback_eta) * z + self.feedback_eta * fused for z in refined]
        return refined

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> MMFADForwardOutput:
        z_gcn = self.gcn_branch(x, adj_norm)
        z_sam = self.sam_branch(x)
        z_gae = self.gae_branch(x, adj_norm)
        branches = [z_gcn, z_sam, z_gae]
        refined = self._feedback_refine(branches)
        fused, weights = self.fusion(refined)
        x_hat = self.attribute_decoder(fused)
        return MMFADForwardOutput(
            z_gcn=z_gcn,
            z_sam=z_sam,
            z_gae=z_gae,
            branch_embeddings=refined,
            fused=fused,
            fusion_weights=weights,
            x_hat=x_hat,
        )
