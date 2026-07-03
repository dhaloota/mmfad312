"""Graph construction, sparse normalization, and negative-edge sampling."""

from __future__ import annotations

from typing import Tuple

import torch


def prepare_model_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    undirected: bool = True,
    add_self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare edges used by the model.

    The raw CSV edge count is preserved separately for dataset reporting. This
    function optionally symmetrizes edges for undirected GCN message passing and
    adds self-loops for standard GCN normalization.
    """

    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if edge_weight.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_weight length must match edge_index edge count.")

    edges = edge_index.clone().long()
    weights = edge_weight.clone().float()

    if undirected:
        rev = torch.stack([edges[1], edges[0]], dim=0)
        edges = torch.cat([edges, rev], dim=1)
        weights = torch.cat([weights, weights], dim=0)

    if add_self_loops:
        loops = torch.arange(num_nodes, dtype=torch.long)
        loop_edges = torch.stack([loops, loops], dim=0)
        loop_weights = torch.ones(num_nodes, dtype=torch.float32)
        edges = torch.cat([edges, loop_edges], dim=1)
        weights = torch.cat([weights, loop_weights], dim=0)

    # Coalesce duplicate entries by summing weights. This can happen when the CSV
    # already contains reciprocal edges and undirected=True.
    sparse = torch.sparse_coo_tensor(edges, weights, size=(num_nodes, num_nodes)).coalesce()
    return sparse.indices().long(), sparse.values().float()


def normalized_adjacency(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Build sparse symmetric GCN normalization D^{-1/2} A D^{-1/2}."""

    row, col = edge_index[0].to(device), edge_index[1].to(device)
    weight = edge_weight.to(device)
    degree = torch.zeros(num_nodes, device=device, dtype=weight.dtype)
    degree.scatter_add_(0, row, weight)
    deg_inv_sqrt = degree.clamp(min=1e-12).pow(-0.5)
    norm_weight = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    adj = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=norm_weight,
        size=(num_nodes, num_nodes),
        device=device,
    ).coalesce()
    return adj


def sample_negative_edges(
    positive_edge_index: torch.Tensor,
    num_nodes: int,
    num_samples: int,
    device: torch.device,
    max_attempt_factor: int = 5,
) -> torch.Tensor:
    """Sample negative directed edges not present in the positive edge set.

    This simple sampler is deterministic under PyTorch's random seed. It avoids
    self-loops and avoids positive CSV edges. It is intended for sparse graph
    autoencoder training without constructing a dense n x n adjacency matrix.
    """

    pos = positive_edge_index.detach().cpu().long()
    existing = set(zip(pos[0].tolist(), pos[1].tolist()))
    negatives = []
    target = int(num_samples)
    attempts = 0
    max_attempts = max(target * max_attempt_factor, 1000)

    while len(negatives) < target and attempts < max_attempts:
        batch = min(max(target - len(negatives), 1024), 65536)
        src = torch.randint(0, num_nodes, (batch,)).tolist()
        dst = torch.randint(0, num_nodes, (batch,)).tolist()
        for s, t in zip(src, dst):
            if s == t:
                continue
            key = (int(s), int(t))
            if key in existing:
                continue
            existing.add(key)
            negatives.append(key)
            if len(negatives) >= target:
                break
        attempts += batch

    if len(negatives) < target:
        raise RuntimeError(
            f"Could sample only {len(negatives)} negative edges out of requested {target}."
        )

    neg = torch.tensor(negatives, dtype=torch.long, device=device).t().contiguous()
    return neg


def normalized_edge_weights(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return edge_index and weights for D^{-1/2} A D^{-1/2} propagation."""

    edge_index = edge_index.to(device).long()
    weight = edge_weight.to(device).float()
    row, col = edge_index[0], edge_index[1]
    degree = torch.zeros(num_nodes, device=device, dtype=weight.dtype)
    degree.scatter_add_(0, row, weight)
    deg_inv_sqrt = degree.clamp(min=1e-12).pow(-0.5)
    norm_weight = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    return edge_index, norm_weight


def graph_propagate(x: torch.Tensor, norm_graph: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Sparse message passing equivalent to A_norm @ X using index_add.

    This avoids slow sparse-matrix backward paths on some CPU installations while
    preserving the same normalized GCN propagation semantics.
    """

    edge_index, norm_weight = norm_graph
    row, col = edge_index[0], edge_index[1]
    messages = x[col] * norm_weight.unsqueeze(-1)
    out = torch.zeros_like(x)
    out.index_add_(0, row, messages)
    return out
