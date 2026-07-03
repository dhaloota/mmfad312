"""Training and inference pipeline for MMFAD."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .data_loading import GraphDataset
from .graph_utils import normalized_edge_weights, sample_negative_edges
from .losses import total_mmfad_loss
from .models import MMFAD
from .scoring import compute_anomaly_scores


def build_model(input_dim: int, config: ExperimentConfig) -> MMFAD:
    """Build the MMFAD model from configuration."""

    return MMFAD(
        input_dim=input_dim,
        hidden_dim=config.training.hidden_dim,
        embedding_dim=config.training.embedding_dim,
        sam_attention_dim=config.training.sam_attention_dim,
        dropout=config.training.dropout,
        feedback_eta=config.feedback.eta,
        feedback_steps=config.feedback.steps,
    )


def train_mmfad(
    dataset: GraphDataset,
    config: ExperimentConfig,
    device: torch.device,
) -> Tuple[MMFAD, pd.DataFrame, Dict[str, np.ndarray | float]]:
    """Train MMFAD and return the model, training log, and final tensors.

    The labels in ``dataset.y`` are deliberately not used in this training loop.
    They are reserved for post-training evaluation.
    """

    x = dataset.x.to(device)
    num_nodes, input_dim = x.shape
    edge_index_model = dataset.edge_index_model.to(device)
    edge_weight_model = dataset.edge_weight_model.to(device)
    edge_index_raw = dataset.edge_index_raw.to(device)

    adj_norm = normalized_edge_weights(edge_index_model, edge_weight_model, num_nodes, device)
    model = build_model(input_dim=input_dim, config=config).to(device)

    if config.training.optimizer.lower() != "adam":
        raise ValueError("This reviewer-facing implementation currently supports optimizer='adam'.")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    num_pos_edges = edge_index_raw.shape[1]
    num_neg_edges = max(1, int(num_pos_edges * config.training.negative_edges_per_positive))
    log_rows = []

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(x, adj_norm)
        neg_edges = sample_negative_edges(
            positive_edge_index=edge_index_raw,
            num_nodes=num_nodes,
            num_samples=num_neg_edges,
            device=device,
        )
        losses = total_mmfad_loss(
            output=output,
            x=x,
            adj_norm=adj_norm,
            positive_edge_index=edge_index_raw,
            negative_edge_index=neg_edges,
            lambda_decorr=config.loss.lambda_decorr,
            beta_reg=config.loss.beta_reg,
            lambda_reconstruction=config.loss.lambda_reconstruction,
            lambda_edge_reconstruction=config.loss.lambda_edge_reconstruction,
        )
        losses.total.backward()
        if config.training.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.gradient_clip_norm)
        optimizer.step()

        row = {"epoch": epoch}
        row.update(losses.to_float_dict())
        weights = output.fusion_weights.detach().cpu().numpy()
        row.update({f"fusion_weight_{i}": float(w) for i, w in enumerate(weights)})
        log_rows.append(row)

    training_log = pd.DataFrame(log_rows)

    model.eval()
    with torch.no_grad():
        final_output = model(x, adj_norm)
        scores = compute_anomaly_scores(
            fused=final_output.fused,
            x=x,
            x_hat=final_output.x_hat,
            adj_norm=adj_norm,
            lambda_reconstruction=config.loss.lambda_reconstruction,
        )

    final = {
        "scores": scores.detach().cpu().numpy(),
        "fusion_weights": final_output.fusion_weights.detach().cpu().numpy(),
        "fused_embeddings": final_output.fused.detach().cpu().numpy(),
    }
    return model, training_log, final
