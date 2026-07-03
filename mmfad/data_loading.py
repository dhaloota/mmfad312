"""CSV data loading for attributed graph anomaly-detection experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from .config import DataConfig


@dataclass
class GraphDataset:
    """Validated and tensorized graph dataset.

    Attributes
    ----------
    node_ids:
        NumPy array of node IDs sorted in the tensor order. Shape: [n].
    x:
        Node attribute tensor. Shape: [n, d].
    y:
        Binary anomaly labels. Shape: [n].
    edge_index_raw:
        Edge index from the CSV before optional symmetrization/self-loops.
        Shape: [2, E_csv].
    edge_weight_raw:
        Edge weights from the CSV. Shape: [E_csv].
    edge_index_model:
        Edge index used by the model after optional symmetrization/self-loops.
        Shape: [2, E_model].
    edge_weight_model:
        Edge weights used by the model. Shape: [E_model].
    feature_columns:
        Ordered attribute column names.
    mean:
        Feature mean used for standardization. Shape: [d].
    std:
        Feature standard deviation used for standardization. Shape: [d].
    """

    node_ids: np.ndarray
    x: torch.Tensor
    y: torch.Tensor
    edge_index_raw: torch.Tensor
    edge_weight_raw: torch.Tensor
    edge_index_model: torch.Tensor
    edge_weight_model: torch.Tensor
    feature_columns: List[str]
    mean: np.ndarray
    std: np.ndarray


def read_csv_files(data_cfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read attributes, edges, and labels from CSV files."""

    attr_path = Path(data_cfg.attributes_path)
    edge_path = Path(data_cfg.edges_path)
    label_path = Path(data_cfg.labels_path)

    for path in [attr_path, edge_path, label_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required CSV file not found: {path}")

    attributes = pd.read_csv(attr_path)
    edges = pd.read_csv(edge_path)
    labels = pd.read_csv(label_path)
    return attributes, edges, labels


def standardize_features_np(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardize node attributes with zero-variance protection."""

    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (x - mean) / std, mean.squeeze(0), std.squeeze(0)


def build_dataset(
    attributes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    data_cfg: DataConfig,
    standardize_features: bool = True,
) -> GraphDataset:
    """Validate CSV dataframes and convert them into tensors.

    Labels are returned for evaluation only and are not used by the unsupervised
    MMFAD training objective.
    """

    from .validation import validate_and_align_dataframes
    from .graph_utils import prepare_model_edges

    aligned_attr, aligned_edges, aligned_labels, feature_columns = validate_and_align_dataframes(
        attributes_df=attributes_df,
        edges_df=edges_df,
        labels_df=labels_df,
        data_cfg=data_cfg,
    )

    node_ids = aligned_attr["node_id"].to_numpy(dtype=np.int64)
    x_np = aligned_attr[feature_columns].to_numpy(dtype=np.float32)
    if standardize_features:
        x_np, mean, std = standardize_features_np(x_np)
    else:
        mean = np.zeros(x_np.shape[1], dtype=np.float32)
        std = np.ones(x_np.shape[1], dtype=np.float32)

    y_np = aligned_labels["label"].to_numpy(dtype=np.int64)

    raw_src = aligned_edges["source"].to_numpy(dtype=np.int64)
    raw_dst = aligned_edges["target"].to_numpy(dtype=np.int64)
    raw_weight = aligned_edges["weight"].to_numpy(dtype=np.float32)

    edge_index_raw = torch.tensor(np.vstack([raw_src, raw_dst]), dtype=torch.long)
    edge_weight_raw = torch.tensor(raw_weight, dtype=torch.float32)

    edge_index_model, edge_weight_model = prepare_model_edges(
        edge_index=edge_index_raw,
        edge_weight=edge_weight_raw,
        num_nodes=len(node_ids),
        undirected=data_cfg.undirected,
        add_self_loops=data_cfg.add_self_loops,
    )

    return GraphDataset(
        node_ids=node_ids,
        x=torch.tensor(x_np, dtype=torch.float32),
        y=torch.tensor(y_np, dtype=torch.long),
        edge_index_raw=edge_index_raw,
        edge_weight_raw=edge_weight_raw,
        edge_index_model=edge_index_model,
        edge_weight_model=edge_weight_model,
        feature_columns=feature_columns,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )


def load_dataset(data_cfg: DataConfig, standardize_features: bool = True) -> GraphDataset:
    """One-call CSV loader returning a validated ``GraphDataset`` object."""

    attributes, edges, labels = read_csv_files(data_cfg)
    return build_dataset(attributes, edges, labels, data_cfg, standardize_features)
