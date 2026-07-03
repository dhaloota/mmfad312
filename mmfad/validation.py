"""Reviewer-facing validation for DBLP-Cit-style attributed graph CSV files."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import DataConfig


def _require_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _is_integer_series(series: pd.Series) -> bool:
    return np.all(pd.to_numeric(series, errors="coerce").notna()) and np.all(
        np.equal(np.asarray(series, dtype=float), np.asarray(series, dtype=int))
    )


def _feature_sort_key(col: str) -> int:
    try:
        return int(col.split("_")[1])
    except Exception as exc:
        raise ValueError(f"Invalid attribute column name '{col}'. Expected attr_0, attr_1, ...") from exc


def validate_and_align_dataframes(
    attributes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    data_cfg: DataConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Validate schemas, consistency, and alignment across graph CSV files.

    Returns sorted attributes and labels in the same node order. Edges are left
    in CSV order after validation and duplicate removal. All node IDs are mapped
    to zero-based contiguous integer IDs by requiring that the CSV already uses
    this canonical convention.
    """

    _require_columns(attributes_df, ["node_id"], "attribute file")
    _require_columns(edges_df, ["source", "target"], "edge file")
    _require_columns(labels_df, ["node_id", "label"], "ground-truth file")

    if "weight" not in edges_df.columns:
        edges_df = edges_df.copy()
        edges_df["weight"] = 1.0

    attr = attributes_df.copy()
    edges = edges_df[["source", "target", "weight"]].copy()
    labels = labels_df[["node_id", "label"]].copy()

    if attr.isna().any().any():
        raise ValueError("attribute file contains missing values.")
    if edges.isna().any().any():
        raise ValueError("edge file contains missing values.")
    if labels.isna().any().any():
        raise ValueError("ground-truth file contains missing values.")

    if not _is_integer_series(attr["node_id"]):
        raise ValueError("attribute node_id column must contain integer IDs only.")
    if not _is_integer_series(labels["node_id"]):
        raise ValueError("label node_id column must contain integer IDs only.")
    if not _is_integer_series(edges["source"]) or not _is_integer_series(edges["target"]):
        raise ValueError("edge source/target columns must contain integer IDs only.")

    attr["node_id"] = attr["node_id"].astype(int)
    labels["node_id"] = labels["node_id"].astype(int)
    edges["source"] = edges["source"].astype(int)
    edges["target"] = edges["target"].astype(int)
    edges["weight"] = pd.to_numeric(edges["weight"], errors="raise").astype(float)

    if attr["node_id"].duplicated().any():
        dup = attr.loc[attr["node_id"].duplicated(), "node_id"].head().tolist()
        raise ValueError(f"attribute file contains duplicated node_id values, e.g. {dup}")
    if labels["node_id"].duplicated().any():
        dup = labels.loc[labels["node_id"].duplicated(), "node_id"].head().tolist()
        raise ValueError(f"ground-truth file contains duplicated node_id values, e.g. {dup}")

    node_ids = sorted(attr["node_id"].tolist())
    n = len(node_ids)
    expected_ids = list(range(n))
    if node_ids != expected_ids:
        raise ValueError(
            "node IDs must be contiguous zero-based integers from 0 to n-1. "
            f"Observed min={min(node_ids)}, max={max(node_ids)}, count={n}."
        )

    label_ids = sorted(labels["node_id"].tolist())
    if label_ids != expected_ids:
        raise ValueError("label file node IDs must exactly match attribute file node IDs.")

    if data_cfg.expected_num_nodes is not None and n != data_cfg.expected_num_nodes:
        raise ValueError(f"expected {data_cfg.expected_num_nodes} nodes, but found {n}.")

    feature_columns = [c for c in attr.columns if c.startswith("attr_")]
    feature_columns = sorted(feature_columns, key=_feature_sort_key)
    if len(feature_columns) == 0:
        raise ValueError("attribute file must contain attr_0, attr_1, ... columns.")

    if data_cfg.expected_num_attributes is not None and len(feature_columns) != data_cfg.expected_num_attributes:
        raise ValueError(
            f"expected {data_cfg.expected_num_attributes} attributes, but found {len(feature_columns)}."
        )

    expected_feature_cols = [f"attr_{i}" for i in range(len(feature_columns))]
    if feature_columns != expected_feature_cols:
        raise ValueError(
            "attribute columns must be contiguous and named attr_0, attr_1, ..., attr_d-1. "
            f"Observed: {feature_columns[:5]} ... {feature_columns[-5:]}"
        )

    for col in feature_columns:
        attr[col] = pd.to_numeric(attr[col], errors="raise")

    invalid_labels = sorted(set(labels["label"].tolist()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"labels must be binary with 0=normal and 1=anomaly. Invalid: {invalid_labels}")
    labels["label"] = labels["label"].astype(int)
    if data_cfg.expected_num_anomalies is not None:
        num_anomalies = int(labels["label"].sum())
        if num_anomalies != data_cfg.expected_num_anomalies:
            raise ValueError(
                f"expected {data_cfg.expected_num_anomalies} anomalies, but found {num_anomalies}."
            )

    all_ids = set(expected_ids)
    bad_edges = edges.loc[~edges["source"].isin(all_ids) | ~edges["target"].isin(all_ids)]
    if len(bad_edges) > 0:
        preview = bad_edges.head().to_dict(orient="records")
        raise ValueError(f"edge file contains endpoints outside node_id range: {preview}")

    before = len(edges)
    edges = edges.drop_duplicates(subset=["source", "target"], keep="first").reset_index(drop=True)
    if len(edges) != before:
        raise ValueError(
            f"edge file contains duplicated directed edges: {before - len(edges)} duplicates detected. "
            "Please clean the edge file or explicitly document duplicated multigraph edges."
        )

    if data_cfg.expected_num_edges is not None and len(edges) != data_cfg.expected_num_edges:
        raise ValueError(f"expected {data_cfg.expected_num_edges} CSV edges, but found {len(edges)}.")

    attr = attr.sort_values("node_id").reset_index(drop=True)
    labels = labels.sort_values("node_id").reset_index(drop=True)

    return attr, edges, labels, feature_columns
