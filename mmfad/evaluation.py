"""Evaluation utilities for ranking-based anomaly detection."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def topk_metrics(y_true: np.ndarray, scores: np.ndarray, topk_values: Iterable[int]) -> pd.DataFrame:
    """Compute Precision@k, Recall@k, and F1@k for higher-is-more-anomalous scores."""

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    order = np.argsort(-scores)
    total_anomalies = int(y_true.sum())
    rows = []
    for k in topk_values:
        k_eff = min(int(k), len(y_true))
        idx = order[:k_eff]
        tp = int(y_true[idx].sum())
        precision = tp / k_eff if k_eff > 0 else 0.0
        recall = tp / total_anomalies if total_anomalies > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append({"k": int(k), "true_positives": tp, "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def auc_and_roc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC-AUC and ROC curve points."""

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan"), np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])
    auc = float(roc_auc_score(y_true, scores))
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    return auc, fpr, tpr, thresholds


def anomaly_score_table(node_ids: np.ndarray, y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Create a reviewer-readable table of all node-level anomaly scores."""

    node_ids = np.asarray(node_ids).astype(int)
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return pd.DataFrame(
        {
            "node_id": node_ids,
            "label": y_true,
            "anomaly_score": scores,
            "rank_descending_score": ranks,
        }
    ).sort_values("rank_descending_score").reset_index(drop=True)
