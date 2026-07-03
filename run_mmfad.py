#!/usr/bin/env python3
"""Run the complete MMFAD reproducibility pipeline.

One-command usage from the repository root:

    python run_mmfad.py --config config.yaml

This script loads DBLP-Cit_attributes.csv, DBLP-Cit_edges.csv, and
DBLP-Cit_ground_truth.csv; validates them; trains the proposed MMFAD model;
computes anomaly scores; evaluates top-k and AUC metrics; and saves all outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from mmfad.config import load_config, save_config
from mmfad.data_loading import load_dataset
from mmfad.evaluation import anomaly_score_table, auc_and_roc, topk_metrics
from mmfad.plotting import save_roc_curve
from mmfad.training import train_mmfad
from mmfad.utils import ensure_dir, package_versions, save_json, select_device, set_deterministic_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate MMFAD on DBLP-Cit CSV files.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file.")
    parser.add_argument("--attributes", type=str, default=None, help="Override path to DBLP-Cit_attributes.csv.")
    parser.add_argument("--edges", type=str, default=None, help="Override path to DBLP-Cit_edges.csv.")
    parser.add_argument("--labels", type=str, default=None, help="Override path to DBLP-Cit_ground_truth.csv.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"], help="Override device.")
    return parser.parse_args()


def apply_cli_overrides(config: Any, args: argparse.Namespace) -> Any:
    if args.attributes is not None:
        config.data.attributes_path = args.attributes
    if args.edges is not None:
        config.data.edges_path = args.edges
    if args.labels is not None:
        config.data.labels_path = args.labels
    if args.output_dir is not None:
        config.outputs.output_dir = args.output_dir
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.device is not None:
        config.device = args.device
    return config


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)

    set_deterministic_seed(config.seed)
    if getattr(config, "num_torch_threads", None) is not None and int(config.num_torch_threads) > 0:
        torch.set_num_threads(int(config.num_torch_threads))
    device = select_device(config.device)
    output_dir = ensure_dir(config.outputs.output_dir)

    print("=" * 78)
    print("MMFAD: Consensus-Guided Multi-Modal Fusion for Attributed Graph Anomaly Detection")
    print("Multi-modal here means multi-encoder representations over one attributed graph.")
    print(f"Device: {device}")
    print("=" * 78)

    dataset = load_dataset(config.data, standardize_features=config.standardize_features)
    print(
        f"Loaded dataset: nodes={dataset.x.shape[0]}, attributes={dataset.x.shape[1]}, "
        f"CSV_edges={dataset.edge_index_raw.shape[1]}, anomalies={int(dataset.y.sum())}."
    )

    model, training_log, final = train_mmfad(dataset, config, device)

    y_np = dataset.y.cpu().numpy()
    scores = np.asarray(final["scores"], dtype=float)
    topk_df = topk_metrics(y_np, scores, config.evaluation.topk_values)
    auc_value, fpr, tpr, thresholds = auc_and_roc(y_np, scores)
    scores_df = anomaly_score_table(dataset.node_ids, y_np, scores)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})

    training_log.to_csv(output_dir / "training_log.csv", index=False)
    topk_df.to_csv(output_dir / "topk_results.csv", index=False)
    scores_df.to_csv(output_dir / "anomaly_scores.csv", index=False)
    roc_df.to_csv(output_dir / "roc_curve_values.csv", index=False)

    if config.evaluation.save_roc_png or config.evaluation.save_roc_pdf:
        save_roc_curve(
            fpr=fpr,
            tpr=tpr,
            auc_value=auc_value,
            output_dir=output_dir,
            save_png=config.evaluation.save_roc_png,
            save_pdf=config.evaluation.save_roc_pdf,
        )

    final_fusion_weights = np.asarray(final["fusion_weights"], dtype=float)
    metrics: Dict[str, Any] = {
        "dataset": {
            "num_nodes": int(dataset.x.shape[0]),
            "num_csv_edges": int(dataset.edge_index_raw.shape[1]),
            "num_model_edges_after_symmetry_and_self_loops": int(dataset.edge_index_model.shape[1]),
            "num_attributes": int(dataset.x.shape[1]),
            "num_anomalies": int(y_np.sum()),
            "anomaly_percentage": float(100.0 * y_np.sum() / len(y_np)),
        },
        "training": {
            "epochs": int(config.training.epochs),
            "learning_rate": float(config.training.learning_rate),
            "embedding_dim": int(config.training.embedding_dim),
            "hidden_dim": int(config.training.hidden_dim),
            "optimizer": config.training.optimizer,
            "labels_used_for_training": False,
        },
        "evaluation": {
            "auc": auc_value,
            "topk": topk_df.to_dict(orient="records"),
        },
        "fusion_weights": {
            "gcn": float(final_fusion_weights[0]),
            "sam": float(final_fusion_weights[1]),
            "gae": float(final_fusion_weights[2]),
        },
        "versions": package_versions(),
    }
    save_json(metrics, output_dir / "metrics.json")
    save_json(package_versions(), output_dir / "package_versions.json")

    if config.outputs.save_config_snapshot:
        save_config(config, output_dir / "config_snapshot.yaml")

    torch.save(model.state_dict(), output_dir / "mmfad_model_state_dict.pt")

    with (output_dir / "run_summary.txt").open("w", encoding="utf-8") as f:
        f.write("MMFAD run completed successfully.\n")
        f.write(f"Device: {device}\n")
        f.write(f"Nodes: {dataset.x.shape[0]}\n")
        f.write(f"CSV edges: {dataset.edge_index_raw.shape[1]}\n")
        f.write(f"Attributes: {dataset.x.shape[1]}\n")
        f.write(f"Anomalies: {int(y_np.sum())}\n")
        f.write(f"AUC: {auc_value}\n")
        f.write(f"Fusion weights [GCN, SAM, GAE]: {final_fusion_weights.tolist()}\n")
        f.write("Labels were used only for evaluation, not for unsupervised training.\n")

    print("Training and evaluation completed.")
    print(f"AUC: {auc_value:.6f}" if np.isfinite(auc_value) else "AUC: unavailable")
    print("Top-k results:")
    print(topk_df.to_string(index=False))
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
