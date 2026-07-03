"""Configuration utilities for the MMFAD reproducibility implementation."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import copy
import yaml


@dataclass
class DataConfig:
    """Paths and graph interpretation settings."""

    attributes_path: str = "data/DBLP-Cit_attributes.csv"
    edges_path: str = "data/DBLP-Cit_edges.csv"
    labels_path: str = "data/DBLP-Cit_ground_truth.csv"
    undirected: bool = True
    add_self_loops: bool = True
    expected_num_nodes: Optional[int] = 12793
    expected_num_edges: Optional[int] = 49743
    expected_num_attributes: Optional[int] = 28
    expected_num_anomalies: Optional[int] = 269


@dataclass
class TrainingConfig:
    """Training hyperparameters exposed for reviewer audit."""

    optimizer: str = "adam"
    learning_rate: float = 0.005
    epochs: int = 20
    embedding_dim: int = 128
    hidden_dim: int = 128
    sam_attention_dim: int = 32
    dropout: float = 0.10
    weight_decay: float = 0.0
    negative_edges_per_positive: float = 1.0
    gradient_clip_norm: Optional[float] = 5.0


@dataclass
class LossConfig:
    """Loss coefficients used in the MMFAD objective."""

    lambda_decorr: float = 0.01
    beta_reg: float = 0.001
    lambda_reconstruction: float = 1.0
    lambda_edge_reconstruction: float = 1.0


@dataclass
class FeedbackConfig:
    """Consensus-feedback refinement settings."""

    eta: float = 0.5
    steps: int = 1


@dataclass
class EvaluationConfig:
    """Ranking-based anomaly-detection evaluation settings."""

    topk_values: List[int] = field(default_factory=lambda: [50, 100, 200, 250])
    compute_auc: bool = True
    save_roc_png: bool = True
    save_roc_pdf: bool = True


@dataclass
class OutputConfig:
    """Output paths and saving behavior."""

    output_dir: str = "outputs"
    save_scores: bool = True
    save_metrics: bool = True
    save_config_snapshot: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    seed: int = 42
    device: str = "auto"
    standardize_features: bool = True
    num_torch_threads: int = 1
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested dictionary."""

    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _from_dict(data: Dict[str, Any]) -> ExperimentConfig:
    """Construct the dataclass configuration from a nested dictionary."""

    return ExperimentConfig(
        seed=int(data.get("seed", 42)),
        device=str(data.get("device", "auto")),
        standardize_features=bool(data.get("standardize_features", True)),
        num_torch_threads=int(data.get("num_torch_threads", 1)),
        data=DataConfig(**data.get("data", {})),
        training=TrainingConfig(**data.get("training", {})),
        loss=LossConfig(**data.get("loss", {})),
        feedback=FeedbackConfig(**data.get("feedback", {})),
        evaluation=EvaluationConfig(**data.get("evaluation", {})),
        outputs=OutputConfig(**data.get("outputs", {})),
    )


def load_config(config_path: Optional[str]) -> ExperimentConfig:
    """Load configuration from YAML and merge it with defaults.

    Parameters
    ----------
    config_path:
        Path to a YAML file. If ``None``, default settings are used.

    Returns
    -------
    ExperimentConfig
        Fully populated configuration object.
    """

    default_cfg = ExperimentConfig()
    if config_path is None:
        return default_cfg

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    merged = _deep_update(default_cfg.to_dict(), loaded)
    return _from_dict(merged)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save a configuration snapshot for reproducibility."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
