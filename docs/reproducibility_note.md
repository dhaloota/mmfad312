# Reproducibility Note

This package is designed to address reviewer concerns about code availability and independent replication.

The code can be executed from scratch with:

```bash
python run_mmfad.py --config config.yaml
```

The pipeline includes deterministic seed control for Python, NumPy, and PyTorch. Labels are reserved for post-training evaluation only. The model saves all outputs needed to audit the result: anomaly scores, top-k metrics, ROC curve values, plots, metrics JSON, training log, configuration snapshot, and package versions.

The implementation is dependency-light and does not require PyTorch Geometric. Graph convolution is implemented directly with PyTorch sparse matrix multiplication.
