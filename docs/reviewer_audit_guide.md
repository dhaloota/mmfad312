# Reviewer Audit Guide

To audit the implementation:

1. Inspect `config.yaml` for all experimental settings.
2. Inspect `mmfad/validation.py` to verify CSV schema and consistency checks.
3. Inspect `mmfad/models.py` to verify GCN, SAM, GAE, fusion, decoder, and feedback refinement.
4. Inspect `mmfad/losses.py` to verify reconstruction, edge reconstruction, decorrelation, and structural regularization losses.
5. Inspect `mmfad/scoring.py` to verify the final anomaly scoring equation.
6. Run `python run_mmfad.py --config config.yaml`.
7. Inspect the generated `outputs/metrics.json`, `outputs/anomaly_scores.csv`, `outputs/topk_results.csv`, and ROC plots.

The code does not use labels during training. Labels are used only after training for AUC and top-k evaluation.
