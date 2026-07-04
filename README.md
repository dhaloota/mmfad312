# MMFAD Anonymous Implementation

This repository provides the implementation of **MMFAD**.

In this implementation, **multi-modal** means **multi-encoder / multi-perspective representation learning over one attributed graph**, not heterogeneous raw inputs such as text, image, or audio.

## What the code does

The pipeline performs the complete experiment:

1. Loads `DBLP-Cit_attributes.csv`, `DBLP-Cit_edges.csv`, and `DBLP-Cit_ground_truth.csv`.
2. Validates schema, node IDs, labels, feature columns, missing values, and edge consistency.
3. Builds sparse normalized graph propagation matrices.
4. Trains three encoder branches:
   - GCN branch for topology-conditioned feature propagation.
   - SAM branch for scalable attribute-dimension self-attention.
   - GAE branch for structural reconstruction.
5. Learns adaptive fusion weights initialized uniformly.
6. Applies consensus-feedback refinement.
7. Optimizes reconstruction, edge reconstruction, decorrelation, and structural regularization losses.
8. Computes node anomaly scores.
9. Evaluates AUC and Precision/Recall/F1 at k = 50, 100, 200, 250.
10. Saves anomaly scores, metrics, training logs, ROC curve, config snapshot, and model weights.

Labels are used **only for evaluation**, not for unsupervised training.

## Expected CSV files

Place these files in the `data/` folder:

```text
data/DBLP-Cit_attributes.csv
data/DBLP-Cit_edges.csv
data/DBLP-Cit_ground_truth.csv
```

Required schemas:

```text
DBLP-Cit_attributes.csv: node_id,attr_0,attr_1,...,attr_27
DBLP-Cit_edges.csv: source,target,weight
DBLP-Cit_ground_truth.csv: node_id,label
```

Label convention: `0 = normal`, `1 = anomalous`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell
pip install -r requirements.txt
```

## One-command reproduction

```bash
python run_mmfad.py --config config.yaml
```

Optional overrides:

```bash
python run_mmfad.py --config config.yaml --epochs 20 --device cpu
python run_mmfad.py --attributes data/DBLP-Cit_attributes.csv --edges data/DBLP-Cit_edges.csv --labels data/DBLP-Cit_ground_truth.csv
```

## Outputs

The default output folder is `outputs/` and contains:

```text
training_log.csv
anomaly_scores.csv
topk_results.csv
metrics.json
roc_curve_values.csv
roc_curve.png
roc_curve.pdf
config_snapshot.yaml
package_versions.json
mmfad_model_state_dict.pt
run_summary.txt
```

## Notes:

The implementation deliberately avoids hidden notebooks and manual post-processing. All important hyperparameters are visible in `config.yaml`. All major model choices are implemented in small inspectable modules under `mmfad/`.

This repository is an anonymized package prepared for double-blind review. It deliberately contains no author names, affiliations, contact details, or identifying paths.
