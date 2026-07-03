# MMFAD Model Card

## Model name

MMFAD: Consensus-Guided Multi-Modal Fusion for Attributed Graph Anomaly Detection.

## Input

An attributed graph with:

- Node attribute matrix `X` of shape `[n, d]`.
- Edge list `edge_index` of shape `[2, E]`.
- Binary labels `y` of shape `[n]` used only for evaluation.

## Encoder branches

- **GCN branch**: topology-conditioned feature propagation.
- **SAM branch**: feature-dimension self-attention for attribute salience.
- **GAE branch**: graph autoencoder branch trained with sparse edge reconstruction.

All branches output embeddings of shape `[n, 128]` by default.

## Fusion

Adaptive fusion learns convex weights over the three branch embeddings. Fusion logits are initialized to zero, producing uniform initial weights.

## Anomaly score

The score is:

```text
score_i = ||fused_i - expected_i||_2 + lambda_r ||x_i - x_hat_i||_2
```

where `expected_i` is the normalized-neighborhood embedding prototype from `A_norm @ F_fused`.

## Outputs

Higher score means more anomalous. Evaluation uses AUC and top-k metrics.
