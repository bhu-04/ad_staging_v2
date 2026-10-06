# E2 — Normalization ablation (validation only)

96×96, locked split, seed 42. Baseline = per-image z-score; alternative = per-image min-max [0,1]. Test split was not accessed.

| Model | Normalization | Val acc | Sel. macro-F1 | Macro-F1 (3 present) | Val loss | ECE | Brier | Best ep | Train s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN2D | zscore | not available |||||||
| CNN2D | minmax | 0.4773 | 0.4658 | 0.4658 | 1.1466 | 0.2720 | 0.6972 | 14 | 14.7 |
| DenseNet2D | zscore | not available |||||||
| DenseNet2D | minmax | 0.4318 | 0.3348 | 0.3348 | 1.3744 | 0.1687 | 0.7440 | 1 | 26.2 |

Caveat: 44 validation images, one seed, and no repeated runs. Differences are exploratory, not evidence of superiority.
