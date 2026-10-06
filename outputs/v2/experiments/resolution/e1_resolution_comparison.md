# E1 — Resolution comparison (validation only)

96×96 = Phase 3 baseline (read-only, validation metrics). 128×128 = E1. Everything else identical (seed 42, demo profile, class-weighted CE, same augmentation/normalisation, locked split). The test split was not accessed.

| Model | Res | Val acc | Sel. macro-F1 | Macro-F1 (3 present) | Val loss | ECE | Brier | Best ep | Train s | Params |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN2D | 96×96 | 0.4318 | 0.4365 | 0.4365 | 1.0533 | 0.2154 | 0.6721 | 9 | 30.6 | 98,148 |
| CNN2D | 128×128 | 0.4545 | 0.4596 | 0.4596 | 1.0432 | 0.1597 | 0.6513 | 9 | 28.3 | 98,148 |
| DenseNet2D | 96×96 | 0.5227 | 0.4977 | 0.4977 | 1.0795 | 0.1632 | 0.6240 | 11 | 72.8 | 20,596 |
| DenseNet2D | 128×128 | 0.4545 | 0.3223 | 0.4297 | 1.2366 | 0.2274 | 0.6920 | 9 | 294.0 | 20,596 |

## Per-class validation recall / F1 (Moderate has no validation subject)

| Model | Class | Recall 96 | Recall 128 | F1 96 | F1 128 | Val n |
|---|---|---:|---:|---:|---:|---:|
| CNN2D | Non Demented | 0.467 | 0.467 | 0.609 | 0.609 | 15 |
| CNN2D | Very Mild Demented | 0.200 | 0.267 | 0.214 | 0.296 | 15 |
| CNN2D | Mild Demented | 0.643 | 0.643 | 0.486 | 0.474 | 14 |
| DenseNet2D | Non Demented | 0.667 | 0.533 | 0.667 | 0.593 | 15 |
| DenseNet2D | Very Mild Demented | 0.200 | 0.067 | 0.300 | 0.118 | 15 |
| DenseNet2D | Mild Demented | 0.714 | 0.786 | 0.526 | 0.579 | 14 |

Caveat: 44 validation images, one seed, no repeated runs — small differences are not evidence that one resolution is better. Resolution choice is made later from validation evidence only.

