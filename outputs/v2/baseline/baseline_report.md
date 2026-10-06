# Phase 3 — V2 Baseline Report

Generated: 2026-10-06T21:33:21+05:30  
Git commit: `7c1e682b460e2c5122bdff19d318c293a3338e8c` (branch `main`; working-tree changes at start: 1)

**VALIDATION = model selection. TEST = final held-out evaluation; no decision was made from test results.**

## A. Baseline configuration (identical for all three models)

| Item | Value |
|---|---|
| mode | demo |
| image_size | (96, 96) |
| random_seed | 42 |
| optimizer | AdamW |
| learning_rate | 0.001 |
| weight_decay | 0.0001 |
| batch_size | 32 |
| epochs | 25 |
| early_stop_patience | 6 |
| lr_scheduler | plateau |
| scheduler_params | ReduceLROnPlateau(mode=min, factor=0.5, patience=3) on val_loss |
| loss | CrossEntropyLoss(weight=inverse-frequency class weights) |
| class_balance_strategy | loss |
| dropout_p | 0.35 |
| max_attempts | 2 |
| checkpoint_selection | val macro-F1 > val accuracy > lower val loss (src/train.py) |
| preprocessing | grayscale -> bilinear resize -> per-image zero-mean/unit-variance |
| train_augmentation | RandomHorizontalFlip(p=0.5) + RandomRotation(+/-10 deg, bilinear, fill=0) |
| eval_augmentation | none (deterministic) |
| pretrained | False |
| ViT | {'patch_size': 8, 'embed_dim': 32, 'depth': 3, 'num_heads': 4, 'mlp_dim': 64} |
| Device | cpu (None) |

## B. Dataset / split (locked V1)

| Split | Images | Subjects | Non Demented | Very Mild Demented | Mild Demented | Moderate Demented |
|---|---:|---:|---:|---:|---:|---:|
| train | 273 | 126 | 70 (70 subj) | 70 (40 subj) | 73 (15 subj) | 60 (1 subj) |
| val | 44 | 27 | 15 (15 subj) | 15 (9 subj) | 14 (3 subj) | 0 (0 subj) |
| test | 83 | 28 | 15 (15 subj) | 15 (9 subj) | 13 (3 subj) | 40 (1 subj) |

Moderate Demented: OAS1_0351 (60 images) → train; OAS1_0308 (40 images) → test; **0 Moderate subjects in validation**.

## C. CNN2D

Parameters: 98,148 · best epoch: 9 (last epoch reached: 15) · attempts used: 1 · degenerate: False · train time: 30.6s · checkpoint sha256: `58470f0d76ec348c…`

**Validation (selection)**

- accuracy 0.4318 · selection macro-F1 (as computed by train.py) 0.4365 · macro-F1 over the 3 classes present 0.4365 · loss 1.0533
- ECE 0.2154 · Brier 0.6721 · ROC-AUC: not reported on validation (Moderate Demented absent from validation split)

**Test (frozen checkpoint, evaluated once)**

- accuracy 0.2651 · macro-P 0.4191 · macro-R 0.3974 · macro-F1 0.2991 · ROC-AUC 0.5994 · ECE 0.3955 · Brier 1.0815

| Class | Val P | Val R | Val F1 | Val n | Test P | Test R | Test F1 | Test n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non Demented | 0.875 | 0.467 | 0.609 | 15 | 1.000 | 0.333 | 0.500 | 15 |
| Very Mild Demented | 0.231 | 0.200 | 0.214 | 15 | 0.500 | 0.333 | 0.400 | 15 |
| Mild Demented | 0.391 | 0.643 | 0.486 | 14 | 0.176 | 0.923 | 0.296 | 13 |
| Moderate Demented | 0.000 | 0.000 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 40 |

Test confusion matrix (rows = true, cols = predicted; order: Non Demented, Very Mild Demented, Mild Demented, Moderate Demented)

```
   5    4    6    0
   0    5   10    0
   0    1   12    0
   0    0   40    0
```

## D. DenseNet2D

Parameters: 20,596 · best epoch: 11 (last epoch reached: 17) · attempts used: 1 · degenerate: False · train time: 72.8s · checkpoint sha256: `27cb54218e1af2cb…`

**Validation (selection)**

- accuracy 0.5227 · selection macro-F1 (as computed by train.py) 0.4977 · macro-F1 over the 3 classes present 0.4977 · loss 1.0795
- ECE 0.1632 · Brier 0.6240 · ROC-AUC: not reported on validation (Moderate Demented absent from validation split)

**Test (frozen checkpoint, evaluated once)**

- accuracy 0.2530 · macro-P 0.3750 · macro-R 0.3782 · macro-F1 0.2733 · ROC-AUC 0.5608 · ECE 0.2337 · Brier 0.9229

| Class | Val P | Val R | Val F1 | Val n | Test P | Test R | Test F1 | Test n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non Demented | 0.667 | 0.667 | 0.667 | 15 | 0.667 | 0.533 | 0.593 | 15 |
| Very Mild Demented | 0.600 | 0.200 | 0.300 | 15 | 0.667 | 0.133 | 0.222 | 15 |
| Mild Demented | 0.417 | 0.714 | 0.526 | 14 | 0.167 | 0.846 | 0.278 | 13 |
| Moderate Demented | 0.000 | 0.000 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 40 |

Test confusion matrix (rows = true, cols = predicted; order: Non Demented, Very Mild Demented, Mild Demented, Moderate Demented)

```
   8    1    6    0
   4    2    9    0
   0    0   11    2
   0    0   40    0
```

## E. ViT2D

Parameters: 32,580 · best epoch: 3 (last epoch reached: 9) · attempts used: 1 · degenerate: False · train time: 29.7s · checkpoint sha256: `a5bd44c7a012228d…`

**Validation (selection)**

- accuracy 0.3864 · selection macro-F1 (as computed by train.py) 0.2420 · macro-F1 over the 3 classes present 0.3227 · loss 1.3249
- ECE 0.0963 · Brier 0.7206 · ROC-AUC: not reported on validation (Moderate Demented absent from validation split)

**Test (frozen checkpoint, evaluated once)**

- accuracy 0.2048 · macro-P 0.1594 · macro-R 0.2859 · macro-F1 0.1396 · ROC-AUC 0.6323 · ECE 0.0972 · Brier 0.7630

| Class | Val P | Val R | Val F1 | Val n | Test P | Test R | Test F1 | Test n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non Demented | 1.000 | 0.067 | 0.125 | 15 | 0.333 | 0.067 | 0.111 | 15 |
| Very Mild Demented | 0.361 | 0.867 | 0.510 | 15 | 0.227 | 1.000 | 0.370 | 15 |
| Mild Demented | 0.750 | 0.214 | 0.333 | 14 | 0.077 | 0.077 | 0.077 | 13 |
| Moderate Demented | 0.000 | 0.000 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 40 |

Test confusion matrix (rows = true, cols = predicted; order: Non Demented, Very Mild Demented, Mild Demented, Moderate Demented)

```
   1   14    0    0
   0   15    0    0
   0   11    1    1
   2   26   12    0
```

## F. Validation-selection procedure

Within each run, the checkpoint is the epoch with the highest validation macro-F1; ties are broken by validation accuracy, then lower validation loss (`src/train.py`). Training stops after 6 epochs without improvement. If the best checkpoint is degenerate (single-class predictions or accuracy ≤ majority baseline) the repo retries once with a new seed, dropout 0.25 and 1.5× epochs; the best attempt by validation macro-F1 is kept. The checkpoint is frozen (sha256 recorded in `frozen_checkpoint.json`), reloaded, and verified to reproduce the recorded validation metrics before the test split is read.

## G. Final test results (summary)

| Model | Test acc | Test macro-F1 | Test ECE | Test Brier | Val acc | Val sel. macro-F1 | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN2D | 0.2651 | 0.2991 | 0.3955 | 1.0815 | 0.4318 | 0.4365 | 9 |
| DenseNet2D | 0.2530 | 0.2733 | 0.2337 | 0.9229 | 0.5227 | 0.4977 | 11 |
| ViT2D | 0.2048 | 0.1396 | 0.0972 | 0.7630 | 0.3864 | 0.2420 | 3 |

This table is descriptive. It is not a model ranking: 44 validation / 83 test images from 27 / 28 subjects, a single seed, and a single-subject Moderate test class do not support claims that any model is superior.

## H. Runtime

- CNN2D: train 30.6s, total 41.0s
- DenseNet2D: train 72.8s, total 82.0s
- ViT2D: train 29.7s, total 38.0s

## I. Implementation issues encountered

- None recorded.

## J. Deviations / notes on the documented baseline

- Phase 3 uses the 'demo' profile in src/config.py (96×96, 25 epochs, patience 6, AdamW lr 1e-3 / wd 1e-4, batch 32, dropout 0.35, ReduceLROnPlateau, loss-weighted CE, seed 42). `run_pipeline.py --mode demo` only performs a forward-pass demonstration and does not train, so this dedicated runner calls src.train.train_model directly. The 'full' profile (64×64, 30 epochs) is NOT the baseline.
- src/train.py::train_model writes a single shared training_history.json to config.METRICS_DIR; the runner redirects METRICS_DIR to each model's own directory during training so nothing shared is overwritten.
- src/train.py computes validation macro-F1 with f1_score(average='macro') and no labels=. Validation has no Moderate images, so the selection metric averages over a prediction-dependent class set (3 or 4 classes). This is the documented repository behaviour and was kept unchanged; macro-F1 over the 3 present classes is reported alongside it.
- run_pipeline.run_demo builds DenseNet2D with dropout_p=0.3, whereas config.DROPOUT_P = 0.35; the baseline uses 0.35 for all models. DenseNet2D is the compact from-scratch network (the pretrained DenseNet121 path is not used).
- train_model's degenerate-checkpoint retry (MAX_ATTEMPTS=2) can change seed (+1), dropout (-0.1) and epochs (×1.5) for a second attempt. It is part of the existing configuration; attempts used and the seed of the selected attempt are recorded.
- train.py does not store best_epoch or write checkpoint metadata; the runner derives best_epoch with the same improvement rule and writes the .meta.json sidecar via checkpoint_utils.
- Smoke-test V1 hashes were recorded on a CRLF (Windows) checkout; Git blobs are LF. The runner accepts a locked hash if the raw bytes or the CRLF-normalised bytes match, and records which.
- Single seed (42); no repeated runs, so run-to-run variance is not estimated.

## K. V1 integrity

- Locked-hash check before run: **PASS**; after run: **PASS**
  - `outputs/final_400/final_400.csv`: locked `cd62ce417cc29a7f…`, matched via raw
  - `outputs/splits/split_assignments.csv`: locked `70246e6e39f4694b…`, matched via raw
- Files snapshotted (final_400, splits, smoke_tests): 21; modified: [], removed: [], added: []
- Result: **V1 and smoke-test artifacts unchanged**

## L. Limitations

- Moderate Demented comes from only two independent subjects. Validation has none, so validation metrics say nothing about Moderate performance; the Moderate test result is a single-subject result (OAS1_0308) after training on one other subject (OAS1_0351).
- Slices from the same subject are not independent; all reported metrics are slice-level, not patient-level. No subject-level aggregation was performed in Phase 3.
- Validation (44 images / 27 subjects) and test (83 images / 28 subjects) are small; small numeric differences between models are not evidence of superiority.
- 2D slices from the OASIS-derived dataset only; no clinical validity is claimed.
- Test ROC-AUC is computed only because all four classes appear in the test split; it inherits the single-subject Moderate limitation.
