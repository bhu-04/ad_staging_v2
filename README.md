# Alzheimer’s Disease Staging — 2D MRI Classification

A research-oriented pipeline for four-class Alzheimer’s disease staging from 2D axial MRI slices derived from the OASIS image dataset.

> **Research status:** V1 is a locked reference experiment. V2 is an explicitly versioned experimental track intended to improve robustness, validation quality, calibration, and generalization without modifying V1.

---

## 1. Overview

This project investigates automated classification of MRI images into four staging categories:

| Class | Label |
|---|---|
| 0 | Non Demented |
| 1 | Very Mild Demented |
| 2 | Mild Demented |
| 3 | Moderate Demented |

The current experiment uses **2D axial JPEG MRI slices**, not complete 3D MRI volumes.

The project compares compact 2D deep-learning architectures and uncertainty-estimation methods while enforcing subject-level separation between training, validation, and test data.

The project is intended as a **research/experimental machine-learning system**, not a clinical diagnostic tool.

---

## 2. Research Objective

The primary objectives are to:

1. Build a reproducible four-class Alzheimer’s staging pipeline.
2. Prevent subject-level data leakage.
3. Compare multiple 2D model architectures.
4. Evaluate predictive uncertainty and calibration.
5. Investigate controlled methods for improving validation performance.
6. Evaluate performance at both image/slice and subject levels where possible.
7. Preserve a completely frozen final test evaluation.
8. Explicitly quantify the limitations caused by the small number of independent Moderate Demented subjects.

The project prioritizes:

**validity → reproducibility → generalization → per-class performance → subject-level performance → calibration → uncertainty → accuracy**

rather than optimizing a single accuracy number.

---

## 3. Dataset

The project uses the OASIS image dataset distributed through the Kaggle dataset:

`ninadaithal/imagesoasis`

The raw dataset contains four class directories:

- `non demented`
- `very mild demented`
- `mild dementia`
- `moderate dementia`

The audited raw inventory contains:

| Class | Images | Subjects |
|---|---:|---:|
| Non Demented | 67,222 | 266 |
| Very Mild Demented | 13,725 | 58 |
| Mild Demented | 5,002 | 21 |
| Moderate Demented | 488 | 2 |
| **Total** | **86,437** | **347** |

All audited raw images were readable and no filename parsing failures were found.

The raw dataset is treated as immutable by the experiment pipeline.

---

## 4. Critical Dataset Limitation

### Moderate Demented

The Moderate Demented class contains images from only **two independent subjects** in the available dataset.

This is the most important limitation of the experiment.

The locked V1 experiment contains:

- 100 Moderate Demented images
- subject `OAS1_0351`: 60 images in training
- subject `OAS1_0308`: 40 images in testing
- 0 Moderate Demented subjects in validation

Therefore:

- image count must not be interpreted as independent sample count;
- validation performance cannot directly estimate Moderate Demented performance;
- the Moderate test result is effectively a single-subject test;
- performance may be strongly affected by subject-specific characteristics;
- supplementary two-fold subject-held-out analysis is required for additional interpretation.

The project must not claim robust generalization to the Moderate Demented population from this dataset alone.

---

# 5. Locked V1 Experiment

V1 is the reference experiment and must remain unchanged.

### Locked files

```text
outputs/final_400/final_400.csv
outputs/splits/split_assignments.csv
```

### Final V1 dataset

400 images total:

- 100 Non Demented
- 100 Very Mild Demented
- 100 Mild Demented
- 100 Moderate Demented

### Subject-level split

| Split | Images | Subjects |
|---|---:|---:|
| Train | 273 | 126 |
| Validation | 44 | 27 |
| Test | 83 | 28 |

No subject occurs in more than one partition.

Class image distribution:

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Non Demented | 70 | 15 | 15 |
| Very Mild Demented | 70 | 15 | 15 |
| Mild Demented | 73 | 14 | 13 |
| Moderate Demented | 60 | 0 | 40 |

V1 is the immutable reference point for subsequent experiments.

**V2 must never silently replace V1.**

---

# 6. Data Selection

The final 400-image experiment was constructed through a deterministic selection process.

The selection hierarchy is:

```text
Class
  ↓
Subject
  ↓
Session / Scan
  ↓
Slice
```

Selection considers:

- subject diversity
- scan diversity
- slice diversity
- image quality
- quality flags
- perceptual redundancy
- adjacency/proximity
- deterministic tie-breaking

Quality flags are ranking signals rather than automatic exclusion criteria.

The Moderate class is treated specially because all available Moderate images originate from only two subjects.

---

# 7. Preprocessing

The current preprocessing pipeline is:

```text
JPEG MRI slice
      ↓
Grayscale conversion
      ↓
Resize
      ↓
Per-image zero-mean / unit-variance normalization
      ↓
Float32 tensor
      ↓
Model
```

Current preprocessing is deterministic for evaluation.

A process-level preprocessing cache is used where configured.

Augmentation is applied dynamically during training and is not cached as part of deterministic preprocessing.

---

# 8. Data Augmentation

The conservative training augmentation policy consists of:

- horizontal flip with probability 0.5
- small rotation within ±10°

Evaluation data is deterministic and receives no stochastic training augmentation.

Future augmentation experiments must be explicitly versioned and compared against the V2 baseline.

---

# 9. Models

The project currently evaluates three compact 2D architectures.

## 9.1 CNN2D

A compact convolutional network containing four convolutional blocks followed by global average pooling and classification.

It is the lightweight convolutional baseline.

## 9.2 DenseNet2D

A compact 2D DenseNet architecture using dense blocks and transition layers.

A separate torchvision DenseNet121 transfer-learning configuration is also available for controlled experiments.

## 9.3 ViT2D

A compact Vision Transformer using:

- 2D patch embedding
- learnable CLS token
- positional embeddings
- Transformer encoder layers
- classification head

Runtime positional-embedding interpolation is used to support different image resolutions.

---

# 10. Training

The training pipeline uses:

- AdamW
- class-weighted cross-entropy
- validation macro-F1 for checkpoint selection
- validation accuracy as a secondary tie-break
- validation loss as an additional tie-break
- early stopping
- deterministic/reproducible seeds
- checkpoint metadata validation

Class weighting and sampler-based balancing must not be silently combined.

Every experiment must record its configuration and split identity.

---

# 11. Uncertainty Quantification

The project contains a 3 × 3 architecture/UQ benchmark.

### Architectures

1. CNN2D
2. DenseNet2D
3. ViT2D

### UQ methods

1. MC Dropout
2. Deep Ensembles
3. Bayesian Evidential Deep Learning (BEDL)

### MC Dropout

Multiple stochastic forward passes are performed with dropout enabled during inference.

### Deep Ensembles

Multiple independently seeded models are trained and combined.

The current lightweight benchmark uses:

`M = 3`

### BEDL

The evidential model predicts Dirichlet evidence and derives uncertainty-related quantities from the resulting concentration parameters.

The BEDL implementation must be mathematically audited before strong claims about its uncertainty interpretation are made.

---

# 12. Evaluation

The evaluation framework includes:

### Discriminative metrics

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Per-class precision
- Per-class recall
- Per-class F1
- Multiclass ROC-AUC where statistically valid

### Calibration

- Expected Calibration Error (ECE)
- Brier score
- Reliability diagrams

### Uncertainty

- Predictive entropy
- Predictive variance
- Risk-coverage curves
- UQ method comparisons

### Diagnostic outputs

- Confusion matrices
- Per-class results
- Prediction distributions
- Experiment summaries

Metrics must be interpreted in the context of the subject-level structure of the data.

---

# 13. Subject-Level Evaluation

A slice is not necessarily an independent clinical observation.

Where multiple slices belong to the same subject, subject-level aggregation may be evaluated.

Potential aggregation methods must be treated as model-selection choices and therefore must be determined using validation data.

Examples include probability aggregation across available slices.

No subject-level aggregation strategy may be selected using final test performance.

---

# 14. Leakage Prevention

The following rules are mandatory:

1. A subject must never occur in multiple partitions.
2. The test set must not influence model selection.
3. Test performance must not determine hyperparameters.
4. Test performance must not determine preprocessing.
5. Test performance must not determine augmentation.
6. Test performance must not determine aggregation.
7. Test performance must not determine UQ method selection.
8. Cached checkpoints must be validated against their experiment metadata.
9. V1 files must not be overwritten.
10. V2 experiments must be stored separately.

The test set is a **final frozen evaluation set**, not a development set.

---

# 15. V1 vs V2

## V1

V1 is the locked reference experiment.

```text
outputs/
├── final_400/
└── splits/
```

V1 must remain untouched.

## V2

V2 is the controlled experimental track.

```text
outputs/v2/
├── baseline/
├── experiments/
│   ├── resolution/
│   ├── normalization/
│   ├── augmentation/
│   ├── class_balance/
│   └── pretrained_densenet/
├── model_selection/
├── subject_level/
├── uncertainty/
├── moderate_cv/
├── final_test/
└── reports/
```

V2 may improve the implementation, but every meaningful methodological change must be recorded.

---

# 16. V2 Experimental Philosophy

V2 must not be treated as:

> "Try random changes until accuracy increases."

Instead:

> **Change one controlled factor, evaluate on validation data, record the result, and retain the change only when it provides meaningful evidence of improvement without compromising generalization or methodological validity.**

Primary considerations:

1. Macro-F1
2. Per-class recall/F1
3. Subject-level performance
4. Calibration
5. Uncertainty quality
6. Accuracy

Accuracy remains important but is not the sole optimization target.

---

# 17. Reproducibility

Every experiment must record:

```text
experiment_id
configuration
random_seed
dataset identity
split hash
class mapping
image resolution
normalization
augmentation
class balancing
model architecture
pretrained status
optimizer
learning rate
batch size
epochs
best epoch
best validation metrics
checkpoint path
software/configuration information
```

Experiments must not overwrite each other's histories.

---

# 18. Recommended V2 Execution Flow

```text
Phase 1
Correctness / bug fixes
        ↓
Phase 2
Smoke tests
        ↓
Phase 3
V2 baseline
        ↓
Phase 4
Controlled validation experiments
        ↓
Phase 5
Subject-level evaluation + UQ
        ↓
Freeze selected configuration
        ↓
Phase 6
One-time frozen test evaluation
        ↓
Phase 7
Final reporting
```

---

# 19. Limitations

This project has several important limitations:

- small final dataset;
- 2D slice-level rather than 3D volumetric modeling;
- only two independent Moderate Demented subjects;
- validation contains no Moderate Demented subject in the locked V1 split;
- slice-level observations are not independent clinical observations;
- limited sample size for robust statistical inference;
- transfer-learning benefits must be empirically established;
- uncertainty estimates require careful interpretation;
- results are not clinical validation.

The system must not be presented as a clinically deployable diagnostic system.

---

# 20. Interpretation

A strong result should mean:

> The selected configuration performed well under the locked subject-level experimental protocol and demonstrated favorable validation behavior, with final performance reported on a previously untouched test set.

It should **not** mean:

> The model diagnoses Alzheimer’s disease accurately in clinical practice.

---

# 21. Research Contract

The following rules are non-negotiable:

- V1 remains immutable.
- V2 remains versioned.
- Four classes must be preserved.
- Subject-level separation must be preserved.
- No test-set tuning.
- No fabricated or selectively reported results.
- Moderate Demented's two-subject limitation must always be disclosed.
- Image-level and subject-level performance must not be conflated.
- Experiments must be reproducible.
- Every material methodological change must be recorded.
- Accuracy must not be optimized at the expense of validity.
- The README must not claim capabilities unsupported by the implementation.

If code and documentation disagree, the discrepancy must be reported and resolved explicitly rather than silently ignored.

---

# 22. Project Structure

```text
AD_staging-main/
├── data/
│   └── oasis_subset/
├── src/
│   ├── models/
│   ├── uncertainty/
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── splitting.py
│   ├── data_acquisition.py
│   └── ...
├── outputs/
│   ├── final_400/
│   ├── splits/
│   └── v2/
├── run_pipeline.py
├── README.md
└── V2_EXPERIMENT_PLAN.md
```

---

# 23. Running the Project

Always inspect the current configuration before running an experiment.

The locked experiment must be loaded from the existing final dataset and split assignments.

Do not regenerate the V1 dataset or split as part of normal V2 execution.

Recommended workflow:

```powershell
python run_pipeline.py --mode demo
```

Then perform the V2 smoke tests and controlled experiments according to:

```text
V2_EXPERIMENT_PLAN.md
```

---

# 24. Final Principle

This project is designed to answer:

> **How well can a reproducible four-class 2D MRI staging pipeline perform under strict subject-level separation and uncertainty-aware evaluation on the available OASIS-derived experiment?**

It is not designed to claim clinical diagnosis or population-level clinical generalization from the current dataset alone.
