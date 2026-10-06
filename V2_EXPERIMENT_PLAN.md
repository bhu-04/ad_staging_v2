# V2_EXPERIMENT_PLAN.md

# Experiment V2 — Controlled Improvement & Validation Protocol

**Status:** Experimental protocol  
**Reference:** Locked V1  
**Primary objective:** Improve model robustness and useful performance without compromising experimental validity.

---

# 1. Purpose

Experiment V2 is the controlled continuation of the locked V1 Alzheimer’s disease staging experiment.

V2 exists to investigate:

- correctness;
- reproducibility;
- model performance;
- preprocessing choices;
- resolution;
- augmentation;
- class balancing;
- transfer learning;
- subject-level aggregation;
- uncertainty estimation;
- calibration;
- per-class behavior.

V2 must never modify or overwrite the V1 experiment.

---

# 2. Non-Negotiable Rules

## Rule 1 — V1 is immutable

Do not modify:

```text
outputs/final_400/final_400.csv
outputs/splits/split_assignments.csv
```

Do not regenerate them during normal V2 execution.

If a new dataset or split is ever required, it must receive a new experiment/version identity.

---

## Rule 2 — No test-set model selection

The test set is completely frozen.

The test set must not be used to choose:

- architecture;
- resolution;
- normalization;
- augmentation;
- optimizer;
- learning rate;
- class balancing;
- pretrained status;
- model checkpoint;
- aggregation method;
- UQ method;
- threshold;
- ensemble size.

All such choices must be finalized using training/validation evidence.

---

## Rule 3 — Record every experiment

Every experiment must have a unique identifier.

Example:

```text
V2-B0
V2-R96
V2-R128
V2-NORM-A
V2-AUG-B
V2-BAL-C
V2-DN121
```

The exact naming scheme may be implemented by Cline, but experiment identity must never be ambiguous.

---

# 3. Selection Criteria

The primary model-selection metric is:

```text
Validation Macro-F1
```

Secondary considerations:

1. Per-class recall/F1
2. Subject-level validation behavior
3. Calibration
4. Validation accuracy
5. Validation loss

Accuracy must not be used as the sole selection criterion.

A configuration that increases accuracy while substantially damaging minority-class recall or calibration should not automatically be selected.

---

# 4. Dataset

V2 uses the locked V1 dataset:

```text
outputs/final_400/final_400.csv
```

and locked split:

```text
outputs/splits/split_assignments.csv
```

Dataset size:

```text
400 images
100 images per class
```

Split:

```text
Train: 273 images
Validation: 44 images
Test: 83 images
```

No subject overlap is permitted.

---

# 5. Moderate Demented Handling

The Moderate class contains only two independent subjects.

Locked V1 assignment:

```text
OAS1_0351 → Train
OAS1_0308 → Test
```

Validation contains zero Moderate subjects.

Therefore:

- do not tune Moderate-specific parameters using validation;
- do not pretend validation Macro-F1 measures Moderate performance;
- report Moderate test performance cautiously;
- run supplementary two-fold subject-held-out Moderate analysis where supported.

### Supplementary CV

Fold 1:

```text
OAS1_0308 → held out
OAS1_0351 → training
```

Fold 2:

```text
OAS1_0351 → held out
OAS1_0308 → training
```

This analysis is supplementary and must not overwrite the locked V1 protocol.

---

# 6. Phase 2 — Smoke Tests

Before any accuracy experiment, verify the complete software path.

## 6.1 Dataset tests

Verify:

- 400 images load;
- four classes exist;
- 100 images per class;
- all files exist;
- no duplicate filepaths;
- no subject overlap;
- split assignments load correctly.

## 6.2 Tensor tests

Verify expected tensor shape:

```text
[B, 1, H, W]
```

for every planned resolution.

## 6.3 Model forward tests

Run:

- CNN2D
- DenseNet2D
- ViT2D

Verify:

```text
output shape = [B, 4]
```

## 6.4 Backward tests

For every architecture:

1. create a small batch;
2. forward;
3. compute loss;
4. backward;
5. optimizer step.

No NaN/Inf values may occur.

## 6.5 Checkpoint tests

Verify:

- save;
- load;
- metadata;
- split hash;
- class count;
- resolution;
- configuration compatibility.

## 6.6 Augmentation tests

Verify augmentation is:

- applied only to training;
- stochastic;
- not cached;
- absent from validation/test.

## 6.7 V1 integrity test

Hash and verify:

```text
outputs/final_400/final_400.csv
outputs/splits/split_assignments.csv
```

before and after smoke testing.

They must remain unchanged.

---

# 7. Phase 3 — V2 Baseline

Establish a clean baseline before changing anything.

Baseline configuration must be explicitly recorded.

Evaluate:

```text
CNN2D
DenseNet2D
ViT2D
```

using the same:

- split;
- resolution;
- normalization;
- augmentation;
- optimizer;
- class balancing;
- evaluation protocol.

Record:

- best epoch;
- validation accuracy;
- validation macro-F1;
- per-class precision;
- per-class recall;
- per-class F1;
- validation loss;
- ECE where applicable;
- Brier score where applicable;
- confusion matrix.

Do not inspect the final test results for the purpose of selecting the baseline.

---

# 8. Phase 4 — Controlled Experiments

Experiments must be performed in controlled groups.

Do not simultaneously modify multiple methodological factors unless the experiment is explicitly defined as a factorial/combined experiment.

---

## 8.1 Resolution Experiment

Candidate resolutions may include:

```text
96 × 96
128 × 128
160 × 160
```

subject to computational feasibility.

Keep everything else fixed.

Record:

- validation Macro-F1;
- per-class metrics;
- training stability;
- calibration;
- runtime;
- memory requirements.

Select resolution using validation evidence only.

---

## 8.2 Normalization Experiment

Compare the current:

```text
per-image zero-mean / unit-variance
```

against carefully controlled alternatives supported by the implementation.

Do not assume a normalization method is better because it is common elsewhere.

The purpose is to determine whether the current normalization removes useful image-intensity information.

---

## 8.3 Augmentation Experiment

Start from the baseline augmentation:

```text
horizontal flip
rotation ±10°
```

Potential controlled alternatives may include:

- no augmentation;
- baseline augmentation;
- slightly stronger geometric augmentation.

Avoid medically implausible transformations.

Do not introduce arbitrary image transformations merely to improve validation accuracy.

---

## 8.4 Class-Balancing Experiment

Compare supported balancing strategies.

Examples:

```text
class-weighted loss
sampler-based balancing
```

Do not combine both unless a separate experiment explicitly evaluates that combination.

Evaluate minority-class recall and macro-F1 carefully.

---

## 8.5 Pretrained DenseNet Experiment

Evaluate the pretrained DenseNet121 pathway separately.

Verify:

- correct 1-channel adaptation;
- correct pretrained weight loading;
- intended frozen/unfrozen layers;
- BatchNorm behavior;
- checkpoint metadata;
- reproducibility.

Compare against compact DenseNet2D.

Do not assume pretrained transfer learning improves performance on this dataset.

---

# 9. Experiment Registry

Every experiment should generate a machine-readable record.

Minimum fields:

```text
experiment_id
parent_experiment
timestamp
seed
dataset_hash
split_hash
architecture
resolution
normalization
augmentation
class_balance
pretrained
optimizer
learning_rate
batch_size
epochs
best_epoch
best_val_accuracy
best_val_macro_f1
val_macro_precision
val_macro_recall
per_class_metrics
ece
brier
checkpoint
status
```

The exact serialization format may be JSON/CSV.

Do not overwrite previous experiment records.

---

# 10. Phase 5 — Subject-Level Evaluation

After the strongest validation configurations have been identified, evaluate subject-level behavior.

Where a subject has multiple slices, aggregate slice predictions using a method selected without looking at test performance.

Possible aggregation:

```text
mean class probability across slices
```

Additional aggregation methods may be investigated as controlled experiments.

Report:

- number of subjects;
- subject-level accuracy;
- subject-level macro-F1;
- subject-level confusion matrix;
- per-class subject metrics where estimable.

Never describe slice-level performance as patient-level performance.

---

# 11. Phase 5 — Uncertainty Benchmark

Run the planned 3 × 3 benchmark:

```text
             MC Dropout   Ensemble   BEDL
CNN2D             ✓           ✓         ✓
DenseNet2D        ✓           ✓         ✓
ViT2D             ✓           ✓         ✓
```

For each combination evaluate:

- accuracy;
- macro-F1;
- ECE;
- Brier;
- predictive entropy;
- predictive variance;
- risk-coverage behavior.

Deep Ensemble benchmark currently uses:

```text
M = 3
```

Treat this as a lightweight benchmark rather than definitive evidence about ensemble asymptotics.

---

# 12. BEDL Audit

Before making strong claims about BEDL:

Verify:

1. evidence activation;
2. Dirichlet parameter construction;
3. expected probability calculation;
4. loss;
5. KL regularization;
6. annealing;
7. class weighting;
8. uncertainty interpretation;
9. checkpoint criterion.

Document the exact implementation used.

Avoid using terms such as "Bayesian posterior" unless the implementation and terminology genuinely support that claim.

---

# 13. Statistical Comparison

The test set must not be used for iterative statistical model selection.

Validation comparisons may be used to select the final configuration.

Because the data are clustered by subject:

- image-level observations should not automatically be treated as independent subjects;
- statistical tests must account for the effective subject structure where possible;
- results with very small subject counts must be reported descriptively.

Multiple-comparison effects should be considered when comparing many configurations.

Do not claim statistical significance simply because one validation accuracy is numerically higher.

---

# 14. Model Selection Freeze

Before Phase 6, freeze:

```text
architecture
checkpoint
resolution
normalization
augmentation
class balancing
optimizer/training configuration
aggregation method
UQ method
ensemble size
any threshold
```

Create a final model-selection record:

```text
selected_experiment_id
selection_metric
selection_reason
alternative_experiments
selection_timestamp
```

After this point, these decisions must not be changed based on test results.

---

# 15. Phase 6 — Final Frozen Test Evaluation

Only after the model is frozen:

1. Load the selected checkpoint.
2. Load the untouched test partition.
3. Run the final evaluation once.
4. Save the complete test report.
5. Save predictions.
6. Save probabilities.
7. Save uncertainty values.
8. Save confusion matrix.
9. Save per-class metrics.
10. Save subject-level metrics where applicable.

The test set is then considered consumed for the final V2 result.

If the test result is disappointing:

**do not tune against it.**

Start a new explicitly defined experiment/version if further research is required.

---

# 16. Final Test Report

Minimum report:

## Overall

- accuracy
- macro precision
- macro recall
- macro F1
- ROC-AUC if valid
- ECE
- Brier score

## Per class

- precision
- recall
- F1
- support

## Subject level

- subject count
- subject accuracy
- subject macro-F1
- per-class subject metrics where estimable

## Uncertainty

- entropy
- variance
- risk-coverage
- calibration

## Moderate class

Explicitly state:

- only two independent subjects exist;
- which subject was used for training;
- which subject was used for final test;
- validation contains no Moderate subject;
- results therefore have substantial subject-specific uncertainty.

---

# 17. Reporting Rules

Never report only the best number.

Every final report must include:

```text
dataset
split
model
configuration
validation result
test result
per-class result
subject-level result
calibration
uncertainty
limitations
```

If an experiment fails, record it as failed.

Do not silently remove failed experiments from the experiment history.

---

# 18. Output Directory

Recommended V2 structure:

```text
outputs/v2/
├── smoke_tests/
├── baseline/
│   ├── cnn2d/
│   ├── densenet2d/
│   └── vit2d/
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

Each experiment should have its own checkpoint, configuration, metrics, predictions, and logs where appropriate.

---

# 19. What Cline Must Not Do

Cline must not:

- overwrite V1;
- regenerate the locked 400 silently;
- regenerate the locked split silently;
- tune on the test set;
- choose the final model from test accuracy;
- hide failed experiments;
- delete inconvenient results;
- fabricate missing metrics;
- call image-level results patient-level results;
- claim clinical validation;
- add medically implausible augmentation;
- combine class weighting and sampling without an explicit experiment;
- overwrite training histories from other experiments;
- reuse an incompatible checkpoint;
- silently change the class mapping;
- silently change the dataset;
- silently change the evaluation protocol.

---

# 20. Cline Execution Protocol

Cline should execute V2 in stages.

### Stage A

Implement/verify Phase 2 smoke tests.

Then STOP.

Return:

```text
SMOKE TEST REPORT

Dataset:
Split:
V1 integrity:
CNN2D:
DenseNet2D:
ViT2D:
96×96:
Other resolution(s):
Backward pass:
Checkpoint:
Augmentation:
Failures:
Warnings:
Files changed:
```

Do not proceed automatically.

### Stage B

After smoke tests pass, run V2 baseline.

Then STOP and report the complete baseline table.

### Stage C

Run controlled Phase 4 experiments.

Do not change multiple factors without recording the experimental design.

### Stage D

Perform subject-level and UQ analysis.

### Stage E

Freeze the selected configuration.

### Stage F

Perform one final frozen test evaluation.

### Stage G

Generate final reports.

---

# 21. Definition of Success

V2 is successful if it provides:

- reproducible execution;
- no V1 contamination;
- no subject leakage;
- transparent experiment tracking;
- improved or better-supported validation performance;
- improved per-class behavior where possible;
- meaningful subject-level analysis;
- calibrated uncertainty evaluation;
- a genuinely frozen final test evaluation.

A higher accuracy number alone is **not sufficient** to declare V2 successful.

---

# 22. Final Research Principle

The objective is not:

> "Find a configuration that produces the highest accuracy on the available test set."

The objective is:

> **Find a reproducible, validation-selected, uncertainty-aware model configuration that generalizes as well as the available data allow, while transparently reporting the severe limitations of the dataset.**

---

# 23. Current Execution State

At the start of V2:

```text
V1:
LOCKED

Phase 1:
COMPLETED

Phase 2:
PENDING

Phase 3:
PENDING

Phase 4:
PENDING

Phase 5:
PENDING

Phase 6:
PENDING

Phase 7:
PENDING
```

The next action is therefore:

**Run Phase 2 smoke tests only.**
