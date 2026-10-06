# Split report
**Strategy:** subject-level, class-stratified (seeded shuffle, seed=42); Moderate assigned deterministically (more-image subject->train, other->test)

## Counts
| class_name         |   train_images |   val_images |   test_images |   train_subjects |   val_subjects |   test_subjects |
|:-------------------|---------------:|-------------:|--------------:|-----------------:|---------------:|----------------:|
| Non Demented       |             70 |           15 |            15 |               70 |             15 |              15 |
| Very Mild Demented |             70 |           15 |            15 |               40 |              9 |               9 |
| Mild Demented      |             73 |           14 |            13 |               15 |              3 |               3 |
| Moderate Demented  |             60 |            0 |            40 |                1 |              0 |               1 |
| TOTAL              |            273 |           44 |            83 |              126 |             27 |              28 |

## Moderate Demented
- OAS1_0308: test (40 images)
- OAS1_0351: train (60 images)
- fold1: held out OAS1_0308, train ['OAS1_0351']
- fold2: held out OAS1_0351, train ['OAS1_0308']

## Validation
- n_images_400: True
- 100_per_class: True
- all_filepaths_once: True
- no_duplicate_filepath: True
- no_duplicate_class_subject_session_scan_slice: True
- no_subject_in_multiple_partitions: True
- all_four_classes_present: True
- split_values_valid: True
- moderate_subjects_exactly_once: True
- ALL_PASSED: True
- deterministic_rerun_identical: True
- final_400_unchanged: True

## Limitations
- Moderate Demented has only 2 subjects: no conventional 3-way subject-independent split exists for it.
- Validation contains NO Moderate images; test Moderate comes from a single subject (OAS1_0308), so Moderate test metrics are single-subject and high-variance.
- Moderate train comes from a single subject: the model sees little Moderate anatomical variability.
- Val-based model selection/early stopping/calibration cannot assess Moderate.
- Image ratios deviate from 70/15/15; subject independence prioritised.
- Moderate 2-fold CV is supplementary; each fold tests on one subject only. Other classes keep main-split assignments.