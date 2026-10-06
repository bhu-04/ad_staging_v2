# OASIS Candidate Pool Report

> **This is a candidate pool, not the final 400-image dataset.**  
> **Near-duplicate analysis is intentionally deferred until the candidate-pool stage.**

## Input inventory

- File: `C:\Users\bhuva\OneDrive\Documents\AD_staging-main\outputs\dataset_audit\image_inventory.csv`
- SHA-256: `bed822c4d6d91b064f5641444dc270c0f41d3d036e9312a13eb0a306450e346a`
- Total records: **86,437**
- Eligible records: **86,437**
- Dataset source: data/oasis_subset (audited 2D OASIS axial JPEG slices; raw files are not read or copied by this stage)

## Target vs actual counts

| class_name | target | actual |
|---|---|---|
| Non Demented | 500 | 500 |
| Very Mild Demented | 500 | 500 |
| Mild Demented | 500 | 500 |
| Moderate Demented | 488 | 488 |

Total selected: **1,988** (target 1,988)

## Exclusions

- None (no unusable records found in the inventory).

## Subject diversity

| class_name | unique_subjects | min_per_subject | median_per_subject | max_per_subject |
|---|---|---|---|---|
| Mild Demented | 266 | 23 | 24.0000 | 24 |
| Moderate Demented | 58 | 244 | 244.0000 | 244 |
| Non Demented | 21 | 1 | 2.0000 | 2 |
| Very Mild Demented | 2 | 8 | 9.0000 | 9 |

## Scan / session diversity

| class_name | unique_sessions | unique_scans | unique_subject_sessions | unique_subject_scans |
|---|---|---|---|---|
| Non Demented | 2 | 6 | 266 | 500 |
| Very Mild Demented | 1 | 4 | 58 | 225 |
| Mild Demented | 1 | 4 | 21 | 82 |
| Moderate Demented | 1 | 4 | 2 | 8 |

Selected images per session/scan:

| class_name | session | scan | selected_count |
|---|---|---|---|
| Non Demented | MR1 | mpr-1 | 116 |
| Non Demented | MR1 | mpr-2 | 116 |
| Non Demented | MR1 | mpr-3 | 115 |
| Non Demented | MR1 | mpr-4 | 115 |
| Non Demented | MR1 | mpr-5 | 1 |
| Non Demented | MR1 | mpr-6 | 1 |
| Non Demented | MR2 | mpr-1 | 9 |
| Non Demented | MR2 | mpr-2 | 9 |
| Non Demented | MR2 | mpr-3 | 9 |
| Non Demented | MR2 | mpr-4 | 9 |
| Very Mild Demented | MR1 | mpr-1 | 125 |
| Very Mild Demented | MR1 | mpr-2 | 125 |
| Very Mild Demented | MR1 | mpr-3 | 125 |
| Very Mild Demented | MR1 | mpr-4 | 125 |
| Mild Demented | MR1 | mpr-1 | 129 |
| Mild Demented | MR1 | mpr-2 | 129 |
| Mild Demented | MR1 | mpr-3 | 129 |
| Mild Demented | MR1 | mpr-4 | 113 |
| Moderate Demented | MR1 | mpr-1 | 122 |
| Moderate Demented | MR1 | mpr-2 | 122 |
| Moderate Demented | MR1 | mpr-3 | 122 |
| Moderate Demented | MR1 | mpr-4 | 122 |

## Slice diversity

| class_name | 100-109 | 110-119 | 120-129 | 130-139 | 140-149 | 150-160 |
|---|---|---|---|---|---|---|
| Non Demented | 92 | 84 | 83 | 83 | 80 | 78 |
| Very Mild Demented | 98 | 97 | 99 | 77 | 74 | 55 |
| Mild Demented | 94 | 95 | 92 | 75 | 78 | 66 |
| Moderate Demented | 80 | 80 | 80 | 80 | 80 | 88 |

| class_name | unique_slice_positions |
|---|---|
| Non Demented | 61 |
| Very Mild Demented | 61 |
| Mild Demented | 61 |
| Moderate Demented | 61 |

## Quality statistics

| class_name | quality_flagged_count | mean_quality_score | median_quality_score | min_quality_score | max_quality_score |
|---|---|---|---|---|---|
| Non Demented | 48 | 0.6789 | 0.6679 | 0.4706 | 0.9902 |
| Very Mild Demented | 16 | 0.6216 | 0.6157 | 0.4327 | 0.8731 |
| Mild Demented | 23 | 0.5798 | 0.5853 | 0.3368 | 0.8619 |
| Moderate Demented | 21 | 0.5154 | 0.5211 | 0.1960 | 0.8061 |

## Selection methodology

- Hierarchy: class > subject > session > scan > slice
- Score = 0.5 x normalized_quality + 0.2 x slice_diversity + 0.15 x scan_diversity + 0.15 x subject_balance - neighbor_penalty (max 0.3, radius 10 slices)
- Slice regions: 100-109, 110-119, 120-129, 130-139, 140-149, 150-160
- Subject quotas (even split, capped) -> scan quotas (even split, class-level scan balancing) -> round-robin slice picks scored by the weighted formula; ties broken by filepath. Quality flags never exclude images. Exact class counts take priority over score.
- Random seed: **42** (no randomness is consumed; ties are broken by filepath, so repeated runs are identical)

## Validation

- PASS: total == 1988
- PASS: Non Demented == 500
- PASS: Very Mild Demented == 500
- PASS: Mild Demented == 500
- PASS: Moderate Demented == 488
- PASS: no duplicate filepath
- PASS: no missing filepath
- PASS: all records readable
- PASS: no UNKNOWN subject_id
- PASS: no filename parse failures
- PASS: no exact duplicate groups
- PASS: all four classes represented
- PASS: all Moderate Demented subjects represented
- PASS: Moderate Demented has 2 subjects (audit)
- PASS: no duplicate (class, subject, session, scan, slice_idx)
- PASS: all eligible Moderate Demented files included

Overall: **PASSED**

## Warnings / limitations

- Moderate Demented comes from only 2 subject(s); the pool necessarily contains every slice of those subjects, many of them adjacent. A subject-level train/val/test split cannot place this class in all three partitions, which is a data limitation to resolve at the final-selection/splitting stage.
- Quality flags are ranking information only: 108 quality-flagged image(s) are in the pool (all kept for Moderate Demented; down-weighted by score elsewhere).
- Exact-duplicate status relies on the audit's SHA-256 results; perceptual near-duplicates have NOT been checked.
