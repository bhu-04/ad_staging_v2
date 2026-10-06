# Final 400-image Selection Report

Deterministic greedy per-class selection, RANDOM_SEED=42; exactly 100 per class. No split, augmentation or training performed. Candidate pool unmodified.

## Scoring

`score = 4.0*subject_gain + 2.0*scan_gain + 1.5*slice_gain + 1.0*quality_pct - 0.3*quality_flag - 1.2*phash_penalty - 1.5*adjacency_penalty`

- subject_gain = 1/(1+images already picked from that subject)
- scan_gain = 0.6/(1+picked from same subject-session-scan) + 0.4*(1 - share of picks with same session/scan label)
- slice_gain = 1/(1+sum exp(-(dSlice/2)^2) over picked) -> spreads picks over slice positions
- quality_pct = within-class percentile of quality_score (not top-N)
- phash_penalty = max over picked of w(d), w(d)=1/(1+exp((d-3.5)/1)): soft, no hard threshold
- adjacency_penalty = same-subject slice proximity vs picked (<=1:1.0, 2:0.6, 3-4:0.3; x0.5 if different scan)
- Hard rules only: class quota, per-subject cap, no byte-identical (sha256) duplicate.

pHash penalty w(d): d=0:0.97, d=2:0.82, d=3:0.62, d=4:0.38, d=5:0.18, d=6:0.08, d=8:0.01

Justification (candidate pool, all pairs): <=2 = 1.14%, <=4 = 6.35%, <=6 = 21.06% of pairs, median distance 9; components merge fully at <=8, so distances >=7 are ordinary MRI similarity and are not penalised.

Per-subject caps: Non Demented=1, Very Mild Demented=3, Mild Demented=8, Moderate Demented=60

## Class summary

| class_name | class_id | pool_images | selected | pool_subjects | selected_subjects | min_imgs_per_subject | median_imgs_per_subject | max_imgs_per_subject | pool_scans | selected_scans | selected_scan_labels | pool_unique_slices | selected_unique_slices | mean_quality | median_quality | pool_mean_quality | quality_flagged_selected | pairs_le2 | pairs_le4 | pairs_le6 | pool_pairs_le2 | pool_pairs_le4 | pool_pairs_le6 | adjacent_same_scan_pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Non Demented | 0 | 500 | 100 | 266 | 100 | 1 | 1 | 1 | 500 | 100 | 8 | 61 | 53 | 0.7414 | 0.733 | 0.6789 | 5 | 0 | 54 | 298 | 3563 | 15487 | 37230 | 0 |
| Very Mild Demented | 1 | 500 | 100 | 58 | 58 | 1 | 2 | 3 | 225 | 100 | 4 | 61 | 54 | 0.6559 | 0.6456 | 0.6216 | 2 | 1 | 69 | 393 | 1176 | 6605 | 24392 | 0 |
| Mild Demented | 2 | 500 | 100 | 21 | 21 | 3 | 5 | 6 | 82 | 80 | 4 | 61 | 51 | 0.6138 | 0.6181 | 0.5798 | 3 | 4 | 75 | 369 | 1510 | 7205 | 21554 | 0 |
| Moderate Demented | 3 | 488 | 100 | 2 | 2 | 40 | 50 | 60 | 8 | 8 | 4 | 61 | 56 | 0.5725 | 0.5798 | 0.5154 | 0 | 98 | 463 | 1195 | 4416 | 14641 | 37278 | 0 |

## Redundancy: final 400 vs candidate pool (ALL = across classes; class rows = within class)

| dataset | scope | n_images | n_pairs | pairs_le2 | pairs_le4 | pairs_le6 | frac_le2 | frac_le4 | frac_le6 | nn_median | same_subject_pairs_le4 | adjacent_same_scan_pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| candidate_pool | ALL | 1988 | 1975078 | 22461 | 125432 | 415955 | 0.01137 | 0.06351 | 0.2106 | 1 | 13622 | 480 |
| final_400 | ALL | 400 | 79800 | 209 | 1712 | 7280 | 0.002619 | 0.02145 | 0.09123 | 3 | 418 | 0 |
| candidate_pool | Non Demented | 500 | 124750 | 3563 | 15487 | 37230 | 0.02856 | 0.1241 | 0.2984 | 1 | 167 | 0 |
| final_400 | Non Demented | 100 | 4950 | 0 | 54 | 298 | 0 | 0.01091 | 0.0602 | 5 | 0 | 0 |
| candidate_pool | Very Mild Demented | 500 | 124750 | 1176 | 6605 | 24392 | 0.009427 | 0.05295 | 0.1955 | 1 | 688 | 0 |
| final_400 | Very Mild Demented | 100 | 4950 | 1 | 69 | 393 | 0.000202 | 0.01394 | 0.07939 | 5 | 2 | 0 |
| candidate_pool | Mild Demented | 500 | 124750 | 1510 | 7205 | 21554 | 0.0121 | 0.05776 | 0.1728 | 1 | 1479 | 0 |
| final_400 | Mild Demented | 100 | 4950 | 4 | 75 | 369 | 0.0008081 | 0.01515 | 0.07455 | 5 | 17 | 0 |
| candidate_pool | Moderate Demented | 488 | 118828 | 4416 | 14641 | 37278 | 0.03716 | 0.1232 | 0.3137 | 0 | 11288 | 480 |
| final_400 | Moderate Demented | 100 | 4950 | 98 | 463 | 1195 | 0.0198 | 0.09354 | 0.2414 | 2 | 399 | 0 |

## Moderate Demented subject counts

{'OAS1_0308': 40, 'OAS1_0351': 60}

## Slice-region distribution (final)

| class_name | 100-109 | 110-119 | 120-129 | 130-139 | 140-149 | 150-160 |
|---|---|---|---|---|---|---|
| Mild Demented | 25 | 18 | 22 | 14 | 12 | 9 |
| Moderate Demented | 21 | 16 | 16 | 17 | 18 | 12 |
| Non Demented | 28 | 18 | 15 | 14 | 11 | 14 |
| Very Mild Demented | 21 | 25 | 16 | 17 | 9 | 12 |

## Warnings

- none