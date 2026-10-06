# Candidate Pool pHash Redundancy Report

No near-duplicate threshold was pre-selected; results are descriptive.

## Validation

- Rows: 1,988 (expected 1,988, ok=True)
- Analyzed: 1,988; missing/invalid pHash: 0
- Pairs compared: 1,975,078
- candidate_pool.csv unmodified: True

## Hamming-distance distribution (all pairs)

Stats: min=0, p1=2, p5=4, p25=7, median=9, mean=9.556, p75=12, max=28

| hamming_distance | pairs | fraction | cumulative_pairs |
|---|---|---|---|
| 0 | 2344 | 0.001187 | 2344 |
| 1 | 3512 | 0.001778 | 5856 |
| 2 | 1.66e+04 | 0.008407 | 2.246e+04 |
| 3 | 2.479e+04 | 0.01255 | 4.725e+04 |
| 4 | 7.818e+04 | 0.03958 | 1.254e+05 |
| 5 | 8.882e+04 | 0.04497 | 2.142e+05 |
| 6 | 2.017e+05 | 0.1021 | 4.16e+05 |
| 7 | 1.624e+05 | 0.08223 | 5.784e+05 |
| 8 | 2.934e+05 | 0.1485 | 8.717e+05 |
| 9 | 1.453e+05 | 0.07355 | 1.017e+06 |
| 10 | 2.478e+05 | 0.1255 | 1.265e+06 |
| 11 | 1.23e+05 | 0.06229 | 1.388e+06 |
| 12 | 1.941e+05 | 0.09827 | 1.582e+06 |
| 13 | 8.918e+04 | 0.04515 | 1.671e+06 |
| 14 | 1.287e+05 | 0.06518 | 1.8e+06 |
| 15 | 5.214e+04 | 0.0264 | 1.852e+06 |
| 16 | 6.58e+04 | 0.03331 | 1.918e+06 |
| 17 | 1.93e+04 | 0.009773 | 1.937e+06 |
| 18 | 2.2e+04 | 0.01114 | 1.959e+06 |
| 19 | 5900 | 0.002987 | 1.965e+06 |
| 20 | 5828 | 0.002951 | 1.971e+06 |
| 21 | 1736 | 0.000879 | 1.973e+06 |
| 22 | 1678 | 0.0008496 | 1.974e+06 |
| 23 | 429 | 0.0002172 | 1.975e+06 |
| 24 | 319 | 0.0001615 | 1.975e+06 |
| 25 | 80 | 4.05e-05 | 1.975e+06 |
| 26 | 33 | 1.671e-05 | 1.975e+06 |
| 27 | 6 | 3.038e-06 | 1.975e+06 |
| 28 | 3 | 1.519e-06 | 1.975e+06 |

## Distance bins

| distance_bin | pairs | fraction_of_all_pairs | cumulative_pairs_le_hi | same_subject | cross_subject | same_session | same_scan | cross_scan | same_class | cross_class | same_subject_cross_scan | prox_adjacent(<=1) | prox_very_close(2) | prox_nearby(3-4) | prox_separated(>4) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-2 | 22461 | 0.01137 | 22461 | 5228 | 17233 | 5228 | 1110 | 21351 | 10665 | 11796 | 4118 | 3300 | 1670 | 2372 | 15119 |
| 3-4 | 102971 | 0.05214 | 125432 | 8394 | 94577 | 8394 | 1935 | 101036 | 33273 | 69698 | 6459 | 6410 | 4372 | 8483 | 83706 |
| 5-6 | 290523 | 0.1471 | 415955 | 12745 | 277778 | 12745 | 3057 | 287466 | 76516 | 214007 | 9688 | 14515 | 9805 | 19526 | 246677 |
| 7-8 | 455770 | 0.2308 | 871725 | 15718 | 440052 | 15718 | 3878 | 451892 | 109358 | 346412 | 11840 | 21970 | 14648 | 28453 | 390699 |
| 9-10 | 393074 | 0.199 | 1264799 | 12524 | 380550 | 12524 | 3173 | 389901 | 97609 | 295465 | 9351 | 18810 | 12146 | 23913 | 338205 |
| 11-15 | 587167 | 0.2973 | 1851966 | 11898 | 575269 | 11898 | 2948 | 584219 | 138907 | 448260 | 8950 | 26571 | 17254 | 33784 | 509558 |
| 16+ | 123112 | 0.06233 | 1975078 | 635 | 122477 | 635 | 152 | 122960 | 26750 | 96362 | 483 | 5803 | 3536 | 6827 | 106946 |

## Pairs with distance <= 10 (1,264,799)

| same_subject | cross_subject | same_scan | cross_scan | same_class | cross_class |
|---|---|---|---|---|---|
| 54609 | 1210190 | 13153 | 1251646 | 327421 | 937378 |

## Groups (direct vs transitive) by threshold

| threshold | groups_ge2 | images_in_groups | largest_group | singleton_images | direct_pairs | transitive_chain_groups | clique_groups |
|---|---|---|---|---|---|---|---|
| 2 | 88 | 1767 | 1523 | 221 | 22461 | 18 | 70 |
| 4 | 18 | 1952 | 1912 | 36 | 125432 | 4 | 14 |
| 6 | 5 | 1980 | 1971 | 8 | 415955 | 1 | 4 |
| 8 | 1 | 1988 | 1988 | 0 | 871725 | 1 | 0 |
| 10 | 1 | 1988 | 1988 | 0 | 1264799 | 1 | 0 |

Clique = every member pair directly within threshold; transitive_chain = connected only via intermediate images.

## Slice proximity (all pairs)

| scope | group_type | group | total_pairs | bin_0-2 | bin_3-4 | bin_5-6 | bin_7-8 | bin_9-10 | bin_11-15 | bin_16+ | pairs_le10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all_pairs | proximity | adjacent(<=1) | 97379 | 3300 | 6410 | 14515 | 21970 | 18810 | 26571 | 5803 | 65005 |
| all_pairs | proximity | very_close(2) | 63431 | 1670 | 4372 | 9805 | 14648 | 12146 | 17254 | 3536 | 42641 |
| all_pairs | proximity | nearby(3-4) | 123358 | 2372 | 8483 | 19526 | 28453 | 23913 | 33784 | 6827 | 82747 |
| all_pairs | proximity | separated(>4) | 1690910 | 15119 | 83706 | 246677 | 390699 | 338205 | 509558 | 106946 | 1074406 |

## Slice proximity (same-scan pairs)

| scope | group_type | group | total_pairs | bin_0-2 | bin_3-4 | bin_5-6 | bin_7-8 | bin_9-10 | bin_11-15 | bin_16+ | pairs_le10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| same_scan_pairs | proximity | adjacent(<=1) | 480 | 398 | 78 | 4 | 0 | 0 | 0 | 0 | 480 |
| same_scan_pairs | proximity | very_close(2) | 473 | 239 | 184 | 45 | 5 | 0 | 0 | 0 | 473 |
| same_scan_pairs | proximity | nearby(3-4) | 951 | 246 | 400 | 228 | 61 | 15 | 1 | 0 | 950 |
| same_scan_pairs | proximity | separated(>4) | 14349 | 227 | 1273 | 2780 | 3812 | 3158 | 2947 | 152 | 11250 |

## Warnings

- none