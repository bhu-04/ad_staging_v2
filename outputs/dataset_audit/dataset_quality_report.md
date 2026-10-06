# OASIS Dataset Quality Report

## Dataset Overview

- Dataset root: `C:\Users\bhuva\OneDrive\Documents\AD_staging-main\data\oasis_subset`
- Total images: **86,437**
- Readable images: **86,437**
- Unreadable images: **0**
- Filename parse failures: **0**
- Unique subjects: **347**
- Quality flagged images: **3,014**
- Exact duplicate groups: **0**

## Class Distribution and Quality

|   class_id | class_name         |   total_images |   readable_images |   unreadable_images |   quality_flagged_images |   unique_subjects |   median_quality |   mean_quality |   median_sharpness |   median_contrast |   median_foreground_ratio |
|-----------:|:-------------------|---------------:|------------------:|--------------------:|-------------------------:|------------------:|-----------------:|---------------:|-------------------:|------------------:|--------------------------:|
|          0 | Non Demented       |          67222 |             67222 |                   0 |                     2324 |               266 |         0.491082 |       0.48589  |            212.039 |           45.2546 |                  0.594872 |
|          1 | Very Mild Demented |          13725 |             13725 |                   0 |                      453 |                58 |         0.482849 |       0.485051 |            222.56  |           44.567  |                  0.589206 |
|          2 | Mild Demented      |           5002 |              5002 |                   0 |                      216 |                21 |         0.496676 |       0.49156  |            225.993 |           44.575  |                  0.590197 |
|          3 | Moderate Demented  |            488 |               488 |                   0 |                       21 |                 2 |         0.521134 |       0.515448 |            237.724 |           44.7131 |                  0.596701 |

## Subject-Level Dataset

|   class_id | class_name         |   subjects |   total_images |   median_images_per_subject |   max_images_per_subject |
|-----------:|:-------------------|-----------:|---------------:|----------------------------:|-------------------------:|
|          0 | Non Demented       |        266 |          67222 |                         244 |                      488 |
|          1 | Very Mild Demented |         58 |          13725 |                         244 |                      244 |
|          2 | Mild Demented      |         21 |           5002 |                         244 |                      244 |
|          3 | Moderate Demented  |          2 |            488 |                         244 |                      244 |

## Slice Distribution

|   class_id | class_name         |   unique_slice_positions |   minimum_slice |   median_slice |   maximum_slice |
|-----------:|:-------------------|-------------------------:|----------------:|---------------:|----------------:|
|          0 | Non Demented       |                       61 |             100 |            130 |             160 |
|          1 | Very Mild Demented |                       61 |             100 |            130 |             160 |
|          2 | Mild Demented      |                       61 |             100 |            130 |             160 |
|          3 | Moderate Demented  |                       61 |             100 |            130 |             160 |

## Methodology

### Quality analysis
- Quality scores are continuous ranking signals, not clinical labels.
- Extreme quality flags use robust class-wise percentile thresholds.
- Images are not automatically deleted because of a quality flag.

### Duplicate analysis
- Exact duplicates are identified using SHA-256.
- Perceptual hashes are stored for later redundancy analysis.
- Full-dataset perceptual near-duplicate comparison is intentionally deferred.
- Near-duplicate detection will be performed on the candidate pool used for the final 400-image selection.

### Data integrity
- Raw dataset files were not modified.
- Raw dataset files were not deleted.
- No augmentation was performed.
- No train/validation/test split was performed.