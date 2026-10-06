#!/usr/bin/env python3

"""
OASIS Dataset Quality Audit
===========================

Purpose
-------
Perform a read-only audit of the full OASIS 2D MRI JPEG dataset before
constructing the final 400-image experimental subset.

Current stage performs:

1. Complete image inventory
2. 4-class label verification
3. Filename / subject metadata extraction
4. Image readability / corruption detection
5. Image dimensions
6. Intensity statistics
7. Foreground estimate
8. Sharpness estimate
9. Contrast estimate
10. Perceptual hash calculation
11. SHA-256 exact duplicate detection
12. Robust quality scoring
13. Quality outlier flagging
14. Subject statistics
15. Slice-position statistics
16. Class-wise quality statistics
17. Visual QA samples
18. JSON / CSV / Markdown reports

IMPORTANT
---------
This script does NOT:

- delete images
- modify images
- augment images
- create the final 400-image subset
- split train/validation/test data
- perform expensive full-dataset near-duplicate comparison

Near-duplicate analysis will be performed later on a much smaller
candidate pool.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from PIL import Image, ImageFile, ImageDraw


# ---------------------------------------------------------------------
# IMPORT CONFIG
# ---------------------------------------------------------------------
#
# IMPORTANT:
# This script is executed as:
#
#     python src\dataset_quality.py
#
# Therefore we import config directly rather than:
#
#     from src import config
#
# ---------------------------------------------------------------------

import config


# ---------------------------------------------------------------------
# PIL configuration
# ---------------------------------------------------------------------

ImageFile.LOAD_TRUNCATED_IMAGES = False


# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

# Percentiles used for robust quality analysis.
LOW_PERCENTILE = 1
HIGH_PERCENTILE = 99

# Quality score weights.
QUALITY_WEIGHTS = {
    "sharpness": 0.35,
    "contrast": 0.25,
    "foreground_ratio": 0.20,
    "intensity_information": 0.20,
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def safe_float(value: Any) -> Optional[float]:
    """
    Convert a value to float safely.
    """
    if value is None:
        return None

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return None


def robust_minmax(
    value: float,
    low: float,
    high: float,
) -> float:
    """
    Robustly normalize a value to [0, 1].
    """

    if value is None:
        return 0.0

    if not np.isfinite(value):
        return 0.0

    if high <= low:
        return 0.5

    return float(
        np.clip(
            (value - low) / (high - low),
            0.0,
            1.0,
        )
    )


def percentile_summary(
    values: pd.Series,
) -> Dict[str, Optional[float]]:
    """
    Calculate useful percentiles for a numeric series.
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return {
            "p1": None,
            "p5": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "p99": None,
        }

    return {
        "p1": float(np.percentile(values, 1)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


# ---------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------

class DatasetQualityAnalyzer:

    def __init__(
        self,
        data_root: Optional[Path] = None,
    ):

        # -------------------------------------------------------------
        # Dataset path
        # -------------------------------------------------------------

        if data_root is None:

            self.data_root = (
                config.PROJECT_ROOT
                / "data"
                / "oasis_subset"
            )

        else:

            self.data_root = Path(
                data_root
            )

        self.data_root = (
            self.data_root
            .resolve()
        )

        # -------------------------------------------------------------
        # Dataset configuration
        # -------------------------------------------------------------

        self.raw_folder_to_class = (
            config.RAW_FOLDER_TO_CLASS
        )

        self.class_names = (
            config.CLASS_NAMES
        )

        self.class_to_idx = (
            config.CLASS_TO_IDX
        )

        # -------------------------------------------------------------
        # Output paths
        # -------------------------------------------------------------

        self.output_root = (
            config.PROJECT_ROOT
            / "outputs"
            / "dataset_audit"
        )

        self.visualization_root = (
            self.output_root
            / "visualizations"
        )

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.visualization_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -------------------------------------------------------------
        # Inventory
        # -------------------------------------------------------------

        self.inventory: List[
            Dict[str, Any]
        ] = []

        self.quality_thresholds = {}


    # =================================================================
    # FILENAME PARSING
    # =================================================================

    @staticmethod
    def parse_filename(
        filename: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Parse OASIS filename.

        Example:

        OAS1_0003_MR1_mpr-2_129.jpg

        Extracts:

        subject_id = OAS1_0003
        session    = MR1
        scan       = mpr-2
        slice_idx  = 129
        """

        pattern = re.compile(
            r"(?P<subject_id>OAS\d+_\d+)"
            r"_(?P<session>MR\d+)"
            r"_(?P<scan>mpr-\d+)"
            r"_(?P<slice_idx>\d+)"
        )

        match = pattern.search(
            filename
        )

        if not match:
            return None

        result = match.groupdict()

        result["slice_idx"] = int(
            result["slice_idx"]
        )

        return result


    # =================================================================
    # PERCEPTUAL HASH
    # =================================================================

    @staticmethod
    def compute_phash(
        image: Image.Image,
    ) -> str:
        """
        Compute a lightweight perceptual hash.

        This is calculated during the inventory pass so that the
        information is available later.

        IMPORTANT:
        We do NOT perform the expensive near-duplicate comparison
        here.
        """

        image = image.convert("L")

        image = image.resize(
            (8, 8),
            Image.Resampling.LANCZOS,
        )

        arr = np.asarray(
            image,
            dtype=np.float32,
        )

        median = float(
            np.median(arr)
        )

        bits = (
            arr >= median
        )

        value = 0

        for bit in bits.flatten():

            value = (
                value << 1
            ) | int(bit)

        return f"{value:016x}"


    # =================================================================
    # IMAGE PROCESSING WORKER
    # =================================================================

    @staticmethod
    def process_single_image(
        args: Tuple[
            Path,
            str,
            str,
            Dict[str, int],
            Path,
        ]
    ) -> Dict[str, Any]:

        (
            image_path,
            raw_folder,
            class_name,
            class_to_idx,
            data_root,
        ) = args

        # -------------------------------------------------------------
        # Parse filename
        # -------------------------------------------------------------

        metadata = (
            DatasetQualityAnalyzer
            .parse_filename(
                image_path.name
            )
        )

        filename_parse_failed = (
            metadata is None
        )

        if metadata is None:

            metadata = {
                "subject_id": "UNKNOWN",
                "session": "UNKNOWN",
                "scan": "UNKNOWN",
                "slice_idx": -1,
            }

        # -------------------------------------------------------------
        # Initialize record
        # -------------------------------------------------------------

        try:

            relative_path = str(
                image_path.relative_to(
                    data_root.parent
                )
            )

        except Exception:

            relative_path = str(
                image_path
            )

        record = {

            # ---------------------------------------------------------
            # File information
            # ---------------------------------------------------------

            "filepath": relative_path,
            "abs_filepath": str(
                image_path
            ),
            "filename": image_path.name,
            "raw_folder": raw_folder,

            # ---------------------------------------------------------
            # Class
            # ---------------------------------------------------------

            "class_name": class_name,
            "class_id": class_to_idx[
                class_name
            ],

            # ---------------------------------------------------------
            # Subject metadata
            # ---------------------------------------------------------

            "subject_id": metadata[
                "subject_id"
            ],

            "session": metadata[
                "session"
            ],

            "scan": metadata[
                "scan"
            ],

            "slice_idx": metadata[
                "slice_idx"
            ],

            "filename_parse_failed":
                filename_parse_failed,

            # ---------------------------------------------------------
            # File properties
            # ---------------------------------------------------------

            "file_size": None,

            "width": None,
            "height": None,
            "channels": None,

            # ---------------------------------------------------------
            # Readability
            # ---------------------------------------------------------

            "is_readable": False,
            "read_error": None,

            # ---------------------------------------------------------
            # Image statistics
            # ---------------------------------------------------------

            "mean_intensity": None,
            "std_intensity": None,
            "min_intensity": None,
            "max_intensity": None,

            "nonzero_ratio": None,
            "foreground_ratio": None,

            # ---------------------------------------------------------
            # Quality metrics
            # ---------------------------------------------------------

            "sharpness": None,
            "contrast": None,

            "quality_score": None,
            "quality_flag": False,
            "quality_reasons": "",

            # ---------------------------------------------------------
            # Duplicate information
            # ---------------------------------------------------------

            "sha256": None,
            "exact_duplicate_group": None,

            "phash": None,
            "near_duplicate_group": None,
        }

        # -------------------------------------------------------------
        # File size
        # -------------------------------------------------------------

        try:

            record["file_size"] = (
                image_path.stat().st_size
            )

        except Exception:

            record["file_size"] = None

        # -------------------------------------------------------------
        # SHA-256
        # -------------------------------------------------------------

        try:

            with open(
                image_path,
                "rb",
            ) as file:

                file_bytes = (
                    file.read()
                )

            record["sha256"] = (
                hashlib.sha256(
                    file_bytes
                ).hexdigest()
            )

        except Exception as exc:

            record["read_error"] = (
                f"SHA256 error: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            return record

        # -------------------------------------------------------------
        # Open image
        # -------------------------------------------------------------

        try:

            with Image.open(
                image_path
            ) as original:

                # Force complete decoding.
                original.load()

                record["width"] = (
                    original.width
                )

                record["height"] = (
                    original.height
                )

                # -----------------------------------------------------
                # Channel information
                # -----------------------------------------------------

                if original.mode in (
                    "1",
                    "L",
                    "I",
                    "F",
                ):

                    record["channels"] = 1

                elif original.mode == "RGB":

                    record["channels"] = 3

                elif original.mode == "RGBA":

                    record["channels"] = 4

                else:

                    record["channels"] = None

                # -----------------------------------------------------
                # Grayscale analysis
                # -----------------------------------------------------

                image = original.convert(
                    "L"
                )

                arr = np.asarray(
                    image,
                    dtype=np.float32,
                )

                if arr.size == 0:

                    raise ValueError(
                        "Image contains zero pixels."
                    )

                # -----------------------------------------------------
                # Intensity statistics
                # -----------------------------------------------------

                record[
                    "mean_intensity"
                ] = float(
                    np.mean(arr)
                )

                record[
                    "std_intensity"
                ] = float(
                    np.std(arr)
                )

                record[
                    "min_intensity"
                ] = float(
                    np.min(arr)
                )

                record[
                    "max_intensity"
                ] = float(
                    np.max(arr)
                )

                # -----------------------------------------------------
                # Nonzero ratio
                # -----------------------------------------------------

                record[
                    "nonzero_ratio"
                ] = float(
                    np.count_nonzero(
                        arr
                    )
                    / arr.size
                )

                # -----------------------------------------------------
                # Foreground estimate
                #
                # This is NOT brain segmentation.
                #
                # It is only a simple indicator of how much of the
                # image contains meaningful non-background intensity.
                # -----------------------------------------------------

                p5 = float(
                    np.percentile(
                        arr,
                        5,
                    )
                )

                p95 = float(
                    np.percentile(
                        arr,
                        95,
                    )
                )

                if p95 > p5:

                    threshold = (
                        p5
                        + 0.10
                        * (
                            p95 - p5
                        )
                    )

                    record[
                        "foreground_ratio"
                    ] = float(
                        np.mean(
                            arr > threshold
                        )
                    )

                else:

                    record[
                        "foreground_ratio"
                    ] = 0.0

                # -----------------------------------------------------
                # Sharpness
                #
                # Variance of a discrete Laplacian.
                # Higher generally means more high-frequency detail.
                # -----------------------------------------------------

                if (
                    arr.shape[0] > 2
                    and arr.shape[1] > 2
                ):

                    center = (
                        arr[1:-1, 1:-1]
                    )

                    laplacian = (
                        arr[1:-1, :-2]
                        +
                        arr[1:-1, 2:]
                        +
                        arr[:-2, 1:-1]
                        +
                        arr[2:, 1:-1]
                        -
                        4.0 * center
                    )

                    record[
                        "sharpness"
                    ] = float(
                        np.var(
                            laplacian
                        )
                    )

                else:

                    record[
                        "sharpness"
                    ] = 0.0

                # -----------------------------------------------------
                # Contrast
                # -----------------------------------------------------

                record[
                    "contrast"
                ] = record[
                    "std_intensity"
                ]

                # -----------------------------------------------------
                # Perceptual hash
                # -----------------------------------------------------

                record[
                    "phash"
                ] = (
                    DatasetQualityAnalyzer
                    .compute_phash(
                        image
                    )
                )

                # -----------------------------------------------------
                # Successful read
                # -----------------------------------------------------

                record[
                    "is_readable"
                ] = True

        except Exception as exc:

            record[
                "is_readable"
            ] = False

            record[
                "read_error"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        return record


    # =================================================================
    # BUILD INVENTORY
    # =================================================================

    def build_inventory(self) -> None:

        print(
            f"Building inventory from "
            f"{self.data_root}..."
        )

        tasks = []

        # -------------------------------------------------------------
        # Discover all images
        # -------------------------------------------------------------

        for (
            raw_folder,
            class_name,
        ) in self.raw_folder_to_class.items():

            folder_path = (
                self.data_root
                / raw_folder
            )

            if not folder_path.is_dir():

                print(
                    f"WARNING: Missing folder: "
                    f"{folder_path}"
                )

                continue

            image_files = []

            for path in folder_path.rglob("*"):

                if (
                    path.is_file()
                    and path.suffix.lower()
                    in IMAGE_EXTENSIONS
                ):

                    image_files.append(
                        path.resolve()
                    )

            image_files = sorted(
                set(image_files)
            )

            print(
                f"  Found "
                f"{len(image_files):,} "
                f"unique images in "
                f"{raw_folder} "
                f"-> "
                f"{class_name}"
            )

            for image_path in image_files:

                tasks.append(
                    (
                        image_path,
                        raw_folder,
                        class_name,
                        self.class_to_idx,
                        self.data_root,
                    )
                )

        # -------------------------------------------------------------
        # Process images
        # -------------------------------------------------------------

        print(
            f"Processing "
            f"{len(tasks):,} "
            f"images across workers..."
        )

        cpu_count = (
            os.cpu_count()
            or 8
        )

        # Avoid excessive disk contention.
        max_workers = min(
            20,
            max(
                4,
                cpu_count,
            ),
        )

        with ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:

            self.inventory = list(
                executor.map(
                    self.process_single_image,
                    tasks,
                )
            )

        print(
            f"Total images inventoried: "
            f"{len(self.inventory):,}"
        )


    # =================================================================
    # EXACT DUPLICATES
    # =================================================================

    def detect_exact_duplicates(
        self,
    ) -> None:

        print(
            "Detecting exact duplicates via SHA-256..."
        )

        hash_groups = defaultdict(
            list
        )

        for record in self.inventory:

            sha = record.get(
                "sha256"
            )

            if sha:

                hash_groups[
                    sha
                ].append(record)

        duplicate_group_id = 0
        redundant_files = 0

        for (
            sha,
            records,
        ) in hash_groups.items():

            if len(records) <= 1:
                continue

            for record in records:

                record[
                    "exact_duplicate_group"
                ] = duplicate_group_id

            redundant_files += (
                len(records) - 1
            )

            duplicate_group_id += 1

        print(
            f"Found "
            f"{duplicate_group_id:,} "
            f"exact duplicate groups "
            f"("
            f"{redundant_files:,} "
            f"redundant files)."
        )


    # =================================================================
    # QUALITY THRESHOLDS
    # =================================================================

    def calculate_quality_thresholds(
        self,
    ) -> None:

        print(
            "Calculating robust quality thresholds..."
        )

        df = pd.DataFrame(
            self.inventory
        )

        readable = df[
            df["is_readable"]
            == True
        ].copy()

        self.quality_thresholds = {}

        metrics = [
            "sharpness",
            "contrast",
            "foreground_ratio",
            "std_intensity",
            "mean_intensity",
        ]

        for class_name in (
            self.class_names
        ):

            class_df = readable[
                readable[
                    "class_name"
                ]
                == class_name
            ]

            class_thresholds = {}

            for metric in metrics:

                class_thresholds[
                    metric
                ] = percentile_summary(
                    class_df[
                        metric
                    ]
                )

            self.quality_thresholds[
                class_name
            ] = class_thresholds

        output_path = (
            self.output_root
            / "quality_thresholds.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.quality_thresholds,
                file,
                indent=2,
            )

        print(
            f"Saved thresholds: "
            f"{output_path}"
        )


    # =================================================================
    # QUALITY SCORES
    # =================================================================

    def calculate_quality_scores(
        self,
    ) -> None:

        print(
            "Calculating image quality scores..."
        )

        df = pd.DataFrame(
            self.inventory
        )

        readable = df[
            df["is_readable"]
            == True
        ]

        ranges = {}

        metrics = [
            "sharpness",
            "contrast",
            "foreground_ratio",
            "std_intensity",
        ]

        # -------------------------------------------------------------
        # Robust global ranges
        # -------------------------------------------------------------

        for metric in metrics:

            values = pd.to_numeric(
                readable[
                    metric
                ],
                errors="coerce",
            ).dropna()

            if values.empty:

                ranges[
                    metric
                ] = (
                    0.0,
                    1.0,
                )

            else:

                low = float(
                    np.percentile(
                        values,
                        LOW_PERCENTILE,
                    )
                )

                high = float(
                    np.percentile(
                        values,
                        HIGH_PERCENTILE,
                    )
                )

                ranges[
                    metric
                ] = (
                    low,
                    high,
                )

        # -------------------------------------------------------------
        # Calculate score
        # -------------------------------------------------------------

        for record in self.inventory:

            if not record[
                "is_readable"
            ]:

                record[
                    "quality_score"
                ] = 0.0

                continue

            sharpness_score = (
                robust_minmax(
                    record[
                        "sharpness"
                    ],
                    *ranges[
                        "sharpness"
                    ],
                )
            )

            contrast_score = (
                robust_minmax(
                    record[
                        "contrast"
                    ],
                    *ranges[
                        "contrast"
                    ],
                )
            )

            foreground_score = (
                robust_minmax(
                    record[
                        "foreground_ratio"
                    ],
                    *ranges[
                        "foreground_ratio"
                    ],
                )
            )

            intensity_score = (
                robust_minmax(
                    record[
                        "std_intensity"
                    ],
                    *ranges[
                        "std_intensity"
                    ],
                )
            )

            score = (

                QUALITY_WEIGHTS[
                    "sharpness"
                ]
                * sharpness_score

                +

                QUALITY_WEIGHTS[
                    "contrast"
                ]
                * contrast_score

                +

                QUALITY_WEIGHTS[
                    "foreground_ratio"
                ]
                * foreground_score

                +

                QUALITY_WEIGHTS[
                    "intensity_information"
                ]
                * intensity_score
            )

            record[
                "quality_score"
            ] = float(
                np.clip(
                    score,
                    0.0,
                    1.0,
                )
            )


    # =================================================================
    # QUALITY FLAGS
    # =================================================================

    def flag_quality_outliers(
        self,
    ) -> None:

        print(
            "Flagging extreme quality outliers..."
        )

        df = pd.DataFrame(
            self.inventory
        )

        for class_name in (
            self.class_names
        ):

            class_mask = (
                (df["class_name"] == class_name)
                &
                (df["is_readable"] == True)
            )

            class_indices = (
                df.index[class_mask]
            )

            if len(class_indices) == 0:
                continue

            class_df = df.loc[
                class_indices
            ]

            def get_percentile(
                metric: str,
                percentile: float,
            ):

                values = pd.to_numeric(
                    class_df[
                        metric
                    ],
                    errors="coerce",
                ).dropna()

                if values.empty:
                    return None

                return float(
                    np.percentile(
                        values,
                        percentile,
                    )
                )

            p1_sharpness = (
                get_percentile(
                    "sharpness",
                    1,
                )
            )

            p1_contrast = (
                get_percentile(
                    "contrast",
                    1,
                )
            )

            p1_foreground = (
                get_percentile(
                    "foreground_ratio",
                    1,
                )
            )

            p1_std = (
                get_percentile(
                    "std_intensity",
                    1,
                )
            )

            p1_mean = (
                get_percentile(
                    "mean_intensity",
                    1,
                )
            )

            p99_mean = (
                get_percentile(
                    "mean_intensity",
                    99,
                )
            )

            # ---------------------------------------------------------
            # Apply flags
            # ---------------------------------------------------------

            for idx in class_indices:

                record = (
                    self.inventory[
                        idx
                    ]
                )

                reasons = []

                if (
                    p1_sharpness is not None
                    and record[
                        "sharpness"
                    ] is not None
                    and record[
                        "sharpness"
                    ] < p1_sharpness
                ):

                    reasons.append(
                        "extremely_low_sharpness"
                    )

                if (
                    p1_contrast is not None
                    and record[
                        "contrast"
                    ] is not None
                    and record[
                        "contrast"
                    ] < p1_contrast
                ):

                    reasons.append(
                        "extremely_low_contrast"
                    )

                if (
                    p1_foreground is not None
                    and record[
                        "foreground_ratio"
                    ] is not None
                    and record[
                        "foreground_ratio"
                    ] < p1_foreground
                ):

                    reasons.append(
                        "extremely_low_foreground"
                    )

                if (
                    p1_std is not None
                    and record[
                        "std_intensity"
                    ] is not None
                    and record[
                        "std_intensity"
                    ] < p1_std
                ):

                    reasons.append(
                        "extremely_low_information"
                    )

                if (
                    p1_mean is not None
                    and record[
                        "mean_intensity"
                    ] is not None
                    and record[
                        "mean_intensity"
                    ] < p1_mean
                ):

                    reasons.append(
                        "extremely_dark"
                    )

                if (
                    p99_mean is not None
                    and record[
                        "mean_intensity"
                    ] is not None
                    and record[
                        "mean_intensity"
                    ] > p99_mean
                ):

                    reasons.append(
                        "extremely_bright"
                    )

                if record[
                    "filename_parse_failed"
                ]:

                    reasons.append(
                        "filename_parse_failed"
                    )

                record[
                    "quality_reasons"
                ] = ";".join(
                    reasons
                )

                record[
                    "quality_flag"
                ] = (
                    len(reasons) > 0
                )

        # -------------------------------------------------------------
        # Unreadable images
        # -------------------------------------------------------------

        for record in self.inventory:

            if not record[
                "is_readable"
            ]:

                record[
                    "quality_flag"
                ] = True

                record[
                    "quality_reasons"
                ] = "unreadable_image"

                record[
                    "quality_score"
                ] = 0.0


    # =================================================================
    # SUBJECT STATISTICS
    # =================================================================

    def build_subject_statistics(
        self,
    ) -> pd.DataFrame:

        df = pd.DataFrame(
            self.inventory
        )

        valid = df[
            df["subject_id"]
            != "UNKNOWN"
        ]

        if valid.empty:
            return pd.DataFrame()

        grouped = (
            valid
            .groupby(
                [
                    "class_id",
                    "class_name",
                    "subject_id",
                ]
            )
            .agg(
                image_count=(
                    "filepath",
                    "count",
                ),

                readable_images=(
                    "is_readable",
                    "sum",
                ),

                median_quality=(
                    "quality_score",
                    "median",
                ),

                min_slice=(
                    "slice_idx",
                    "min",
                ),

                max_slice=(
                    "slice_idx",
                    "max",
                ),
            )
            .reset_index()
        )

        return grouped


    # =================================================================
    # SLICE STATISTICS
    # =================================================================

    def build_slice_statistics(
        self,
    ) -> pd.DataFrame:

        df = pd.DataFrame(
            self.inventory
        )

        valid = df[
            df["slice_idx"]
            >= 0
        ]

        if valid.empty:
            return pd.DataFrame()

        grouped = (
            valid
            .groupby(
                [
                    "class_id",
                    "class_name",
                    "slice_idx",
                ]
            )
            .size()
            .reset_index(
                name="image_count"
            )
        )

        # -------------------------------------------------------------
        # Percentage within each class
        # -------------------------------------------------------------

        grouped[
            "class_percentage"
        ] = (
            grouped
            .groupby(
                "class_name"
            )["image_count"]
            .transform(
                lambda x:
                100.0
                * x
                / x.sum()
            )
        )

        return grouped


    # =================================================================
    # CLASS QUALITY SUMMARY
    # =================================================================

    def build_class_quality_summary(
        self,
    ) -> pd.DataFrame:

        df = pd.DataFrame(
            self.inventory
        )

        rows = []

        for class_name in (
            self.class_names
        ):

            class_df = df[
                df["class_name"]
                == class_name
            ]

            readable = class_df[
                class_df["is_readable"]
                == True
            ]

            unique_subjects = (
                class_df[
                    class_df[
                        "subject_id"
                    ]
                    != "UNKNOWN"
                ]["subject_id"]
                .nunique()
            )

            row = {

                "class_id":
                    self.class_to_idx[
                        class_name
                    ],

                "class_name":
                    class_name,

                "total_images":
                    len(class_df),

                "readable_images":
                    len(readable),

                "unreadable_images":
                    len(class_df)
                    - len(readable),

                "quality_flagged_images":
                    int(
                        class_df[
                            "quality_flag"
                        ].sum()
                    ),

                "unique_subjects":
                    unique_subjects,

                "median_quality":
                    safe_float(
                        readable[
                            "quality_score"
                        ].median()
                    ),

                "mean_quality":
                    safe_float(
                        readable[
                            "quality_score"
                        ].mean()
                    ),

                "median_sharpness":
                    safe_float(
                        readable[
                            "sharpness"
                        ].median()
                    ),

                "median_contrast":
                    safe_float(
                        readable[
                            "contrast"
                        ].median()
                    ),

                "median_foreground_ratio":
                    safe_float(
                        readable[
                            "foreground_ratio"
                        ].median()
                    ),
            }

            rows.append(row)

        return pd.DataFrame(
            rows
        )


    # =================================================================
    # SAVE INVENTORY
    # =================================================================

    def save_inventory(
        self,
    ) -> pd.DataFrame:

        df = pd.DataFrame(
            self.inventory
        )

        output_path = (
            self.output_root
            / "image_inventory.csv"
        )

        df.to_csv(
            output_path,
            index=False,
        )

        print(
            f"Saved inventory: "
            f"{output_path}"
        )

        return df


    # =================================================================
    # SAVE REPORTS
    # =================================================================

    def save_reports(
        self,
        subject_stats: pd.DataFrame,
        slice_stats: pd.DataFrame,
        class_summary: pd.DataFrame,
    ) -> None:

        df = pd.DataFrame(
            self.inventory
        )

        # -------------------------------------------------------------
        # Class summary
        # -------------------------------------------------------------

        class_summary.to_csv(
            self.output_root
            / "class_quality_summary.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # Subject statistics
        # -------------------------------------------------------------

        subject_stats.to_csv(
            self.output_root
            / "subject_statistics.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # Slice statistics
        # -------------------------------------------------------------

        slice_stats.to_csv(
            self.output_root
            / "slice_distribution.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # Exact duplicate summary
        # -------------------------------------------------------------

        exact = df[
            df[
                "exact_duplicate_group"
            ].notna()
        ]

        if exact.empty:

            exact_summary = pd.DataFrame(
                columns=[
                    "exact_duplicate_group",
                    "class_name",
                    "subject_id",
                    "filepath",
                    "sha256",
                ]
            )

        else:

            exact_summary = exact[
                [
                    "exact_duplicate_group",
                    "class_name",
                    "subject_id",
                    "filepath",
                    "sha256",
                ]
            ]

        exact_summary.to_csv(
            self.output_root
            / "duplicate_summary.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # Near duplicate placeholder
        # -------------------------------------------------------------
        #
        # IMPORTANT:
        # Full dataset near-duplicate analysis is intentionally
        # deferred until we construct a candidate pool.
        #

        near_duplicate_placeholder = (
            pd.DataFrame(
                columns=[
                    "filepath_a",
                    "filepath_b",
                    "class_name",
                    "subject_a",
                    "subject_b",
                    "slice_a",
                    "slice_b",
                    "phash_distance",
                ]
            )
        )

        near_duplicate_placeholder.to_csv(
            self.output_root
            / "near_duplicate_summary.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # Quality distribution
        # -------------------------------------------------------------

        readable = df[
            df["is_readable"]
            == True
        ]

        distribution = {}

        for class_name in (
            self.class_names
        ):

            class_df = readable[
                readable[
                    "class_name"
                ]
                == class_name
            ]

            distribution[
                class_name
            ] = {

                "quality_score":
                    percentile_summary(
                        class_df[
                            "quality_score"
                        ]
                    ),

                "sharpness":
                    percentile_summary(
                        class_df[
                            "sharpness"
                        ]
                    ),

                "contrast":
                    percentile_summary(
                        class_df[
                            "contrast"
                        ]
                    ),

                "foreground_ratio":
                    percentile_summary(
                        class_df[
                            "foreground_ratio"
                        ]
                    ),

                "mean_intensity":
                    percentile_summary(
                        class_df[
                            "mean_intensity"
                        ]
                    ),

                "std_intensity":
                    percentile_summary(
                        class_df[
                            "std_intensity"
                        ]
                    ),
            }

        with open(
            self.output_root
            / "quality_distribution.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                distribution,
                file,
                indent=2,
            )

        # -------------------------------------------------------------
        # Machine-readable report
        # -------------------------------------------------------------

        report = {

            "dataset_root":
                str(
                    self.data_root
                ),

            "total_images":
                len(df),

            "readable_images":
                int(
                    df[
                        "is_readable"
                    ].sum()
                ),

            "unreadable_images":
                int(
                    (
                        ~df[
                            "is_readable"
                        ]
                    ).sum()
                ),

            "filename_parse_failures":
                int(
                    df[
                        "filename_parse_failed"
                    ].sum()
                ),

            "unique_subjects":
                int(
                    df[
                        df[
                            "subject_id"
                        ]
                        != "UNKNOWN"
                    ]["subject_id"]
                    .nunique()
                ),

            "quality_flagged_images":
                int(
                    df[
                        "quality_flag"
                    ].sum()
                ),

            "exact_duplicate_groups":
                int(
                    df[
                        df[
                            "exact_duplicate_group"
                        ].notna()
                    ][
                        "exact_duplicate_group"
                    ].nunique()
                ),

            "near_duplicate_analysis":
                "Deferred until candidate-pool stage.",

            "class_summary":
                class_summary.to_dict(
                    orient="records"
                ),
        }

        with open(
            self.output_root
            / "dataset_quality_report.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=2,
            )

        # -------------------------------------------------------------
        # Markdown report
        # -------------------------------------------------------------

        self.write_markdown_report(
            report,
            class_summary,
            subject_stats,
            slice_stats,
        )


    # =================================================================
    # MARKDOWN REPORT
    # =================================================================

    def write_markdown_report(
        self,
        report: Dict[str, Any],
        class_summary: pd.DataFrame,
        subject_stats: pd.DataFrame,
        slice_stats: pd.DataFrame,
    ) -> None:

        lines = []

        lines.append(
            "# OASIS Dataset Quality Report"
        )

        lines.append("")

        lines.append(
            "## Dataset Overview"
        )

        lines.append("")

        lines.append(
            f"- Dataset root: "
            f"`{self.data_root}`"
        )

        lines.append(
            f"- Total images: "
            f"**{report['total_images']:,}**"
        )

        lines.append(
            f"- Readable images: "
            f"**{report['readable_images']:,}**"
        )

        lines.append(
            f"- Unreadable images: "
            f"**{report['unreadable_images']:,}**"
        )

        lines.append(
            f"- Filename parse failures: "
            f"**{report['filename_parse_failures']:,}**"
        )

        lines.append(
            f"- Unique subjects: "
            f"**{report['unique_subjects']:,}**"
        )

        lines.append(
            f"- Quality flagged images: "
            f"**{report['quality_flagged_images']:,}**"
        )

        lines.append(
            f"- Exact duplicate groups: "
            f"**{report['exact_duplicate_groups']:,}**"
        )

        lines.append("")

        # -------------------------------------------------------------
        # Class summary
        # -------------------------------------------------------------

        lines.append(
            "## Class Distribution and Quality"
        )

        lines.append("")

        lines.append(
            class_summary.to_markdown(
                index=False
            )
        )

        lines.append("")

        # -------------------------------------------------------------
        # Subject information
        # -------------------------------------------------------------

        lines.append(
            "## Subject-Level Dataset"
        )

        lines.append("")

        if subject_stats.empty:

            lines.append(
                "No subject-level statistics were available."
            )

        else:

            subject_summary = (
                subject_stats
                .groupby(
                    [
                        "class_id",
                        "class_name",
                    ]
                )
                .agg(
                    subjects=(
                        "subject_id",
                        "nunique",
                    ),

                    total_images=(
                        "image_count",
                        "sum",
                    ),

                    median_images_per_subject=(
                        "image_count",
                        "median",
                    ),

                    max_images_per_subject=(
                        "image_count",
                        "max",
                    ),
                )
                .reset_index()
            )

            lines.append(
                subject_summary.to_markdown(
                    index=False
                )
            )

        lines.append("")

        # -------------------------------------------------------------
        # Slice information
        # -------------------------------------------------------------

        lines.append(
            "## Slice Distribution"
        )

        lines.append("")

        if slice_stats.empty:

            lines.append(
                "No slice statistics were available."
            )

        else:

            slice_summary = (
                slice_stats
                .groupby(
                    [
                        "class_id",
                        "class_name",
                    ]
                )
                .agg(
                    unique_slice_positions=(
                        "slice_idx",
                        "nunique",
                    ),

                    minimum_slice=(
                        "slice_idx",
                        "min",
                    ),

                    median_slice=(
                        "slice_idx",
                        "median",
                    ),

                    maximum_slice=(
                        "slice_idx",
                        "max",
                    ),
                )
                .reset_index()
            )

            lines.append(
                slice_summary.to_markdown(
                    index=False
                )
            )

        lines.append("")

        # -------------------------------------------------------------
        # Methodology
        # -------------------------------------------------------------

        lines.append(
            "## Methodology"
        )

        lines.append("")

        lines.append(
            "### Quality analysis"
        )

        lines.append(
            "- Quality scores are continuous ranking "
            "signals, not clinical labels."
        )

        lines.append(
            "- Extreme quality flags use robust "
            "class-wise percentile thresholds."
        )

        lines.append(
            "- Images are not automatically deleted "
            "because of a quality flag."
        )

        lines.append("")

        lines.append(
            "### Duplicate analysis"
        )

        lines.append(
            "- Exact duplicates are identified using "
            "SHA-256."
        )

        lines.append(
            "- Perceptual hashes are stored for later "
            "redundancy analysis."
        )

        lines.append(
            "- Full-dataset perceptual near-duplicate "
            "comparison is intentionally deferred."
        )

        lines.append(
            "- Near-duplicate detection will be performed "
            "on the candidate pool used for the final "
            "400-image selection."
        )

        lines.append("")

        lines.append(
            "### Data integrity"
        )

        lines.append(
            "- Raw dataset files were not modified."
        )

        lines.append(
            "- Raw dataset files were not deleted."
        )

        lines.append(
            "- No augmentation was performed."
        )

        lines.append(
            "- No train/validation/test split was "
            "performed."
        )

        output_path = (
            self.output_root
            / "dataset_quality_report.md"
        )

        output_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        print(
            f"Saved report: "
            f"{output_path}"
        )


    # =================================================================
    # VISUAL QA
    # =================================================================

    def create_visual_qa(
        self,
    ) -> None:

        print(
            "Generating visual QA..."
        )

        df = pd.DataFrame(
            self.inventory
        )

        readable = df[
            df["is_readable"]
            == True
        ].copy()

        if readable.empty:

            print(
                "No readable images available "
                "for visual QA."
            )

            return

        # -------------------------------------------------------------
        # Random sample per class
        # -------------------------------------------------------------

        for class_name in (
            self.class_names
        ):

            class_df = readable[
                readable[
                    "class_name"
                ]
                == class_name
            ]

            if class_df.empty:
                continue

            count = min(
                12,
                len(class_df),
            )

            sampled = class_df.sample(
                n=count,
                random_state=RANDOM_SEED,
            )

            filename = (
                "random_"
                + class_name
                .lower()
                .replace(
                    " ",
                    "_",
                )
                + ".jpg"
            )

            self.make_grid(
                sampled,
                self.visualization_root
                / filename,
                class_name,
            )

        # -------------------------------------------------------------
        # Lowest quality
        # -------------------------------------------------------------

        lowest = (
            readable
            .sort_values(
                "quality_score"
            )
            .head(20)
        )

        self.make_grid(
            lowest,
            self.visualization_root
            / "lowest_quality.jpg",
            "Lowest Quality Candidates",
        )

        # -------------------------------------------------------------
        # Highest quality
        # -------------------------------------------------------------

        highest = (
            readable
            .sort_values(
                "quality_score",
                ascending=False,
            )
            .head(20)
        )

        self.make_grid(
            highest,
            self.visualization_root
            / "highest_quality.jpg",
            "Highest Quality Candidates",
        )

        print(
            "Visual QA grids generated."
        )


    # =================================================================
    # IMAGE GRID
    # =================================================================

    def make_grid(
        self,
        records: pd.DataFrame,
        output_path: Path,
        title: str,
    ) -> None:

        if records.empty:
            return

        records = records.head(
            20
        )

        tile_width = 190
        tile_height = 180
        label_height = 45

        columns = 5

        rows = math.ceil(
            len(records)
            / columns
        )

        canvas = Image.new(
            "RGB",
            (
                columns
                * tile_width,

                rows
                * (
                    tile_height
                    + label_height
                ),
            ),
            "white",
        )

        draw = ImageDraw.Draw(
            canvas
        )

        for position, (
            _,
            row,
        ) in enumerate(
            records.iterrows()
        ):

            try:

                with Image.open(
                    row[
                        "abs_filepath"
                    ]
                ) as image:

                    image = image.convert(
                        "L"
                    )

                    image.thumbnail(
                        (
                            tile_width
                            - 10,

                            tile_height
                            - 10,
                        )
                    )

                    x = (
                        position
                        % columns
                    ) * tile_width

                    y = (
                        position
                        // columns
                    ) * (
                        tile_height
                        + label_height
                    )

                    tile = Image.new(
                        "RGB",
                        (
                            tile_width,
                            tile_height,
                        ),
                        "white",
                    )

                    offset_x = (
                        tile_width
                        - image.width
                    ) // 2

                    offset_y = (
                        tile_height
                        - image.height
                    ) // 2

                    tile.paste(
                        image.convert(
                            "RGB"
                        ),
                        (
                            offset_x,
                            offset_y,
                        ),
                    )

                    canvas.paste(
                        tile,
                        (
                            x,
                            y,
                        ),
                    )

                    quality = (
                        row[
                            "quality_score"
                        ]
                    )

                    if pd.isna(
                        quality
                    ):

                        quality = 0.0

                    label = (
                        f"{row['subject_id']} "
                        f"s{row['slice_idx']} "
                        f"q={quality:.2f}"
                    )

                    draw.text(
                        (
                            x + 4,
                            y
                            + tile_height
                            + 4,
                        ),
                        label,
                        fill="black",
                    )

            except Exception:

                continue

        canvas.save(
            output_path,
            quality=90,
        )


    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    def print_final_summary(
        self,
    ) -> None:

        df = pd.DataFrame(
            self.inventory
        )

        print("")
        print(
            "=" * 72
        )

        print(
            "DATASET QUALITY ANALYSIS COMPLETE"
        )

        print(
            "=" * 72
        )

        print("")

        print(
            f"Total images: "
            f"{len(df):,}"
        )

        print(
            f"Readable images: "
            f"{int(df['is_readable'].sum()):,}"
        )

        print(
            f"Unreadable images: "
            f"{int((~df['is_readable']).sum()):,}"
        )

        print(
            f"Filename parse failures: "
            f"{int(df['filename_parse_failed'].sum()):,}"
        )

        print(
            f"Quality flagged images: "
            f"{int(df['quality_flag'].sum()):,}"
        )

        print(
            f"Unique subjects: "
            f"{df[df['subject_id'] != 'UNKNOWN']['subject_id'].nunique():,}"
        )

        print(
            f"Exact duplicate groups: "
            f"{df['exact_duplicate_group'].notna().sum():,} duplicate records"
        )

        print("")

        print(
            "Near-duplicate detection:"
        )

        print(
            "  DEFERRED to candidate-pool stage"
        )

        print("")

        print(
            "Output directory:"
        )

        print(
            f"  {self.output_root}"
        )

        print("")

        print(
            "The raw dataset was NOT modified."
        )

        print(
            "No final 400-image subset was created."
        )

        print(
            "=" * 72
        )


    # =================================================================
    # MAIN PIPELINE
    # =================================================================

    def run_analysis(
        self,
    ) -> None:

        print("")
        print(
            "=" * 72
        )

        print(
            "OASIS DATASET QUALITY AUDIT"
        )

        print(
            "=" * 72
        )

        print(
            f"Dataset:"
            f"\n  {self.data_root}"
        )

        print(
            f"Output:"
            f"\n  {self.output_root}"
        )

        print("")

        # -------------------------------------------------------------
        # STEP 1
        # -------------------------------------------------------------

        self.build_inventory()

        if not self.inventory:

            raise RuntimeError(
                "No images were discovered."
            )

        # -------------------------------------------------------------
        # STEP 2
        # -------------------------------------------------------------

        self.detect_exact_duplicates()

        # -------------------------------------------------------------
        # STEP 3
        # -------------------------------------------------------------

        self.calculate_quality_thresholds()

        # -------------------------------------------------------------
        # STEP 4
        # -------------------------------------------------------------

        self.calculate_quality_scores()

        # -------------------------------------------------------------
        # STEP 5
        # -------------------------------------------------------------

        self.flag_quality_outliers()

        # -------------------------------------------------------------
        # STEP 6
        #
        # DO NOT RUN FULL-DATASET NEAR-DUPLICATE ANALYSIS HERE.
        # -------------------------------------------------------------

        print("")
        print(
            "Skipping full-dataset perceptual "
            "near-duplicate comparison."
        )

        print(
            "Near-duplicate analysis will be performed "
            "later on the candidate pool."
        )

        # -------------------------------------------------------------
        # STEP 7
        # -------------------------------------------------------------

        print("")
        print(
            "Building subject/class/slice summaries..."
        )

        subject_stats = (
            self.build_subject_statistics()
        )

        slice_stats = (
            self.build_slice_statistics()
        )

        class_summary = (
            self.build_class_quality_summary()
        )

        # -------------------------------------------------------------
        # STEP 8
        # -------------------------------------------------------------

        self.save_inventory()

        # -------------------------------------------------------------
        # STEP 9
        # -------------------------------------------------------------

        print(
            "Saving reports..."
        )

        self.save_reports(
            subject_stats,
            slice_stats,
            class_summary,
        )

        # -------------------------------------------------------------
        # STEP 10
        # -------------------------------------------------------------

        try:

            self.create_visual_qa()

        except Exception as exc:

            print(
                ""
            )

            print(
                "WARNING: Visual QA generation "
                "failed, but the dataset audit "
                "reports have already been saved."
            )

            print(
                f"Reason: "
                f"{type(exc).__name__}: {exc}"
            )

        # -------------------------------------------------------------
        # STEP 11
        # -------------------------------------------------------------

        self.print_final_summary()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    analyzer = (
        DatasetQualityAnalyzer()
    )

    analyzer.run_analysis()