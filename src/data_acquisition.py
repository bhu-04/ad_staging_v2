"""
Dataset acquisition and locked-experiment manifest handling.

The raw OASIS dataset is kept untouched.

For the actual experiment, the pipeline MUST use:
    outputs/final_400/final_400.csv
    outputs/splits/split_assignments.csv

These files define the already-selected 400-image dataset and the
already-validated subject-level train/validation/test split.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FILENAME_RE = re.compile(
    r"(?P<subject_id>OAS\d+_\d+)_(?P<session>MR\d+)_(?P<scan>mpr-\d+)_(?P<slice_idx>\d+)"
)


# ---------------------------------------------------------------------------
# Raw dataset manifest
# ---------------------------------------------------------------------------

def _parse_filename(filename: str) -> dict:
    """Extract OASIS subject/session/scan/slice information."""
    match = FILENAME_RE.search(filename)

    if match:
        d = match.groupdict()
        d["slice_idx"] = int(d["slice_idx"])
        return d

    logger.warning(
        "Filename '%s' did not match expected OASIS pattern.",
        filename,
    )

    stem = Path(filename).stem

    return {
        "subject_id": stem,
        "session": "unknown",
        "scan": "unknown",
        "slice_idx": -1,
    }


def build_manifest(
    data_root: Path = config.DATA_ROOT,
    folder_to_class: dict = config.RAW_FOLDER_TO_CLASS,
) -> pd.DataFrame:
    """
    Walk the raw dataset and build a complete manifest.

    This function is retained for dataset auditing/reproducibility.

    IMPORTANT:
        Training does NOT call this function.
        Training uses load_locked_experiment() below.
    """

    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset root '{data_root}' does not exist."
        )

    rows = []

    for raw_folder, class_label in folder_to_class.items():

        folder_path = data_root / raw_folder

        if not folder_path.is_dir():
            logger.warning(
                "Expected raw folder not found: %s",
                folder_path,
            )
            continue

        image_files = sorted(
            p
            for p in folder_path.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )

        for img_path in image_files:

            ident = _parse_filename(img_path.name)

            rows.append(
                {
                    "filepath": str(img_path),
                    "raw_folder": raw_folder,
                    "class_label": class_label,
                    "class_idx": config.CLASS_TO_IDX[class_label],
                    "subject_id": ident["subject_id"],
                    "session": ident["session"],
                    "scan": ident["scan"],
                    "slice_idx": ident["slice_idx"],
                }
            )

    if not rows:
        raise RuntimeError(
            f"No images found under {data_root}."
        )

    manifest = pd.DataFrame(rows)

    manifest = manifest.sort_values(
        [
            "class_label",
            "subject_id",
            "session",
            "scan",
            "slice_idx",
        ]
    ).reset_index(drop=True)

    return manifest


def summarize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:

    return (
        manifest.groupby("class_label")
        .agg(
            n_files=("filepath", "count"),
            n_subjects=("subject_id", "nunique"),
        )
        .reindex(config.CLASS_NAMES)
    )


def save_manifest(
    manifest: pd.DataFrame,
    path: Path = config.MANIFEST_CSV,
) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(path, index=False)

    logger.info(
        "Manifest saved to %s (%d rows)",
        path,
        len(manifest),
    )


# ---------------------------------------------------------------------------
# Locked 400-image experiment
# ---------------------------------------------------------------------------

def _resolve_filepath(filepath: str) -> Path:
    """
    Resolve a filepath from final_400.csv.

    The locked CSV stores paths relative to data/oasis_subset.
    Absolute paths are also accepted for robustness.
    """

    path = Path(str(filepath))

    if path.is_absolute():
        return path

    # CSV was generated on Windows and therefore uses backslashes.
    # Path() handles them correctly on Windows.
    return config.DATA_ROOT / path.relative_to(
        Path("oasis_subset")
    )


def _validate_locked_files(
    final_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> None:
    """Perform integrity checks before training."""

    required_final = {
        "filepath",
        "class_name",
        "class_id",
        "subject_id",
        "session",
        "scan",
        "slice_idx",
    }

    required_split = {
        "filepath",
        "class_name",
        "class_id",
        "subject_id",
        "session",
        "scan",
        "slice_idx",
        "split",
    }

    missing_final = required_final - set(final_df.columns)
    missing_split = required_split - set(split_df.columns)

    if missing_final:
        raise ValueError(
            f"final_400.csv is missing columns: {sorted(missing_final)}"
        )

    if missing_split:
        raise ValueError(
            "split_assignments.csv is missing columns: "
            f"{sorted(missing_split)}"
        )

    if len(final_df) != 400:
        raise ValueError(
            f"Expected exactly 400 final images, got {len(final_df)}."
        )

    if len(split_df) != 400:
        raise ValueError(
            f"Expected exactly 400 split assignments, got {len(split_df)}."
        )

    final_paths = set(final_df["filepath"].astype(str))
    split_paths = set(split_df["filepath"].astype(str))

    if final_paths != split_paths:
        only_final = final_paths - split_paths
        only_split = split_paths - final_paths

        raise ValueError(
            "final_400.csv and split_assignments.csv do not contain "
            "the same 400 filepaths.\n"
            f"Only in final_400: {len(only_final)}\n"
            f"Only in splits: {len(only_split)}"
        )

    if final_df["filepath"].duplicated().any():
        raise ValueError("Duplicate filepath found in final_400.csv.")

    if split_df["filepath"].duplicated().any():
        raise ValueError(
            "Duplicate filepath found in split_assignments.csv."
        )

    valid_splits = {"train", "val", "test"}

    actual_splits = set(split_df["split"].astype(str))

    if actual_splits != valid_splits:
        raise ValueError(
            f"Expected splits {valid_splits}, got {actual_splits}."
        )

    # Verify metadata consistency between the two locked files.
    metadata_cols = [
        "class_name",
        "class_id",
        "subject_id",
        "session",
        "scan",
        "slice_idx",
    ]

    merged = final_df[
        ["filepath"] + metadata_cols
    ].merge(
        split_df[
            ["filepath"] + metadata_cols + ["split"]
        ],
        on="filepath",
        suffixes=("_final", "_split"),
        how="inner",
        validate="one_to_one",
    )

    for col in metadata_cols:

        mismatch = (
            merged[f"{col}_final"].astype(str)
            != merged[f"{col}_split"].astype(str)
        )

        if mismatch.any():
            raise ValueError(
                f"Metadata mismatch between final_400.csv and "
                f"split_assignments.csv for column '{col}'."
            )

    # Check that every physical image exists.
    missing_files = []

    for filepath in final_df["filepath"]:
        path = _resolve_filepath(filepath)

        if not path.is_file():
            missing_files.append(str(path))

    if missing_files:

        preview = "\n".join(missing_files[:10])

        raise FileNotFoundError(
            f"{len(missing_files)} locked images do not exist.\n"
            f"First missing files:\n{preview}"
        )

    # Subject leakage check.
    subject_splits = (
        split_df.groupby("subject_id")["split"]
        .nunique()
    )

    leaking_subjects = subject_splits[
        subject_splits > 1
    ]

    if not leaking_subjects.empty:
        raise ValueError(
            "Subject leakage detected. Subjects appear in multiple "
            f"splits: {list(leaking_subjects.index)}"
        )


def load_locked_experiment():
    """
    Load the immutable 400-image experiment and its locked split.

    Returns
    -------
    train_manifest
    val_manifest
    test_manifest
    """

    final_csv = (
        config.OUTPUT_ROOT
        / "final_400"
        / "final_400.csv"
    )

    split_csv = (
        config.OUTPUT_ROOT
        / "splits"
        / "split_assignments.csv"
    )

    if not final_csv.exists():
        raise FileNotFoundError(
            f"Locked dataset manifest not found:\n{final_csv}"
        )

    if not split_csv.exists():
        raise FileNotFoundError(
            f"Locked split manifest not found:\n{split_csv}"
        )

    final_df = pd.read_csv(final_csv)
    split_df = pd.read_csv(split_csv)

    _validate_locked_files(final_df, split_df)

    # Use the split file as the authoritative partition assignment,
    # while retaining all metadata from final_400.csv.
    metadata_cols = [
        "filepath",
        "class_name",
        "class_id",
        "subject_id",
        "session",
        "scan",
        "slice_idx",
    ]

    manifest = (
        final_df[metadata_cols]
        .merge(
            split_df[["filepath", "split"]],
            on="filepath",
            how="inner",
            validate="one_to_one",
        )
    )

    # Convert the relative filepath to the actual filesystem path.
    manifest["filepath"] = manifest["filepath"].map(
        lambda x: str(_resolve_filepath(x))
    )

    manifest["class_label"] = manifest["class_name"]
    manifest["class_idx"] = manifest["class_id"].astype(int)

    train_manifest = (
        manifest[manifest["split"] == "train"]
        .drop(columns=["split"])
        .reset_index(drop=True)
    )

    val_manifest = (
        manifest[manifest["split"] == "val"]
        .drop(columns=["split"])
        .reset_index(drop=True)
    )

    test_manifest = (
        manifest[manifest["split"] == "test"]
        .drop(columns=["split"])
        .reset_index(drop=True)
    )

    logger.info(
        "LOCKED EXPERIMENT: %d train / %d val / %d test",
        len(train_manifest),
        len(val_manifest),
        len(test_manifest),
    )

    logger.info(
        "Classes: %s",
        ", ".join(config.CLASS_NAMES),
    )

    return train_manifest, val_manifest, test_manifest


if __name__ == "__main__":

    train, val, test = load_locked_experiment()

    print("\nLOCKED EXPERIMENT")
    print("=" * 60)

    for name, df in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:

        print(
            f"{name:>5}: "
            f"{len(df):3d} images | "
            f"{df.subject_id.nunique():3d} subjects"
        )

        print(
            df["class_label"]
            .value_counts()
            .reindex(config.CLASS_NAMES, fill_value=0)
            .to_dict()
        )