"""
Locked Alzheimer's Disease Staging Pipeline
============================================

Uses the already-created locked experiment:

    outputs/final_400/
    outputs/splits/split_assignments.csv

IMPORTANT:
- Does NOT regenerate the dataset.
- Does NOT rerun subject splitting.
- Uses the exact locked 400-image experiment.
- Keeps compatibility with the existing dataset.py interface.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src import config
from src import data_acquisition
from src import dataset
from src import augmentation
from src import reporting


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("AD-Staging")


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

OUTPUTS = ROOT / "outputs"

FINAL_400_DIR = OUTPUTS / "final_400"
SPLITS_DIR = OUTPUTS / "splits"

RUN_OUTPUT_DIR = OUTPUTS / "pipeline_runs"
DEMO_OUTPUT_DIR = RUN_OUTPUT_DIR / "demo"
FULL_OUTPUT_DIR = RUN_OUTPUT_DIR / "full"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

CLASS_ORDER = [
    "Non Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented",
]


def ensure_directories() -> None:
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------
# Locked experiment loader
# ---------------------------------------------------------------------

def build_locked_manifest_and_split():
    """
    Load the already-created locked 400-image experiment.

    Does NOT regenerate the dataset or rerun splitting.py.
    Adapts the tuple returned by data_acquisition.load_locked_experiment()
    to the interface expected by dataset.py.
    """

    print_header("STAGE 1 — LOAD LOCKED EXPERIMENT")

    # -------------------------------------------------------------
    # Load authoritative locked experiment
    # -------------------------------------------------------------

    locked = data_acquisition.load_locked_experiment()

    logger.info(
        "Loaded locked experiment from data_acquisition."
    )

    # -------------------------------------------------------------
    # The current data_acquisition implementation returns a tuple.
    # -------------------------------------------------------------

    if not isinstance(locked, tuple):
        raise RuntimeError(
            "Expected load_locked_experiment() to return a tuple, "
            f"but received {type(locked)}"
        )

    logger.info(
        "Locked experiment tuple length: %d",
        len(locked),
    )

    # -------------------------------------------------------------
    # Convert tuple components into something inspectable.
    # -------------------------------------------------------------

    for i, item in enumerate(locked):
        logger.info(
            "Locked tuple[%d]: %s",
            i,
            type(item),
        )

    # -------------------------------------------------------------
    # Find DataFrame components.
    # -------------------------------------------------------------

    dataframes = []

    for i, item in enumerate(locked):

        if isinstance(item, pd.DataFrame):

            logger.info(
                "Found DataFrame at tuple[%d]: shape=%s columns=%s",
                i,
                item.shape,
                list(item.columns),
            )

            dataframes.append(
                (
                    i,
                    item.copy(),
                )
            )

    # -------------------------------------------------------------
    # We expect either:
    #
    #   (manifest, split_result)
    #
    # or:
    #
    #   (train_df, val_df, test_df)
    #
    # or another equivalent tuple.
    # -------------------------------------------------------------

    manifest = None
    split_result = None

    # -------------------------------------------------------------
    # CASE 1:
    # Tuple contains a complete 400-row DataFrame.
    # -------------------------------------------------------------

    complete_candidates = [
        (i, df)
        for i, df in dataframes
        if len(df) == 400
    ]

    if complete_candidates:

        _, manifest = complete_candidates[0]

        logger.info(
            "Using 400-row DataFrame from locked experiment."
        )

    # -------------------------------------------------------------
    # CASE 2:
    # Tuple contains 273 + 44 + 83 DataFrames.
    # -------------------------------------------------------------

    if manifest is None:

        train_candidates = [
            (i, df)
            for i, df in dataframes
            if len(df) == 273
        ]

        val_candidates = [
            (i, df)
            for i, df in dataframes
            if len(df) == 44
        ]

        test_candidates = [
            (i, df)
            for i, df in dataframes
            if len(df) == 83
        ]

        if (
            train_candidates
            and val_candidates
            and test_candidates
        ):

            train_df = train_candidates[0][1].copy()
            val_df = val_candidates[0][1].copy()
            test_df = test_candidates[0][1].copy()

            train_df["split"] = "train"
            val_df["split"] = "val"
            test_df["split"] = "test"

            manifest = pd.concat(
                [
                    train_df,
                    val_df,
                    test_df,
                ],
                ignore_index=True,
            )

            logger.info(
                "Constructed locked manifest from "
                "train/val/test DataFrames."
            )

    # -------------------------------------------------------------
    # If no manifest yet, inspect tuple for dictionaries.
    # -------------------------------------------------------------

    if manifest is None:

        dictionaries = [
            item
            for item in locked
            if isinstance(item, dict)
        ]

        for item in dictionaries:

            # Possible manifest.
            for key in [
                "manifest",
                "data",
                "final_400",
                "final_manifest",
            ]:

                value = item.get(key)

                if isinstance(value, pd.DataFrame):

                    manifest = value.copy()

                    logger.info(
                        "Found manifest under dictionary key '%s'.",
                        key,
                    )

                    break

            if manifest is not None:
                break

    # -------------------------------------------------------------
    # Nothing usable found.
    # -------------------------------------------------------------

    if manifest is None:

        structure = []

        for i, item in enumerate(locked):

            if isinstance(item, pd.DataFrame):

                structure.append(
                    f"tuple[{i}]: DataFrame "
                    f"shape={item.shape}"
                )

            elif isinstance(item, dict):

                structure.append(
                    f"tuple[{i}]: dict "
                    f"keys={list(item.keys())}"
                )

            else:

                structure.append(
                    f"tuple[{i}]: {type(item).__name__}"
                )

        raise RuntimeError(
            "Could not identify the locked manifest.\n\n"
            "Returned tuple structure:\n"
            + "\n".join(structure)
        )

    # -------------------------------------------------------------
    # Normalize column names.
    # -------------------------------------------------------------

    manifest = manifest.copy()

    manifest.columns = [
        str(c).strip()
        for c in manifest.columns
    ]

    # -------------------------------------------------------------
    # Find filepath column.
    # -------------------------------------------------------------

    filepath_col = next(
        (
            c
            for c in [
                "filepath",
                "file_path",
                "path",
                "image_path",
            ]
            if c in manifest.columns
        ),
        None,
    )

    if filepath_col is None:

        raise RuntimeError(
            "Locked manifest has no filepath column.\n"
            f"Columns: {list(manifest.columns)}"
        )

    if filepath_col != "filepath":

        manifest = manifest.rename(
            columns={
                filepath_col: "filepath"
            }
        )

    # -------------------------------------------------------------
    # Find / normalize split information.
    # -------------------------------------------------------------

    if "split" not in manifest.columns:

        # Look for split assignment DataFrame in tuple.
        split_df = None

        for _, df in dataframes:

            if "split" in df.columns:

                split_df = df.copy()
                break

        if split_df is not None:

            split_fp_col = next(
                (
                    c
                    for c in [
                        "filepath",
                        "file_path",
                        "path",
                        "image_path",
                    ]
                    if c in split_df.columns
                ),
                None,
            )

            if split_fp_col is not None:

                if split_fp_col != "filepath":

                    split_df = split_df.rename(
                        columns={
                            split_fp_col: "filepath"
                        }
                    )

                split_lookup = split_df[
                    [
                        "filepath",
                        "split",
                    ]
                ].copy()

                split_lookup["filepath"] = (
                    split_lookup["filepath"]
                    .astype(str)
                )

                manifest["filepath"] = (
                    manifest["filepath"]
                    .astype(str)
                )

                manifest = manifest.merge(
                    split_lookup,
                    on="filepath",
                    how="left",
                    validate="one_to_one",
                )

    # -------------------------------------------------------------
    # If split still doesn't exist, infer it from tuple DataFrames.
    # -------------------------------------------------------------

    if "split" not in manifest.columns:

        train_df = next(
            (
                df
                for _, df in dataframes
                if len(df) == 273
            ),
            None,
        )

        val_df = next(
            (
                df
                for _, df in dataframes
                if len(df) == 44
            ),
            None,
        )

        test_df = next(
            (
                df
                for _, df in dataframes
                if len(df) == 83
            ),
            None,
        )

        if (
            train_df is not None
            and val_df is not None
            and test_df is not None
        ):

            def get_paths(df):

                fp = next(
                    (
                        c
                        for c in [
                            "filepath",
                            "file_path",
                            "path",
                            "image_path",
                        ]
                        if c in df.columns
                    ),
                    None,
                )

                if fp is None:
                    return set()

                return set(
                    df[fp]
                    .astype(str)
                )

            train_paths = get_paths(train_df)
            val_paths = get_paths(val_df)
            test_paths = get_paths(test_df)

            manifest["filepath"] = (
                manifest["filepath"]
                .astype(str)
            )

            manifest["split"] = "unknown"

            manifest.loc[
                manifest["filepath"].isin(train_paths),
                "split",
            ] = "train"

            manifest.loc[
                manifest["filepath"].isin(val_paths),
                "split",
            ] = "val"

            manifest.loc[
                manifest["filepath"].isin(test_paths),
                "split",
            ] = "test"

    # -------------------------------------------------------------
    # Normalize split values.
    # -------------------------------------------------------------

    if "split" not in manifest.columns:

        raise RuntimeError(
            "Could not determine train/val/test assignments "
            "from locked experiment."
        )

    manifest["split"] = (
        manifest["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "training": "train",
                "validation": "val",
                "testing": "test",
            }
        )
    )

    # -------------------------------------------------------------
    # Validate exact locked experiment.
    # -------------------------------------------------------------

    split_counts = (
        manifest["split"]
        .value_counts()
        .to_dict()
    )

    expected = {
        "train": 273,
        "val": 44,
        "test": 83,
    }

    for split_name, expected_count in expected.items():

        actual_count = split_counts.get(
            split_name,
            0,
        )

        if actual_count != expected_count:

            raise RuntimeError(
                f"Locked {split_name} count mismatch: "
                f"expected {expected_count}, "
                f"got {actual_count}"
            )

    if len(manifest) != 400:

        raise RuntimeError(
            f"Expected 400 locked images, "
            f"got {len(manifest)}"
        )

    # -------------------------------------------------------------
    # Validate filepath uniqueness.
    # -------------------------------------------------------------

    manifest["filepath"] = (
        manifest["filepath"]
        .astype(str)
    )

    if manifest["filepath"].duplicated().any():

        raise RuntimeError(
            "Duplicate filepath detected in locked manifest."
        )

    # -------------------------------------------------------------
    # Build interface required by dataset.py.
    # -------------------------------------------------------------

    split_result = {
        "strategy": "locked-subject-level",

        "splits": {
            "train": manifest.loc[
                manifest["split"] == "train",
                "filepath",
            ].tolist(),

            "val": manifest.loc[
                manifest["split"] == "val",
                "filepath",
            ].tolist(),

            "test": manifest.loc[
                manifest["split"] == "test",
                "filepath",
            ].tolist(),
        },
    }

    # -------------------------------------------------------------
    # Final path validation.
    # -------------------------------------------------------------

    all_paths = (
        split_result["splits"]["train"]
        + split_result["splits"]["val"]
        + split_result["splits"]["test"]
    )

    if len(all_paths) != 400:

        raise RuntimeError(
            f"Expected 400 split paths, "
            f"got {len(all_paths)}"
        )

    if len(set(all_paths)) != 400:

        raise RuntimeError(
            "Duplicate paths detected across splits."
        )

    # -------------------------------------------------------------
    # Print summary.
    # -------------------------------------------------------------

    print_locked_summary(
        manifest,
        split_result,
    )

    return manifest, split_result
# ---------------------------------------------------------------------
# Locked experiment summary
# ---------------------------------------------------------------------

def print_locked_summary(
    manifest: pd.DataFrame,
    split_result: dict,
) -> None:

    print_header("LOCKED EXPERIMENT SUMMARY")

    print(f"Total images : {len(manifest)}")

    print()

    print("Split sizes:")

    for split_name in ["train", "val", "test"]:
        df = manifest[
            manifest["split"] == split_name
        ]

        print(
            f"  {split_name:5s}: "
            f"{len(df):3d} images | "
            f"{df['subject_id'].nunique() if 'subject_id' in df.columns else 'N/A'} subjects"
        )

    print()

    print("Class distribution:")

    class_col = next(
        (
            c for c in [
                "label_name",
                "class_name",
                "class",
                "diagnosis",
            ]
            if c in manifest.columns
        ),
        None,
    )

    if class_col is None:
        logger.warning(
            "Could not find class-name column for summary."
        )
        return

    summary = pd.crosstab(
        manifest["split"],
        manifest[class_col],
    )

    # Ensure desired class order when possible.
    ordered_cols = [
        c for c in CLASS_ORDER
        if c in summary.columns
    ]

    remaining_cols = [
        c for c in summary.columns
        if c not in ordered_cols
    ]

    summary = summary[
        ordered_cols + remaining_cols
    ]

    print(summary.to_string())

    print()

    print("Class × split details:")

    for split_name in ["train", "val", "test"]:

        df = manifest[
            manifest["split"] == split_name
        ]

        print(f"\n{split_name.upper()}")

        counts = df[class_col].value_counts()

        for class_name in CLASS_ORDER:
            if class_name in counts:
                print(
                    f"  {class_name:20s}: "
                    f"{counts[class_name]}"
                )


# ---------------------------------------------------------------------
# Loader construction
# ---------------------------------------------------------------------

def build_locked_dataloaders(
    manifest: pd.DataFrame,
    split_result: dict,
):
    """
    Build dataloaders using the locked split.

    This does NOT invoke splitting.py.
    """

    print_header("STAGE 3 — BUILD DATALOADERS")

    loaders = dataset.build_dataloaders(
        manifest=manifest,
        split_result=split_result,
        train_transform=augmentation.get_train_transform(),
        eval_transform=augmentation.get_eval_transform(),
    )

    for split_name in ["train", "val", "test"]:

        loader = loaders[split_name]

        print(
            f"{split_name:5s}: "
            f"{len(loader.dataset):3d} images | "
            f"{len(loader):3d} batches"
        )

    return loaders


# ---------------------------------------------------------------------
# Save JSON helper
# ---------------------------------------------------------------------

def save_json(data, path: Path) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            default=str,
        )


# ---------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------

def run_demo(
    manifest: pd.DataFrame,
    split_result: dict,
    loaders,
) -> None:

    print_header("DEMO MODE")

    print(
        "Running a lightweight DenseNet2D demonstration "
        "on the locked experiment."
    )

    try:
        from src.models.densenet2d import DenseNet2D
    except ImportError:
        from src.models.densenet2d import (
            build_densenet_backbone
        )

    # -------------------------------------------------------------
    # Build model.
    # -------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # -------------------------------------------------------------
    # Prefer compact DenseNet2D.
    # -------------------------------------------------------------

    try:

        model = DenseNet2D(
            in_channels=1,
            num_classes=4,
            dropout_p=0.3,
        )

    except TypeError:

        try:
            model = DenseNet2D(
                num_classes=4,
                dropout_p=0.3,
            )

        except TypeError:

            model = DenseNet2D(
                num_classes=4,
            )

    model = model.to(device)

    # -------------------------------------------------------------
    # One forward pass.
    # -------------------------------------------------------------

    model.eval()

    batch = next(
        iter(loaders["train"])
    )

    # Handle common dataloader formats.
    if isinstance(batch, (tuple, list)):
        images = batch[0]
        labels = batch[1]

    elif isinstance(batch, dict):
        images = batch.get("image")
        labels = batch.get("label")

        if images is None:
            images = batch.get("images")

        if labels is None:
            labels = batch.get("labels")

    else:
        raise RuntimeError(
            f"Unsupported batch type: {type(batch)}"
        )

    images = images.to(device)

    with torch.no_grad():

        logits = model(images)

    predictions = torch.argmax(
        logits,
        dim=1,
    )

    print(
        f"Batch input : {tuple(images.shape)}"
    )

    print(
        f"Logits      : {tuple(logits.shape)}"
    )

    print(
        f"Predictions : {tuple(predictions.shape)}"
    )

    # -------------------------------------------------------------
    # Save demo summary.
    # -------------------------------------------------------------

    summary = {
        "mode": "demo",
        "device": str(device),
        "image_size": list(
            getattr(
                config,
                "IMAGE_SIZE",
                (96, 96),
            )
        ),
        "total_images": len(manifest),

        "train_images": len(
            split_result["splits"]["train"]
        ),

        "val_images": len(
            split_result["splits"]["val"]
        ),

        "test_images": len(
            split_result["splits"]["test"]
        ),

        "test_subjects": manifest[
            manifest["split"] == "test"
        ]["subject_id"].nunique()
        if "subject_id" in manifest.columns
        else None,

        "batch_shape": list(
            images.shape
        ),

        "logits_shape": list(
            logits.shape
        ),

        "num_classes": 4,
    }

    save_json(
        summary,
        DEMO_OUTPUT_DIR / "demo_summary.json",
    )

    print()
    print(
        f"Demo summary saved to:\n"
        f"{DEMO_OUTPUT_DIR / 'demo_summary.json'}"
    )

    print()
    print("DEMO COMPLETED SUCCESSFULLY.")


# ---------------------------------------------------------------------
# Full mode
# ---------------------------------------------------------------------

def run_full(
    manifest: pd.DataFrame,
    split_result: dict,
    loaders,
) -> None:

    print_header("FULL MODE")

    print(
        "Starting comparative evaluation on the "
        "locked 400-image experiment."
    )

    try:
        from src import comparative_evaluation
    except ImportError as exc:
        raise RuntimeError(
            "Could not import comparative_evaluation."
        ) from exc

    # -------------------------------------------------------------
    # Run existing comparative study.
    # -------------------------------------------------------------

    try:

        results = (
            comparative_evaluation
            .run_comparative_study(
                manifest=manifest,
                split_result=split_result,
                loaders=loaders,
            )
        )

    except TypeError:

        # ---------------------------------------------------------
        # Compatibility fallback for older function signatures.
        # ---------------------------------------------------------

        try:

            results = (
                comparative_evaluation
                .run_comparative_study(
                    manifest,
                    split_result,
                    loaders,
                )
            )

        except TypeError as exc:

            raise RuntimeError(
                "The installed comparative_evaluation.py "
                "uses a different run_comparative_study() "
                "signature. Inspect that function before "
                "running full mode."
            ) from exc

    # -------------------------------------------------------------
    # Save returned results when possible.
    # -------------------------------------------------------------

    if results is not None:

        try:

            save_json(
                results,
                FULL_OUTPUT_DIR / "comparative_results.json",
            )

        except Exception as exc:

            logger.warning(
                "Could not serialize comparative results: %s",
                exc,
            )

    print()
    print("FULL COMPARATIVE STUDY COMPLETED.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Locked Alzheimer's Disease Staging Pipeline"
        )
    )

    parser.add_argument(
        "--mode",
        choices=["demo", "full"],
        default="demo",
        help="Pipeline mode.",
    )

    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help=(
            "Accepted for compatibility with the previous "
            "runner. Training modules decide whether a "
            "checkpoint is reused."
        ),
    )

    args = parser.parse_args()

    ensure_directories()

    print_header(
        "ALZHEIMER'S DISEASE STAGING PIPELINE"
    )

    print(
        f"Mode       : {args.mode}"
    )

    print(
        f"Project    : {ROOT}"
    )

    print(
        f"Device     : "
        f"{'CUDA' if torch.cuda.is_available() else 'CPU'}"
    )

    print(
        f"Image size : "
        f"{getattr(config, 'IMAGE_SIZE', 'unknown')}"
    )

    print()

    # -------------------------------------------------------------
    # Configure project.
    # -------------------------------------------------------------

    try:
        config.configure(
            profile=args.mode
        )

    except TypeError:

        # Some config versions use a different signature.
        try:
            config.configure(args.mode)
        except Exception:
            logger.warning(
                "Could not call config.configure() "
                "with profile; continuing with current config."
            )

    # -------------------------------------------------------------
    # LOAD LOCKED DATASET
    # -------------------------------------------------------------

    manifest, split_result = (
        build_locked_manifest_and_split()
    )

    # -------------------------------------------------------------
    # BUILD LOADERS
    # -------------------------------------------------------------

    loaders = build_locked_dataloaders(
        manifest,
        split_result,
    )

    # -------------------------------------------------------------
    # RUN SELECTED MODE
    # -------------------------------------------------------------

    if args.mode == "demo":

        run_demo(
            manifest,
            split_result,
            loaders,
        )

    else:

        run_full(
            manifest,
            split_result,
            loaders,
        )

    print_header("PIPELINE FINISHED")

    print(
        "The locked dataset and locked subject-level split "
        "were not modified."
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()