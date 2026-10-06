"""
Module 3 - Data Augmentation
=============================

Training-only augmentation for the 2D OASIS MRI slices.

Design principles:
    - Augmentation is applied AFTER deterministic preprocessing.
    - Validation/test data are never randomly augmented.
    - Only anatomically conservative transformations are used.
    - Augmentation is NOT cached.

Training augmentations:
    1. Horizontal flip with p=0.5
    2. Small rotation within +/-10 degrees

No:
    - Vertical flips
    - Large rotations
    - Random crops
    - Elastic deformation
    - Color jitter
    - MixUp
    - CutMix
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """
    Convert a standardized grayscale numpy array:

        (H, W) float32

    into a PyTorch tensor:

        (1, H, W) float32
    """
    # Ensure contiguous memory before torch conversion.
    arr = np.ascontiguousarray(arr, dtype=np.float32)

    return torch.from_numpy(arr).unsqueeze(0).float()


def get_train_transform(rotation_degrees: float = 10.0):
    """
    Return the stochastic augmentation pipeline used ONLY for training.

    The input is already:
        - grayscale
        - resized
        - intensity normalized

    Augmentation is deliberately conservative for MRI.
    """
    return transforms.Compose([
        _to_tensor,

        # Left-right symmetry is a reasonable approximation
        # for axial brain slices.
        transforms.RandomHorizontalFlip(p=0.5),

        # Small head-position variation.
        transforms.RandomRotation(
            degrees=rotation_degrees,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0.0,
        ),
    ])


def get_eval_transform():
    """
    Deterministic transform for validation and test.

    No random augmentation is applied.
    """
    return transforms.Compose([
        _to_tensor,
    ])


def validate_transform_shapes(
    sample: np.ndarray,
    train_transform=None,
    eval_transform=None,
):
    """
    Basic sanity check for augmentation outputs.

    Returns True when both training and evaluation transforms
    produce finite [1, H, W] float32 tensors.
    """
    if train_transform is None:
        train_transform = get_train_transform()

    if eval_transform is None:
        eval_transform = get_eval_transform()

    train_out = train_transform(sample)
    eval_out = eval_transform(sample)

    expected_shape = (1, sample.shape[0], sample.shape[1])

    train_ok = (
        isinstance(train_out, torch.Tensor)
        and train_out.shape == expected_shape
        and train_out.dtype == torch.float32
        and torch.isfinite(train_out).all()
    )

    eval_ok = (
        isinstance(eval_out, torch.Tensor)
        and eval_out.shape == expected_shape
        and eval_out.dtype == torch.float32
        and torch.isfinite(eval_out).all()
    )

    return bool(train_ok and eval_ok)


if __name__ == "__main__":
    from . import data_acquisition
    from . import preprocessing

    manifest = data_acquisition.build_manifest()

    sample_path = manifest.iloc[0]["filepath"]

    # Deterministic preprocessing.
    std_arr = preprocessing.preprocess_slice(sample_path)

    train_tf = get_train_transform()
    eval_tf = get_eval_transform()

    out_train_1 = train_tf(std_arr)
    out_train_2 = train_tf(std_arr)
    out_eval_1 = eval_tf(std_arr)
    out_eval_2 = eval_tf(std_arr)

    print(f"Sample: {sample_path}")
    print(f"Input shape: {std_arr.shape}")
    print(f"Input dtype: {std_arr.dtype}")
    print(f"Input mean: {std_arr.mean():.6f}")
    print(f"Input std: {std_arr.std():.6f}")

    print()
    print(f"Train output shape: {tuple(out_train_1.shape)}")
    print(f"Train output dtype: {out_train_1.dtype}")
    print(f"Train finite: {torch.isfinite(out_train_1).all().item()}")

    print()
    print(f"Eval output shape: {tuple(out_eval_1.shape)}")
    print(f"Eval output dtype: {out_eval_1.dtype}")
    print(f"Eval finite: {torch.isfinite(out_eval_1).all().item()}")

    # Eval must be deterministic.
    eval_identical = torch.equal(out_eval_1, out_eval_2)

    print()
    print(f"Eval deterministic: {eval_identical}")

    # Train transforms are stochastic in principle.
    train_different = not torch.equal(out_train_1, out_train_2)

    print(f"Train augmentation varied: {train_different}")

    print()
    print(
        f"Transform sanity check: "
        f"{validate_transform_shapes(std_arr)}"
    )