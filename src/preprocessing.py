"""
Module 2 - MRI Preprocessing
==============================
Implements Algorithm 1 ("MRI Preprocessing") from the project document,
adapted to the actual input format.

Algorithm 1 as written assumes a raw NIfTI volume and performs
skull-stripping -> intensity normalization -> registration -> resampling.
The Kaggle release used here ships already skull-stripped, already
registered 2D axial slices (this is standard for OASIS-derived slice
datasets), so steps 1 and 3 of Algorithm 1 are pre-applied upstream by the
dataset publisher. What remains -- and what this module actually
implements -- is:

    1. Grayscale conversion (source JPEGs are RGB with the value
       replicated across channels; single-channel is the correct
       representation for MRI intensity).
    2. Intensity normalization to zero-mean / unit-variance
       (Algorithm 1, step 2), matching the doc's stated normalization
       target exactly.
    3. Resampling to a fixed target grid, i.e. resizing to a fixed
       (H, W) (Algorithm 1, step 4).

PERF NOTE: target_size defaults are resolved *lazily* from config.IMAGE_SIZE
inside each function body (not as a literal default-argument value), so
that config.configure("demo"/"full") -- called any time before these
functions actually run -- is always respected, regardless of import order.

CACHE NOTE (Section 1 of the optimization pass): the original code re-ran
this full pipeline (JPEG decode -> resize -> normalize) from scratch inside
every __getitem__ call, i.e. once per image per epoch per model trained.
`get_cached_slice()` below preprocesses each file at most once per process
and reuses the standardized array afterwards. Augmentation is NEVER cached
-- callers must keep applying augmentation.get_train_transform() on top of
the cached array for the train split, dynamically, per __getitem__ call.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

logger = logging.getLogger(__name__)


def load_slice_as_array(filepath: str) -> np.ndarray:
    """Load a raw slice image and convert to single-channel float32 array."""
    img = Image.open(filepath).convert("L")  # grayscale
    return np.asarray(img, dtype=np.float32)


def resize_array(arr: np.ndarray, target_size=None) -> np.ndarray:
    """Resample to the fixed target grid (Algorithm 1, step 4)."""
    target_size = target_size if target_size is not None else config.IMAGE_SIZE
    img = Image.fromarray(arr)
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)  # PIL wants (W, H)
    return np.asarray(img, dtype=np.float32)


def normalize_intensity(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Zero-mean / unit-variance normalization (Algorithm 1, step 2)."""
    mean, std = arr.mean(), arr.std()
    return (arr - mean) / (std + eps)


def preprocess_slice(filepath: str, target_size=None) -> np.ndarray:
    """Full Algorithm-1-equivalent pipeline for one slice.

    Returns a standardized (H, W) float32 array, zero-mean/unit-variance,
    at the fixed target shape -- the V_std referenced in Algorithm 1.
    """
    target_size = target_size if target_size is not None else config.IMAGE_SIZE
    arr = load_slice_as_array(filepath)
    arr = resize_array(arr, target_size)
    arr = normalize_intensity(arr)
    return arr


def quality_check(std_arr: np.ndarray, target_size=None) -> bool:
    """Algorithm 1's 'Evaluation' step: verify output shape and that
    normalization actually produced a near-zero mean / near-unit variance
    array (rather than silently passing through a corrupt/blank image)."""
    target_size = target_size if target_size is not None else config.IMAGE_SIZE
    shape_ok = std_arr.shape == tuple(target_size)
    stats_ok = abs(std_arr.mean()) < 1e-2 and 0.9 < std_arr.std() < 1.1
    return bool(shape_ok and stats_ok)


# ---------------------------------------------------------------------------
# Deterministic preprocessing cache
# ---------------------------------------------------------------------------
# Per-process in-memory cache, keyed by (filepath, image_size), so a cache
# built under one IMAGE_SIZE never leaks stale results if the mode changes
# mid-process. No label information is stored here -- this is pure input
# preprocessing, so sharing it across train/val/test creates no leakage:
# the same deterministic function of the same file is computed once
# instead of N times, independent of which split the file belongs to.
_MEM_CACHE: dict = {}


def _cache_key(filepath: str, target_size) -> tuple:
    return (filepath, tuple(target_size))


def _disk_cache_path(filepath: str, target_size) -> Path:
    import hashlib
    h = hashlib.sha1(f"{filepath}|{target_size}".encode()).hexdigest()
    return config.CACHE_DIR / f"{h}.npy"


def _estimated_mb(n_files: int, target_size) -> float:
    bytes_per_image = target_size[0] * target_size[1] * 4  # float32
    return (n_files * bytes_per_image) / (1024 ** 2)


def get_cached_slice(filepath: str, target_size=None) -> np.ndarray:
    """Preprocess `filepath` once and reuse the result for every later
    call in this process (or this DataLoader worker). Falls back to always
    recomputing when config.USE_PREPROCESS_CACHE is False."""
    target_size = target_size if target_size is not None else config.IMAGE_SIZE
    if not config.USE_PREPROCESS_CACHE:
        return preprocess_slice(filepath, target_size)

    key = _cache_key(filepath, target_size)
    cached = _MEM_CACHE.get(key)
    if cached is not None:
        return cached

    disk_path = _disk_cache_path(filepath, target_size)
    if disk_path.exists():
        arr = np.load(disk_path)
        _MEM_CACHE[key] = arr
        return arr

    arr = preprocess_slice(filepath, target_size)

    # Stay in memory while the running total is comfortably under the
    # configured RAM budget; otherwise spill to disk so RAM stays bounded
    # (this branch matters for the full OASIS release, not this ~480-image
    # demo subset, which fits in a few tens of MB).
    approx_mb = (len(_MEM_CACHE) + 1) * _estimated_mb(1, target_size)
    if approx_mb > config.CACHE_IN_MEMORY_MAX_MB:
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(disk_path, arr)
    _MEM_CACHE[key] = arr
    return arr


def clear_cache() -> None:
    """Explicitly drop the in-memory cache (does not touch the on-disk
    cache under config.CACHE_DIR)."""
    _MEM_CACHE.clear()


if __name__ == "__main__":
    from . import data_acquisition

    manifest = data_acquisition.build_manifest()
    sample_path = manifest.iloc[0]["filepath"]
    std = preprocess_slice(sample_path)
    print(f"Sample: {sample_path}")
    print(f"Output shape: {std.shape}, mean: {std.mean():.4f}, std: {std.std():.4f}")
    print(f"Quality check passed: {quality_check(std)}")
