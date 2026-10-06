"""
Central configuration for the Uncertainty-Aware sMRI AD Staging pipeline.

ENGINEERING PASS (demo/perf fixes):
  - Added DEMO vs FULL execution profiles (configure()). DEMO is the
    default and is meant to run in a reasonable time on a normal laptop;
    FULL restores the original 3-backbone x 3-uncertainty-method study.
  - Added hardware detection (CUDA/MPS/CPU) and centralized DataLoader
    settings (previously hardcoded num_workers=0 everywhere).
  - Added preprocessing-cache settings (see preprocessing.get_cached_slice).
  - Added a single CLASS_BALANCE_STRATEGY switch. The original code used
    BOTH a WeightedRandomSampler AND class-weighted CrossEntropyLoss at
    the same time, which double-compensates for imbalance. Only one is
    applied now; default is loss-weighting (cheaper: no sampler overhead,
    still touches every sample as intended for a WeightedRandomSampler
    but without the extra bookkeeping/replacement sampling cost).
  - Added AdamW weight decay and early-stopping patience for train.py.

IMPORTANT ordering note: several modules (preprocessing.py, vit2d.py,
train.py, dataset.py) resolve IMAGE_SIZE/EPOCHS/etc. *lazily inside the
function body* rather than as literal default-argument values, so calling
configure() at any point before those functions are actually invoked is
safe -- you do not need to call configure() before importing the rest of
`src`. run_pipeline.py still calls it as early as possible for clarity.
"""
from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "oasis_subset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MANIFEST_CSV = OUTPUT_ROOT / "manifests" / "manifest.csv"
FINAL_400_CSV = OUTPUT_ROOT / "final_400" / "final_400.csv"
SPLIT_ASSIGNMENTS_CSV = OUTPUT_ROOT / "splits" / "split_assignments.csv"
SPLITS_JSON = OUTPUT_ROOT / "manifests" / "splits.json"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
METRICS_DIR = OUTPUT_ROOT / "metrics"
PLOTS_DIR = OUTPUT_ROOT / "plots"
CACHE_DIR = OUTPUT_ROOT / "cache"

# ---------------------------------------------------------------------------
# Class labels for the 4-class Alzheimer's staging problem:
# Non Demented, Very Mild Demented, Mild Demented, Moderate Demented.
# ---------------------------------------------------------------------------
RAW_FOLDER_TO_CLASS = {
    "non demented": "Non Demented",
    "very mild demented": "Very Mild Demented",
    "Mild dementia": "Mild Demented",
    "Moderate dementia": "Moderate Demented",
}
CLASS_NAMES = ["Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Hardware detection (Section 17)
# ---------------------------------------------------------------------------
def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _detect_device()
CPU_THREADS = multiprocessing.cpu_count()
# Keep worker counts modest -- the requirement is "sensible", not "maximal".
# In-memory preprocessing cache means later epochs are cheap regardless of
# worker count, so we don't need many workers to hide I/O latency.
if DEVICE.type == "cpu":
    NUM_WORKERS = min(2, max(0, CPU_THREADS - 1))
else:
    NUM_WORKERS = min(4, max(0, CPU_THREADS - 1))
PIN_MEMORY = DEVICE.type == "cuda"
PERSISTENT_WORKERS = NUM_WORKERS > 0
USE_AMP = DEVICE.type == "cuda"  # mixed precision only where it's safe (CUDA)

# ---------------------------------------------------------------------------
# Preprocessing cache (Section 1)
# ---------------------------------------------------------------------------
USE_PREPROCESS_CACHE = os.environ.get("USE_PREPROCESS_CACHE", "True") == "True"
# If the whole preprocessed dataset is estimated to fit under this budget,
# cache in a plain in-process dict; otherwise spill to an on-disk .npy
# cache under CACHE_DIR keyed by (filepath, image_size).
CACHE_IN_MEMORY_MAX_MB = int(os.environ.get("CACHE_IN_MEMORY_MAX_MB", "2048"))

# ---------------------------------------------------------------------------
# Class balancing (Section 4) -- pick ONE, not both.
# ---------------------------------------------------------------------------
# "loss"    -> inverse-frequency weighted CrossEntropyLoss (cheap: computed
#              once from the manifest, no sampler bookkeeping).
# "sampler" -> WeightedRandomSampler only, unweighted loss.
CLASS_BALANCE_STRATEGY = os.environ.get("CLASS_BALANCE_STRATEGY", "loss")

# ---------------------------------------------------------------------------
# Training (Section 3)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4          # AdamW; was plain Adam with no decay
DROPOUT_P = 0.35             # shared by classifier head and MC-Dropout
EARLY_STOP_PATIENCE = 6      # stop if val macro-F1 hasn't improved in N epochs
LR_SCHEDULER = os.environ.get("LR_SCHEDULER", "plateau")  # "plateau" | "cosine"
CHECKPOINT_METRIC = "macro_f1"  # was accuracy-only; see train.py

# ---------------------------------------------------------------------------
# Vision Transformer specific (FULL mode only)
# ---------------------------------------------------------------------------
VIT_PATCH_SIZE = 8
VIT_EMBED_DIM = 32
VIT_DEPTH = 3
VIT_NUM_HEADS = 4
VIT_MLP_DIM = 64

# ---------------------------------------------------------------------------
# Uncertainty estimation
# ---------------------------------------------------------------------------
MC_DROPOUT_SAMPLES = 15      # T in Algorithm 6 (was 30; see diagnosis)
ENSEMBLE_SIZE = 3            # M in Algorithm 8 (FULL mode default)

# ---------------------------------------------------------------------------
# Transfer learning (Section 5)
# ---------------------------------------------------------------------------
USE_PRETRAINED_DEMO_BACKBONE = os.environ.get("USE_PRETRAINED_DEMO_BACKBONE", "True") == "True"
FREEZE_BACKBONE_UNTIL = "denseblock4"  # unfreeze only the last dense block + head initially

# ---------------------------------------------------------------------------
# Execution profiles: DEMO (default) vs FULL (Sections 0/14)
# ---------------------------------------------------------------------------
MODE = "demo"
IMAGE_SIZE = (96, 96)
EPOCHS = 25
MAX_ATTEMPTS = 2

_PROFILES = {
    "demo": dict(
        IMAGE_SIZE=(96, 96),
        EPOCHS=25,
        MAX_ATTEMPTS=2,
        MC_DROPOUT_SAMPLES=12,
        ENSEMBLE_SIZE=2,
    ),
    "full": dict(
        IMAGE_SIZE=(64, 64),
        EPOCHS=30,
        MAX_ATTEMPTS=3,
        MC_DROPOUT_SAMPLES=30,
        ENSEMBLE_SIZE=3,
    ),
}


def configure(mode: str = "demo") -> dict:
    """Apply a named profile (demo/full) by mutating this module's globals.

    Safe to call at any point: every function elsewhere in the codebase
    that depends on these values reads config.<NAME> lazily inside its own
    body rather than baking it into a default-argument at import time.
    """
    global MODE
    if mode not in _PROFILES:
        raise ValueError(f"Unknown mode '{mode}'. Choose one of {list(_PROFILES)}.")
    MODE = mode
    for key, value in _PROFILES[mode].items():
        globals()[key] = value
    return dict(_PROFILES[mode])


def print_hardware_banner() -> None:
    gpu_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "n/a"
    print("-" * 70)
    print(f"Device:            {DEVICE}")
    print(f"GPU available:     {DEVICE.type == 'cuda'}")
    print(f"GPU name:          {gpu_name}")
    print(f"CPU threads:       {CPU_THREADS}")
    print(f"Image size:        {IMAGE_SIZE}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Number of workers: {NUM_WORKERS}")
    print(f"Demo/full mode:    {MODE}")
    print("-" * 70)


# Apply the demo profile immediately so importing config alone (e.g. from a
# unit test or a REPL) already reflects the demo defaults described above.
configure(MODE)