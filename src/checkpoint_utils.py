"""
Checkpoint metadata sidecar (CHECKPOINT INTEGRITY task).

Every checkpoint .pt file gets a matching .meta.json recording the
configuration it was trained under. Before reusing an existing checkpoint,
compare its metadata against the CURRENT configuration; only reuse on an
exact match. This turns "checkpoint reload silently gives wrong numbers"
into an explicit, loud failure instead of a silent one.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def hash_split(manifest, split_result: dict, split_name: str) -> str:
    """Deterministic hash of (filepath, class_idx) pairs for one split, in
    the exact order the DataLoader will see them. Two runs with the same
    hash are guaranteed to be evaluating the same files in the same order."""
    path_to_class = dict(zip(manifest["filepath"], manifest["class_idx"]))
    paths = split_result["splits"][split_name]
    payload = "\n".join(f"{p}|{path_to_class[p]}" for p in paths)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_metadata(model_name: str, config, split_result: dict, manifest,
                    best_history: dict, pretrained: bool) -> dict:
    return {
        "class_names": list(config.CLASS_NAMES),
        "image_size": list(config.IMAGE_SIZE),
        "dropout_p": config.DROPOUT_P,
        "model_name": model_name,
        "pretrained": bool(pretrained),
        "split_strategy": split_result.get("strategy"),
        "split_sizes": {k: len(v) for k, v in split_result["splits"].items()},
        "val_split_hash": hash_split(manifest, split_result, "val"),
        "test_split_hash": hash_split(manifest, split_result, "test"),
        "best_val_acc": best_history.get("best_val_acc"),
        "best_val_macro_f1": best_history.get("best_val_macro_f1"),
        "seed": best_history.get("seed"),
    }


def save_metadata(checkpoint_path: Path, metadata: dict) -> Path:
    meta_path = Path(checkpoint_path).with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return meta_path


def is_checkpoint_valid(checkpoint_path: Path, current_metadata: dict) -> tuple[bool, str]:
    """Compare a checkpoint's saved metadata against the current config.
    Returns (ok, reason). ok=False means: do NOT reuse this checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    meta_path = checkpoint_path.with_suffix(".meta.json")

    if not checkpoint_path.exists():
        return False, "checkpoint file does not exist"
    if not meta_path.exists():
        return False, f"no metadata sidecar at {meta_path} (checkpoint predates metadata tracking)"

    saved = json.loads(meta_path.read_text())

    # Keys where an exact match is required for reuse to be safe.
    strict_keys = ["class_names", "image_size", "dropout_p", "model_name",
                   "pretrained", "split_strategy", "val_split_hash", "test_split_hash"]
    for key in strict_keys:
        if saved.get(key) != current_metadata.get(key):
            return False, (f"metadata mismatch on '{key}': "
                            f"checkpoint has {saved.get(key)!r}, current run has {current_metadata.get(key)!r}")

    return True, "metadata matches current configuration"