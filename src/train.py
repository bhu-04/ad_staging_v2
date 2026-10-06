"""
Training Procedure (Algorithm 3, "Training Procedure / Optimization")
=======================================================================
Standard supervised training loop: shuffle -> forward -> loss -> backprop
-> optimizer step -> validate -> checkpoint the best model -> retrain from
a new seed if the result is degenerate.

OPTIMIZATION-PASS CHANGES vs. the original module:
  1. AdamW (decoupled weight decay, config.WEIGHT_DECAY) instead of plain
     Adam -- helps regularize a small model on a small dataset.
  2. Checkpointing now uses validation MACRO-F1 first (tie-broken by
     accuracy, then by lower loss), not accuracy alone. This is a 3-class
     medical staging problem with class imbalance (see config docs), and
     accuracy alone can look good while missing the minority classes
     entirely -- macro-F1 penalizes that.
  3. Per-class precision/recall/F1 is computed every epoch and the full
     history (train/val loss & acc, val macro-F1, per-class P/R/F1) is
     kept, not just loss/acc.
  4. Early stopping: training now stops once val macro-F1 hasn't improved
     for `config.EARLY_STOP_PATIENCE` epochs, instead of always burning
     the full epoch budget. This is the single biggest per-run time
     saver alongside the preprocessing cache.
  5. Class weights are no longer computed by iterating the DataLoader
     (which forces a full pass over every image before training even
     starts, once per call). `class_weights_from_manifest()` computes
     them directly from the manifest/split via pandas/numpy instead;
     `train_model` accepts a precomputed `class_weights` argument so the
     caller only computes this once and passes it to every training call
     that needs it (backbone, ensemble members, BEDL).
  6. config-derived defaults (epochs, lr, checkpoint path, seed, attempts)
     are resolved lazily inside the function body instead of as literal
     default-argument values, so config.configure("demo"/"full") is
     always respected regardless of import order.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support

from . import config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class weights -- computed once from labels, not by iterating a DataLoader
# ---------------------------------------------------------------------------
def class_weights_from_labels(labels, num_classes: int, device=None) -> torch.Tensor:
    """Inverse-frequency class weights (normalized), from a plain array of
    integer labels. O(n) over labels only -- no image loading."""
    device = device if device is not None else config.DEVICE
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (num_classes * counts)  # inverse frequency, normalized
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def class_weights_from_manifest(manifest: pd.DataFrame, split_result: dict,
                                 num_classes: int, split: str = "train",
                                 device=None) -> torch.Tensor:
    """Same result as the original compute_class_weights(loader, ...), but
    computed directly from the manifest's class_idx column for the given
    split -- no DataLoader iteration, no image decoding."""
    filepaths = set(split_result["splits"][split])
    labels = manifest.loc[manifest["filepath"].isin(filepaths), "class_idx"].to_numpy()
    return class_weights_from_labels(labels, num_classes, device=device)


def compute_class_weights(loader, num_classes: int, device=None) -> torch.Tensor:
    """Kept for backwards compatibility with callers that only have a
    DataLoader handy (e.g. ad-hoc scripts). Prefer class_weights_from_manifest
    in any hot path -- this iterates every batch's labels."""
    device = device if device is not None else config.DEVICE
    counts = torch.zeros(num_classes)
    for _, y in loader:
        counts += torch.bincount(y, minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


# ---------------------------------------------------------------------------
# Epoch loop
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            n += x.size(0)
            all_preds.append(preds.detach().cpu())
            all_labels.append(y.detach().cpu())
    all_preds = torch.cat(all_preds).numpy() if all_preds else np.array([])
    all_labels = torch.cat(all_labels).numpy() if all_labels else np.array([])
    return total_loss / max(n, 1), correct / max(n, 1), all_preds, all_labels


def _is_degenerate(val_preds: np.ndarray, val_labels: np.ndarray, num_classes: int) -> bool:
    """A checkpoint is degenerate if it collapses to predicting a single
    class for the entire validation set, or scores at/below the naive
    majority-class baseline it could reach without looking at the image."""
    n_unique_preds = len(np.unique(val_preds))
    if n_unique_preds <= 1:
        return True
    majority_baseline = np.bincount(val_labels, minlength=num_classes).max() / max(len(val_labels), 1)
    val_acc = (val_preds == val_labels).mean()
    return val_acc <= majority_baseline + 1e-9


def _per_class_metrics(y_true, y_pred, num_classes: int) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    return {
        config.CLASS_NAMES[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(num_classes)
    }


def _train_once(model, train_loader, val_loader, epochs: int, lr: float, weight_decay: float,
                 checkpoint_path: Path, device, seed: int,
                 class_weights: torch.Tensor = None, patience: int = None,
                 scheduler_kind: str = None) -> dict:
    patience = patience if patience is not None else config.EARLY_STOP_PATIENCE
    scheduler_kind = scheduler_kind if scheduler_kind is not None else config.LR_SCHEDULER

    torch.manual_seed(seed)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_val_macro_f1, best_val_acc, best_val_loss = -1.0, -1.0, float("inf")
    best_val_preds, best_val_labels = None, None
    epochs_since_improve = 0

    num_classes = len(config.CLASS_NAMES)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0) if len(val_labels) else 0.0

        if scheduler_kind == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        # Checkpoint by validation MACRO-F1 first (tie-broken by accuracy,
        # then by lower loss) -- see module docstring.
        improved = (
            val_macro_f1 > best_val_macro_f1
            or (val_macro_f1 == best_val_macro_f1 and val_acc > best_val_acc)
            or (val_macro_f1 == best_val_macro_f1 and val_acc == best_val_acc and val_loss < best_val_loss)
        )
        if improved:
            best_val_macro_f1, best_val_acc, best_val_loss = val_macro_f1, val_acc, val_loss
            best_val_preds, best_val_labels = val_preds, val_labels
            torch.save(model.state_dict(), checkpoint_path)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        logger.info(
            "Epoch %2d/%2d | train_loss=%.4f train_acc=%.3f | val_loss=%.4f val_acc=%.3f val_macroF1=%.3f%s",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc, val_macro_f1,
            "  <- checkpoint saved" if improved else "",
        )

        if epochs_since_improve >= patience:
            logger.info("Early stopping at epoch %d (no val macro-F1 improvement in %d epochs).",
                        epoch, patience)
            break

    degenerate = _is_degenerate(best_val_preds, best_val_labels, num_classes=num_classes)
    history["best_val_acc"] = best_val_acc
    history["best_val_loss"] = best_val_loss
    history["best_val_macro_f1"] = best_val_macro_f1
    history["best_val_per_class"] = _per_class_metrics(best_val_labels, best_val_preds, num_classes)
    history["degenerate"] = bool(degenerate)
    history["seed"] = seed
    history["epochs_run"] = epoch
    return history


def train_model(model, train_loader, val_loader,
                 epochs: int = None,
                 lr: float = None,
                 weight_decay: float = None,
                 checkpoint_path: Path = None,
                 device=None,
                 max_attempts: int = None,
                 base_seed: int = None,
                 class_weights: torch.Tensor = None,
                 patience: int = None,
                 scheduler_kind: str = None) -> dict:
    """Train `model`, retrying from a fresh random seed (re-initializing
    weights) up to `max_attempts` times if the resulting checkpoint is
    degenerate. Always leaves the best attempt's weights at
    `checkpoint_path` when done.

    `class_weights`, if not supplied, is computed via `compute_class_weights`
    (DataLoader iteration) for backwards compatibility -- callers on a hot
    path should pass a precomputed tensor from
    `class_weights_from_manifest()` instead so the train split isn't
    iterated an extra time just to get label counts.
    """
    epochs = epochs if epochs is not None else config.EPOCHS
    lr = lr if lr is not None else config.LEARNING_RATE
    weight_decay = weight_decay if weight_decay is not None else config.WEIGHT_DECAY
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else config.CHECKPOINT_DIR / "cnn2d_best.pt"
    device = device if device is not None else config.DEVICE
    max_attempts = max_attempts if max_attempts is not None else config.MAX_ATTEMPTS
    base_seed = base_seed if base_seed is not None else config.RANDOM_SEED

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model_cls = type(model)
    best_history = None
    best_state_dict = None

    if class_weights is None and config.CLASS_BALANCE_STRATEGY == "loss":
        class_weights = compute_class_weights(train_loader, num_classes=len(config.CLASS_NAMES), device=device)
    elif config.CLASS_BALANCE_STRATEGY != "loss":
        class_weights = None  # sampler is already handling balance; don't double up

    if class_weights is not None:
        logger.info("Class weights (inverse frequency, order=%s): %s",
                    config.CLASS_NAMES, [round(w, 3) for w in class_weights.tolist()])

    for attempt in range(1, max_attempts + 1):
        seed = base_seed + attempt - 1
        attempt_epochs = epochs
        if attempt > 1:
            attempt_epochs = int(epochs * (1 + 0.5 * (attempt - 1)))
            retry_dropout = max(0.1, config.DROPOUT_P - 0.1 * (attempt - 1))
            try:
                model = model_cls(num_classes=len(config.CLASS_NAMES), dropout_p=retry_dropout)
            except TypeError:
                # Some model factories (e.g. pretrained backbones) don't take
                # dropout_p directly -- fall back to the model's own default.
                model = model_cls(num_classes=len(config.CLASS_NAMES))
            logger.warning(
                "Attempt %d/%d: previous attempt's checkpoint was degenerate. "
                "Re-initializing with seed=%d, epochs=%d (was %d).",
                attempt, max_attempts, seed, attempt_epochs, epochs,
            )

        history = _train_once(model, train_loader, val_loader, attempt_epochs, lr, weight_decay,
                               checkpoint_path, device, seed,
                               class_weights=class_weights, patience=patience,
                               scheduler_kind=scheduler_kind)

        # BUG FIX: _train_once() already wrote the BEST epoch's weights to
        # checkpoint_path on disk every time val macro-F1 improved. But the
        # in-memory `model` object is left at whatever epoch the loop last
        # ran (early stopping means this is *after* the best epoch, not at
        # it) -- e.g. best epoch 6 (val_acc=0.875) vs. final epoch 12
        # (val_acc=0.312) in the run that surfaced this bug. Reload the
        # correct best-epoch weights from disk before capturing
        # best_state_dict below, or this function ends up re-saving the
        # worse, post-early-stopping weights over the correct checkpoint.
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        if best_history is None or (
            history["degenerate"] is False and best_history["degenerate"] is True
        ) or (
            history["degenerate"] == best_history["degenerate"]
            and history["best_val_macro_f1"] > best_history["best_val_macro_f1"]
        ):
            best_history = history
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        if not history["degenerate"]:
            break

    if best_history["degenerate"]:
        logger.warning(
            "All %d attempt(s) produced a degenerate model for this backbone "
            "(best val_macro_f1=%.3f). Treat results from this backbone with extra caution.",
            max_attempts, best_history["best_val_macro_f1"],
        )
    torch.save(best_state_dict, checkpoint_path)

    history_path = config.METRICS_DIR / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(best_history, f, indent=2)
    logger.info("Training history saved to %s", history_path)
    logger.info(
        "Best checkpoint (val_acc=%.3f, val_macro_f1=%.3f, degenerate=%s) saved to %s",
        best_history["best_val_acc"], best_history["best_val_macro_f1"],
        best_history["degenerate"], checkpoint_path,
    )

    return best_history


if __name__ == "__main__":
    from . import data_acquisition, splitting, augmentation, dataset
    from .models.cnn2d import CNN2D

    manifest = data_acquisition.build_manifest()
    split_result = splitting.subject_stratified_split(manifest)
    loaders = dataset.build_dataloaders(
        manifest, split_result,
        train_transform=augmentation.get_train_transform(),
        eval_transform=augmentation.get_eval_transform(),
    )

    model = CNN2D(num_classes=len(config.CLASS_NAMES), dropout_p=config.DROPOUT_P)
    class_weights = class_weights_from_manifest(manifest, split_result, num_classes=len(config.CLASS_NAMES))
    history = train_model(model, loaders["train"], loaders["val"], class_weights=class_weights)