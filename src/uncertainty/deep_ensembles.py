"""
Module - Deep Ensembles (Algorithm 8)
========================================
Implements Algorithm 8 exactly as specified: train M independently
initialized models (different random seeds, different data shuffling),
then at inference use disagreement across the M members' softmax outputs
as the uncertainty signal -- no stochastic test-time sampling required
(unlike MC Dropout), each member gives one deterministic prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .. import config
from ..train import train_model

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_ensemble(model_fn, train_loader, val_loader,
                    M: int = None, epochs: int = None,
                    checkpoint_dir: Path = None,
                    base_seed: int = None,
                    force_retrain: bool = False) -> list:
    """Train M independently-seeded copies of model_fn() and checkpoint each.

    model_fn: a zero-arg callable returning a fresh, untrained nn.Module
              instance (e.g. lambda: CNN2D(num_classes=4)).
    Returns the list of checkpoint paths, one per ensemble member.

    If `force_retrain` is False (default) and a member's checkpoint file
    already exists, that member is NOT retrained -- see Section 10 of the
    optimization pass (avoid redundant training).
    """
    M = M if M is not None else config.ENSEMBLE_SIZE
    epochs = epochs if epochs is not None else config.EPOCHS
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else config.CHECKPOINT_DIR / "ensemble"
    base_seed = base_seed if base_seed is not None else config.RANDOM_SEED
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = []
    for m in range(M):
        ckpt_path = checkpoint_dir / f"member_{m}.pt"
        if ckpt_path.exists() and not force_retrain:
            logger.info("Ensemble member %d/%d checkpoint already exists at %s; reusing "
                        "(pass force_retrain=True to retrain).", m + 1, M, ckpt_path)
            checkpoint_paths.append(ckpt_path)
            continue
        # BUG FIX: previously this seed was only used for the outer
        # torch.manual_seed() call below and never reached train_model's
        # internal retry logic, which defaulted base_seed to
        # config.RANDOM_SEED for every member -- so any member that needed
        # a retry (degenerate first attempt) used the exact same retry
        # seeds as every other member, producing bit-identical, non-diverse
        # "ensemble" members. Each member now gets its own seed block
        # (spaced by 1000, well clear of max_attempts) and that block is
        # passed all the way through to train_model via base_seed.
        seed_block_start = base_seed + m * 1000
        torch.manual_seed(seed_block_start)
        logger.info("Training ensemble member %d/%d (seed block starting at %d)",
                    m + 1, M, seed_block_start)
        model = model_fn()
        train_model(model, train_loader, val_loader, epochs=epochs, checkpoint_path=ckpt_path,
                    base_seed=seed_block_start)
        checkpoint_paths.append(ckpt_path)

    logger.info("Ensemble training complete: %d members saved to %s", M, checkpoint_dir)
    return checkpoint_paths


def load_ensemble_members(model_fn, checkpoint_paths: list, device: str = "cpu") -> list:
    """Load every ensemble member from disk ONCE. Callers that need to
    predict on many individual samples (e.g. the visualization loop) should
    load once via this function and reuse the list, rather than reloading
    from disk on every sample (see deep_ensemble_predict_with_members)."""
    members = []
    for ckpt_path in checkpoint_paths:
        model = model_fn()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        members.append(model)
    return members


@torch.no_grad()
def deep_ensemble_predict_with_members(members: list, x: torch.Tensor, device: str = "cpu") -> dict:
    """Same mechanics as deep_ensemble_predict, but takes already-loaded
    members instead of reloading checkpoints from disk on every call."""
    x = x.to(device)
    member_probs = [torch.softmax(m(x), dim=1) for m in members]
    probs_stack = torch.stack(member_probs, dim=0)  # (M, B, C)

    p_mean = probs_stack.mean(dim=0)
    y_pred = p_mean.argmax(dim=1)
    variance = probs_stack.var(dim=0).mean(dim=1)
    eps = 1e-12
    entropy = -(p_mean * torch.log(p_mean + eps)).sum(dim=1)
    return {"p_mean": p_mean, "y_pred": y_pred, "variance": variance, "entropy": entropy}


@torch.no_grad()
def deep_ensemble_predict(model_fn, checkpoint_paths: list, x: torch.Tensor,
                           device: str = "cpu") -> dict:
    """Single-batch version of deep_ensemble_predict_dataset -- same
    (deterministic, one forward pass per member) mechanics. Prefer
    load_ensemble_members() + deep_ensemble_predict_with_members() when
    calling this repeatedly (e.g. once per visualization sample), so
    checkpoints aren't reloaded from disk on every call."""
    members = load_ensemble_members(model_fn, checkpoint_paths, device=device)
    x = x.to(device)
    member_probs = [torch.softmax(m(x), dim=1) for m in members]
    probs_stack = torch.stack(member_probs, dim=0)  # (M, B, C)

    p_mean = probs_stack.mean(dim=0)
    y_pred = p_mean.argmax(dim=1)
    variance = probs_stack.var(dim=0).mean(dim=1)
    eps = 1e-12
    entropy = -(p_mean * torch.log(p_mean + eps)).sum(dim=1)
    return {"p_mean": p_mean, "y_pred": y_pred, "variance": variance, "entropy": entropy}


@torch.no_grad()
def deep_ensemble_predict_dataset(model_fn, checkpoint_paths: list, loader,
                                   device: str = "cpu") -> dict:
    """Run every ensemble member once (deterministic forward pass, eval
    mode, no dropout sampling) over the dataset and combine into the same
    result schema mc_dropout_predict_dataset uses, so evaluate.py works
    unmodified against either uncertainty method."""
    members = []
    for ckpt_path in checkpoint_paths:
        model = model_fn()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        members.append(model)

    all_p_mean, all_variance, all_entropy, all_y_true, all_y_pred = [], [], [], [], []

    for x, y in loader:
        x = x.to(device)
        member_probs = []
        for model in members:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            member_probs.append(probs)
        probs_stack = torch.stack(member_probs, dim=0)  # (M, B, C)

        p_mean = probs_stack.mean(dim=0)
        y_pred = p_mean.argmax(dim=1)
        variance = probs_stack.var(dim=0).mean(dim=1)
        eps = 1e-12
        entropy = -(p_mean * torch.log(p_mean + eps)).sum(dim=1)

        all_p_mean.append(p_mean.cpu())
        all_y_pred.append(y_pred.cpu())
        all_variance.append(variance.cpu())
        all_entropy.append(entropy.cpu())
        all_y_true.append(y)

    return {
        "p_mean": torch.cat(all_p_mean).numpy(),
        "y_pred": torch.cat(all_y_pred).numpy(),
        "variance": torch.cat(all_variance).numpy(),
        "entropy": torch.cat(all_entropy).numpy(),
        "y_true": torch.cat(all_y_true).numpy(),
    }


if __name__ == "__main__":
    from .. import data_acquisition, splitting, augmentation, dataset
    from ..models.cnn2d import CNN2D

    manifest = data_acquisition.build_manifest()
    split_result = splitting.subject_stratified_split(manifest)
    loaders = dataset.build_dataloaders(
        manifest, split_result,
        train_transform=augmentation.get_train_transform(),
        eval_transform=augmentation.get_eval_transform(),
    )

    model_fn = lambda: CNN2D(num_classes=len(config.CLASS_NAMES), dropout_p=config.DROPOUT_P)
    ckpts = train_ensemble(model_fn, loaders["train"], loaders["val"], M=3, epochs=6)
    results = deep_ensemble_predict_dataset(model_fn, ckpts, loaders["test"])
    print(f"Ensemble of {len(ckpts)} members on {len(results['y_true'])} test samples")
    print(f"Mean predictive variance: {results['variance'].mean():.5f}")
    print(f"Mean predictive entropy:  {results['entropy'].mean():.5f}")