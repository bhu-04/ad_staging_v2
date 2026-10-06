"""
Module - Monte Carlo Dropout (Algorithm 6)
=============================================
Implements Algorithm 6 exactly as specified: T stochastic forward passes at
inference time with dropout layers active while every other layer
(BatchNorm included) stays in eval mode, using the spread of the resulting
predictions as the uncertainty signal.

PERF NOTES (Section 9 of the optimization pass):
  - T defaults to config.MC_DROPOUT_SAMPLES, resolved lazily so
    config.configure("demo"/"full") is respected (demo: 12, full: 30 by
    default -- see config.py).
  - The T stochastic passes are now done as ONE batched forward pass
    (the input is tiled T times along the batch dimension) instead of a
    Python loop of T sequential forward() calls. This cuts per-call Python
    /autograd-engine overhead roughly T-fold and lets the backend batch
    the matmuls; the predictive mean is numerically identical to running
    T separate passes since dropout masks are independent per sample.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config


def enable_mc_dropout(model: nn.Module) -> None:
    """Set the whole model to eval mode, then re-enable train-mode (active)
    behaviour on Dropout layers only -- matching Algorithm 6's
    Initialization step precisely."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor,
                        T: int = None) -> dict:
    """Run T stochastic forward passes for a batch and return the
    predictive mean, predicted label, and predictive variance
    (uncertainty score), per Algorithm 6.

    Returns
    -------
    dict with keys:
        p_mean:  (B, num_classes) mean softmax probability across T passes
        y_pred:  (B,) argmax(p_mean)
        variance: (B,) scalar per-sample uncertainty = mean class-wise
                  variance across T stochastic passes
        entropy: (B,) predictive entropy of p_mean
    """
    T = T if T is not None else config.MC_DROPOUT_SAMPLES
    enable_mc_dropout(model)

    B = x.shape[0]
    # Batch all T stochastic passes into one forward call: (T*B, ...).
    # Dropout masks are sampled independently per row, so this is exactly
    # equivalent to T separate forward() calls, just without the Python
    # loop overhead.
    x_tiled = x.repeat(T, *([1] * (x.dim() - 1)))
    logits = model(x_tiled)
    probs = F.softmax(logits, dim=1).view(T, B, -1)  # (T, B, C)

    p_mean = probs.mean(dim=0)                # (B, C)
    y_pred = p_mean.argmax(dim=1)              # (B,)
    variance = probs.var(dim=0).mean(dim=1)    # (B,)

    eps = 1e-12
    entropy = -(p_mean * torch.log(p_mean + eps)).sum(dim=1)  # (B,)

    return {
        "p_mean": p_mean,
        "y_pred": y_pred,
        "variance": variance,
        "entropy": entropy,
    }


def mc_dropout_predict_dataset(model: nn.Module, loader, device=None,
                                T: int = None) -> dict:
    """Run MC Dropout inference over an entire DataLoader and concatenate
    results, alongside ground-truth labels."""
    device = device if device is not None else config.DEVICE
    model.to(device)
    all_p_mean, all_y_pred, all_variance, all_entropy, all_y_true = [], [], [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = mc_dropout_predict(model, x, T=T)
        all_p_mean.append(out["p_mean"].cpu())
        all_y_pred.append(out["y_pred"].cpu())
        all_variance.append(out["variance"].cpu())
        all_entropy.append(out["entropy"].cpu())
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

    model = CNN2D(num_classes=len(config.CLASS_NAMES), dropout_p=config.DROPOUT_P)
    ckpt = config.CHECKPOINT_DIR / "cnn2d_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    results = mc_dropout_predict_dataset(model, loaders["test"], T=config.MC_DROPOUT_SAMPLES)
    print(f"Ran MC Dropout with T={config.MC_DROPOUT_SAMPLES} on {len(results['y_true'])} test samples")
    print(f"Mean predictive variance: {results['variance'].mean():.5f}")
    print(f"Mean predictive entropy:  {results['entropy'].mean():.5f}")
