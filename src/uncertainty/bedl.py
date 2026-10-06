"""
Module 9 - Bayesian Evidential Deep Learning (Algorithm 7)
==============================================================
Implements Algorithm 7 following Sensoy, Kaplan & Kandemir (2018): replace
the softmax output with a non-negative "evidence" vector, interpret
evidence+1 as the concentration parameters of a Dirichlet distribution
over the class simplex, and train with an expected-cross-entropy loss
plus an annealed KL-divergence regularizer that pushes evidence for
*incorrect* classes toward zero. This gives a single deterministic
forward pass that separates:
  - belief mass b_k        (how much evidence supports class k)
  - vacuity / uncertainty u (how much of the simplex is unassigned,
                              i.e. "the model doesn't know")
without any stochastic sampling (unlike MC Dropout) or multiple models
(unlike Deep Ensembles).

ARCHITECTURE NOTE: no new model class is needed. CNN2D / DenseNet2D /
ViT2D's final nn.Linear layer already outputs raw, un-activated scores per
class -- exactly what an evidence head needs. BEDL only changes (a) the
loss function used during training and (b) how those raw outputs are
interpreted at inference time (softplus -> evidence -> Dirichlet, instead
of softmax -> probability). This is why the same three backbone classes
plug into MC Dropout, Deep Ensembles, and BEDL alike.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evidence / Dirichlet mechanics
# ---------------------------------------------------------------------------

def logits_to_alpha(logits: torch.Tensor) -> torch.Tensor:
    """Non-negative evidence via softplus, then alpha = evidence + 1
    (Algorithm 7's Evidence Layer -> Dirichlet Parameters step)."""
    evidence = F.softplus(logits)
    return evidence + 1.0


def dirichlet_expected_prob(alpha: torch.Tensor) -> torch.Tensor:
    """p_k = alpha_k / S, the expected probability under the Dirichlet."""
    S = alpha.sum(dim=1, keepdim=True)
    return alpha / S


def dirichlet_vacuity(alpha: torch.Tensor) -> torch.Tensor:
    """u = K / S, the "I don't know" uncertainty mass (Algorithm 7's
    epistemic-uncertainty output)."""
    K = alpha.shape[1]
    S = alpha.sum(dim=1, keepdim=True).squeeze(1)
    return K / S


def dirichlet_aleatoric(alpha: torch.Tensor) -> torch.Tensor:
    """Expected data (aleatoric) uncertainty under the Dirichlet: the
    mean per-class variance of a Categorical draw from p ~ Dir(alpha),
    i.e. E[p_k(1-p_k)] summed then normalized -- the aleatoric counterpart
    to the vacuity's epistemic reading."""
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    var = p * (1 - p) / (S + 1.0)
    return var.sum(dim=1)


# ---------------------------------------------------------------------------
# Evidential loss (Sensoy et al. 2018, Eq. 5 + Eq. 8 KL regularizer)
# ---------------------------------------------------------------------------

def _kl_dirichlet_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL(Dir(alpha) || Dir(1,...,1)), used to penalize evidence assigned
    to classes that are NOT the ground truth (misleading evidence)."""
    K = alpha.shape[1]
    beta = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=1, keepdim=True)
    S_beta = beta.sum(dim=1, keepdim=True)

    lgamma = torch.lgamma
    term1 = lgamma(S_alpha).squeeze(1) - lgamma(torch.tensor(float(K)))
    term1 = term1 - lgamma(alpha).sum(dim=1)
    term1 = term1 + lgamma(beta).sum(dim=1) - lgamma(S_beta).squeeze(1)

    term2 = ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(dim=1)
    return term1 + term2


def evidential_loss(logits: torch.Tensor, y: torch.Tensor, num_classes: int,
                     epoch: int, annealing_epochs: int = 10,
                     class_weights: torch.Tensor = None) -> torch.Tensor:
    """Expected cross-entropy under the Dirichlet (digamma form) plus an
    annealed KL term that only penalizes evidence for the wrong classes,
    so the model isn't punished for being confident about the right one
    early in training.

    `class_weights` (inverse class frequency, same as train.py uses for
    CrossEntropyLoss) scales each sample's CE term by its true class's
    weight, so BEDL doesn't collapse to the majority (AD) class the same
    way the unweighted version did."""
    alpha = logits_to_alpha(logits)
    S = alpha.sum(dim=1, keepdim=True)
    y_onehot = F.one_hot(y, num_classes=num_classes).float()

    # Expected cross-entropy: sum_k y_k * (digamma(S) - digamma(alpha_k))
    ce = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
    if class_weights is not None:
        ce = ce * class_weights[y]

    # Remove evidence for the correct class before computing the KL penalty,
    # so only misleading (wrong-class) evidence is regularized.
    alpha_tilde = y_onehot + (1 - y_onehot) * alpha
    kl = _kl_dirichlet_to_uniform(alpha_tilde)

    annealing_coef = min(1.0, epoch / max(annealing_epochs, 1))
    loss = ce + annealing_coef * kl
    return loss.mean()


# ---------------------------------------------------------------------------
# Training (mirrors train.py's structure: accuracy-based checkpointing +
# degenerate-model retry, but with the evidential loss and alpha-based
# predictions instead of cross-entropy/softmax)
# ---------------------------------------------------------------------------

def _run_epoch_bedl(model, loader, optimizer, device, train: bool, epoch: int,
                     class_weights: torch.Tensor = None):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds = []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = evidential_loss(logits, y, num_classes=len(config.CLASS_NAMES), epoch=epoch,
                                    class_weights=class_weights)
            if train:
                loss.backward()
                optimizer.step()
            alpha = logits_to_alpha(logits)
            preds = alpha.argmax(dim=1)
            total_loss += loss.item() * x.size(0)
            correct += (preds == y).sum().item()
            n += x.size(0)
            all_preds.append(preds.detach().cpu())
    all_preds = torch.cat(all_preds).numpy() if all_preds else np.array([])
    return total_loss / n, correct / n, all_preds


def _is_degenerate(val_preds: np.ndarray, val_labels: np.ndarray, num_classes: int) -> bool:
    n_unique_preds = len(np.unique(val_preds))
    if n_unique_preds <= 1:
        return True
    majority_baseline = np.bincount(val_labels, minlength=num_classes).max() / max(len(val_labels), 1)
    val_acc = (val_preds == val_labels).mean()
    return val_acc <= majority_baseline + 1e-9


def _train_bedl_once(model, train_loader, val_loader, epochs, lr, checkpoint_path, device, seed,
                      class_weights: torch.Tensor = None):
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_val_loss = -1.0, float("inf")
    best_val_preds = None
    val_labels = torch.cat([y for _, y in val_loader]).numpy()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _ = _run_epoch_bedl(model, train_loader, optimizer, device, True, epoch,
                                                     class_weights=class_weights)
        val_loss, val_acc, val_preds = _run_epoch_bedl(model, val_loader, optimizer, device, False, epoch,
                                                         class_weights=class_weights)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_acc, best_val_loss = val_acc, val_loss
            best_val_preds = val_preds
            torch.save(model.state_dict(), checkpoint_path)

        logger.info(
            "[BEDL] Epoch %2d/%2d | train_loss=%.4f train_acc=%.3f | val_loss=%.4f val_acc=%.3f%s",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
            "  <- checkpoint saved" if improved else "",
        )

    degenerate = _is_degenerate(best_val_preds, val_labels, num_classes=len(config.CLASS_NAMES))
    history["best_val_acc"] = best_val_acc
    history["best_val_loss"] = best_val_loss
    history["degenerate"] = bool(degenerate)
    history["seed"] = seed
    return history


def train_bedl(model, train_loader, val_loader,
                epochs: int = None,
                lr: float = None,
                checkpoint_path: Path = None,
                device: str = None,
                max_attempts: int = None,
                base_seed: int = None,
                class_weights: torch.Tensor = None,
                force_retrain: bool = False) -> dict:
    """Same retry-on-degenerate contract as train.train_model, but trains
    with the evidential loss and checks degeneracy against alpha-argmax
    predictions instead of softmax-argmax.

    `class_weights`, if not supplied, is computed by iterating
    `train_loader` (backwards-compatible fallback) -- pass a precomputed
    tensor from train.class_weights_from_manifest() to avoid that extra
    pass over the data (Section 10 of the optimization pass).
    """
    epochs = epochs if epochs is not None else config.EPOCHS
    lr = lr if lr is not None else config.LEARNING_RATE
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else config.CHECKPOINT_DIR / "bedl_best.pt"
    device = device if device is not None else str(config.DEVICE)
    max_attempts = max_attempts if max_attempts is not None else config.MAX_ATTEMPTS
    base_seed = base_seed if base_seed is not None else config.RANDOM_SEED

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists() and not force_retrain:
        logger.info("[BEDL] Checkpoint already exists at %s; reusing (pass "
                    "force_retrain=True to retrain).", checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        return {"reused_checkpoint": True, "checkpoint_path": str(checkpoint_path)}

    model_cls = type(model)

    from .. import train as train_mod
    if class_weights is None and config.CLASS_BALANCE_STRATEGY == "loss":
        class_weights = train_mod.compute_class_weights(train_loader, num_classes=len(config.CLASS_NAMES), device=device)
    elif config.CLASS_BALANCE_STRATEGY != "loss":
        class_weights = None
    if class_weights is not None:
        logger.info("[BEDL] Class weights (order=%s): %s",
                    config.CLASS_NAMES, [round(w, 3) for w in class_weights.tolist()])

    best_history, best_state_dict = None, None

    for attempt in range(1, max_attempts + 1):
        seed = base_seed + attempt - 1
        if attempt > 1:
            torch.manual_seed(seed)
            model = model_cls(num_classes=len(config.CLASS_NAMES), dropout_p=config.DROPOUT_P)
            logger.warning("[BEDL] Attempt %d/%d: previous attempt degenerate, "
                            "retraining fresh with seed=%d.", attempt, max_attempts, seed)

        history = _train_bedl_once(model, train_loader, val_loader, epochs, lr, checkpoint_path, device, seed,
                                    class_weights=class_weights)

        if best_history is None or (
            history["degenerate"] is False and best_history["degenerate"] is True
        ) or (
            history["degenerate"] == best_history["degenerate"]
            and history["best_val_acc"] > best_history["best_val_acc"]
        ):
            best_history = history
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        if not history["degenerate"]:
            break

    if best_history["degenerate"]:
        logger.warning("[BEDL] All %d attempt(s) degenerate (best val_acc=%.3f).",
                        max_attempts, best_history["best_val_acc"])
    torch.save(best_state_dict, checkpoint_path)

    history_path = config.METRICS_DIR / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(best_history, f, indent=2)
    logger.info("[BEDL] Best checkpoint (val_acc=%.3f, degenerate=%s) saved to %s",
                best_history["best_val_acc"], best_history["degenerate"], checkpoint_path)
    return best_history


# ---------------------------------------------------------------------------
# Inference -- same return schema as mc_dropout_predict_dataset /
# deep_ensemble_predict_dataset so evaluate.py works unmodified.
# ---------------------------------------------------------------------------

@torch.no_grad()
def bedl_predict(model: nn.Module, x: torch.Tensor) -> dict:
    model.eval()
    logits = model(x)
    alpha = logits_to_alpha(logits)
    p_mean = dirichlet_expected_prob(alpha)
    y_pred = p_mean.argmax(dim=1)
    variance = dirichlet_vacuity(alpha)          # epistemic: "how much don't I know"
    eps = 1e-12
    entropy = -(p_mean * torch.log(p_mean + eps)).sum(dim=1)
    return {"p_mean": p_mean, "y_pred": y_pred, "variance": variance, "entropy": entropy,
            "aleatoric": dirichlet_aleatoric(alpha)}


@torch.no_grad()
def bedl_predict_dataset(model: nn.Module, loader, device: str = "cpu") -> dict:
    model.to(device)
    all_p_mean, all_y_pred, all_variance, all_entropy, all_y_true = [], [], [], [], []
    for x, y in loader:
        x = x.to(device)
        out = bedl_predict(model, x)
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
    train_bedl(model, loaders["train"], loaders["val"], epochs=6)
    results = bedl_predict_dataset(model, loaders["test"])
    print(f"Test accuracy: {(results['y_pred'] == results['y_true']).mean():.3f}")
    print(f"Mean vacuity (epistemic uncertainty): {results['variance'].mean():.4f}")