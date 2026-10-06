"""
Module 11 - Comparative Evaluation (Algorithm 9)
===================================================
Computes the discriminative metrics (accuracy, precision, recall, F1,
ROC-AUC) and calibration metrics (Expected Calibration Error, Brier score,
predictive entropy) specified in Algorithm 9 / the Novelty section of the
project document, for one architecture-uncertainty combination
(CNN2D + Monte Carlo Dropout, the only combination currently implemented --
see README.md for the other eight).

This currently evaluates ONE of the nine planned combinations. The
statistical cross-combination comparison and Decision Layer (Best-Combination
Selector) described in the project document require the other eight
combinations to exist first and are explicitly out of scope for this
increment.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, brier_score_loss,
)

def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-class precision/recall/F1/support -- required for honest
    reporting on an imbalanced 3-class medical staging problem (Section 11
    of the optimization pass); macro averages alone can hide a class the
    model never predicts correctly."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(config.CLASS_NAMES))), zero_division=0
    )
    return {
        config.CLASS_NAMES[i]: {
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i]),
        }
        for i in range(len(config.CLASS_NAMES))
    }

from . import config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_discriminative_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                    p_mean: np.ndarray) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(y_true, p_mean, multi_class="ovr", average="macro")
    except ValueError as e:
        logger.warning("ROC-AUC could not be computed (%s); reporting NaN. "
                        "This happens when a class is absent from a tiny test split.", e)
        roc_auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "roc_auc_macro_ovr": roc_auc,
    }


def _one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out


def expected_calibration_error(y_true: np.ndarray, p_mean: np.ndarray, n_bins: int = 10) -> float:
    """Multiclass ECE: bin by the model's top predicted-class confidence,
    compare average confidence to actual accuracy within each bin."""
    confidences = p_mean.max(axis=1)
    predictions = p_mean.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def multiclass_brier_score(y_true: np.ndarray, p_mean: np.ndarray) -> float:
    y_onehot = _one_hot(y_true, p_mean.shape[1])
    return float(np.mean(np.sum((p_mean - y_onehot) ** 2, axis=1)))


def compute_calibration_metrics(y_true: np.ndarray, p_mean: np.ndarray,
                                 entropy: np.ndarray, variance: np.ndarray) -> dict:
    return {
        "expected_calibration_error": expected_calibration_error(y_true, p_mean),
        "brier_score": multiclass_brier_score(y_true, p_mean),
        "mean_predictive_entropy": float(entropy.mean()),
        "mean_predictive_variance": float(variance.mean()),
    }


def risk_coverage_curve(y_true: np.ndarray, y_pred: np.ndarray,
                         uncertainty: np.ndarray, n_points: int = 20) -> pd.DataFrame:
    """Sort by ascending uncertainty; at each coverage level, report the
    error rate among the most-confident fraction of predictions."""
    order = np.argsort(uncertainty)
    correct = (y_pred == y_true).astype(float)[order]
    n = len(correct)
    coverages = np.linspace(1.0 / n_points, 1.0, n_points)
    rows = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        risk = 1.0 - correct[:k].mean()
        rows.append({"coverage": cov, "risk": risk})
    return pd.DataFrame(rows)


def plot_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(config.CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(config.CLASS_NAMES)))
    ax.set_yticks(range(len(config.CLASS_NAMES)))
    ax.set_xticklabels(config.CLASS_NAMES)
    ax.set_yticklabels(config.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reliability_diagram(y_true, p_mean, out_path: Path, n_bins: int = 10):
    confidences = p_mean.max(axis=1)
    predictions = p_mean.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_acc, bin_conf = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_acc.append(correct[mask].mean())
        bin_conf.append(confidences[mask].mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.bar(bin_centers, bin_acc, width=1.0 / n_bins, alpha=0.7, edgecolor="black", label="Accuracy in bin")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_risk_coverage(rc_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rc_df["coverage"], rc_df["risk"], marker="o")
    ax.set_xlabel("Coverage (fraction of most-confident predictions kept)")
    ax.set_ylabel("Risk (error rate)")
    ax.set_title("Risk-Coverage Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

@torch.no_grad()
def deterministic_predict_dataset(model, loader, device="cpu"):
    model.to(device)
    model.eval()

    all_p_mean = []
    all_y_pred = []
    all_y_true = []

    for x, y in loader:
        x = x.to(device)

        logits = model(x)
        p_mean = torch.softmax(logits, dim=1)
        y_pred = p_mean.argmax(dim=1)

        all_p_mean.append(p_mean.cpu())
        all_y_pred.append(y_pred.cpu())
        all_y_true.append(y)

    return {
        "p_mean": torch.cat(all_p_mean).numpy(),
        "y_pred": torch.cat(all_y_pred).numpy(),
        "y_true": torch.cat(all_y_true).numpy(),
    }

def run_full_evaluation(mc_results: dict, combination_name: str = "CNN2D + MC_Dropout") -> dict:
    y_true = mc_results["y_true"]
    y_pred = mc_results["y_pred"]
    p_mean = mc_results["p_mean"]
    entropy = mc_results["entropy"]
    variance = mc_results["variance"]

    discriminative = compute_discriminative_metrics(y_true, y_pred, p_mean)
    calibration = compute_calibration_metrics(y_true, p_mean, entropy, variance)

    plot_confusion_matrix(y_true, y_pred, config.PLOTS_DIR / "confusion_matrix.png")
    plot_reliability_diagram(y_true, p_mean, config.PLOTS_DIR / "reliability_diagram.png")
    rc_df = risk_coverage_curve(y_true, y_pred, variance)
    plot_risk_coverage(rc_df, config.PLOTS_DIR / "risk_coverage_curve.png")
    rc_df.to_csv(config.METRICS_DIR / "risk_coverage_curve.csv", index=False)

    report = {"combination": combination_name, **discriminative, **calibration}

    metrics_path = config.METRICS_DIR / "evaluation_report.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation report saved to %s", metrics_path)
    logger.info("Plots saved to %s", config.PLOTS_DIR)
    return report


def evaluate_combination(mc_results: dict, combination_name: str,
                          make_plots: bool = False) -> dict:
    """Lightweight version of run_full_evaluation for use inside a
    multi-combination comparison loop: computes the metric dict for one
    (backbone, uncertainty method) combination without necessarily
    overwriting the single-combination plot files each time."""
    y_true = mc_results["y_true"]
    y_pred = mc_results["y_pred"]
    p_mean = mc_results["p_mean"]
    entropy = mc_results["entropy"]
    variance = mc_results["variance"]

    discriminative = compute_discriminative_metrics(y_true, y_pred, p_mean)
    calibration = compute_calibration_metrics(y_true, p_mean, entropy, variance)
    report = {"combination": combination_name, **discriminative, **calibration}

    if make_plots:
        safe_name = combination_name.replace(" ", "_").replace("+", "").replace("__", "_")
        plot_confusion_matrix(y_true, y_pred, config.PLOTS_DIR / f"confusion_matrix_{safe_name}.png")
        plot_reliability_diagram(y_true, p_mean, config.PLOTS_DIR / f"reliability_diagram_{safe_name}.png")

    return report


def compare_combinations(reports: list) -> pd.DataFrame:
    """Algorithm 9's cross-combination comparison table: one row per
    (backbone, uncertainty method) combination that has actually been run."""
    df = pd.DataFrame(reports).set_index("combination")
    path = config.METRICS_DIR / "combination_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.info("Combination comparison table saved to %s", path)
    return df


def statistical_comparison(all_results: dict) -> pd.DataFrame:
    """Statistical Comparator (named in the project's Evaluation Layer):
    paired significance testing between the best-accuracy combination and
    every other combination, on a per-sample basis.

    Uses per-sample correctness (0/1) as the paired variable -- the same
    test items are scored by every combination since they all share the
    same held-out test split, which is exactly the "matched conditions"
    requirement Algorithm 9 specifies. Runs both a paired t-test and a
    Wilcoxon signed-rank test (the doc names both as options); Wilcoxon
    doesn't assume normality of the paired differences, which matters
    with only ~72 paired samples.

    Parameters
    ----------
    all_results: dict mapping combination_name -> result dict (must
        contain 'y_true' and 'y_pred', same schema mc_dropout_predict_dataset
        / deep_ensemble_predict_dataset / bedl_predict_dataset all return).
    """
    from scipy import stats

    names = list(all_results.keys())
    accuracies = {name: (all_results[name]["y_pred"] == all_results[name]["y_true"]).mean()
                  for name in names}
    best_name = max(accuracies, key=accuracies.get)
    best_correct = (all_results[best_name]["y_pred"] == all_results[best_name]["y_true"]).astype(int)

    rows = []
    for name in names:
        if name == best_name:
            continue
        other_correct = (all_results[name]["y_pred"] == all_results[name]["y_true"]).astype(int)
        diff = best_correct - other_correct

        if np.all(diff == 0):
            t_stat, t_p = 0.0, 1.0
            w_stat, w_p = 0.0, 1.0
        else:
            t_stat, t_p = stats.ttest_rel(best_correct, other_correct)
            try:
                w_stat, w_p = stats.wilcoxon(best_correct, other_correct)
            except ValueError:
                # Wilcoxon fails if all differences are zero after ties removed
                w_stat, w_p = 0.0, 1.0

        rows.append({
            "compared_to": name,
            "best_combination": best_name,
            "accuracy_best": accuracies[best_name],
            "accuracy_other": accuracies[name],
            "paired_t_statistic": float(t_stat),
            "paired_t_pvalue": float(t_p),
            "wilcoxon_statistic": float(w_stat),
            "wilcoxon_pvalue": float(w_p),
            "significant_at_0.05": bool(t_p < 0.05 or w_p < 0.05),
        })

    df = pd.DataFrame(rows)
    path = config.METRICS_DIR / "statistical_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Statistical comparison (paired t-test + Wilcoxon vs. best combination) saved to %s", path)
    return df


def select_best_combination(comparison_df: pd.DataFrame,
                             accuracy_weight: float = 0.6,
                             calibration_weight: float = 0.4,
                             stats_df: pd.DataFrame = None) -> dict:
    """Best-Combination Selector / Decision Layer.

    Ranks by a weighted score (normalized accuracy minus normalized ECE),
    then, if `stats_df` (from statistical_comparison()) is supplied,
    reports whether the recommended combination's advantage over each
    runner-up is statistically significant (paired t-test / Wilcoxon,
    alpha=0.05) rather than only reporting the score itself.
    """
    df = comparison_df.copy()
    acc_norm = (df["accuracy"] - df["accuracy"].min()) / (df["accuracy"].max() - df["accuracy"].min() + 1e-12)
    ece_norm = (df["expected_calibration_error"] - df["expected_calibration_error"].min()) / (
        df["expected_calibration_error"].max() - df["expected_calibration_error"].min() + 1e-12
    )
    score = accuracy_weight * acc_norm - calibration_weight * ece_norm
    df["decision_score"] = score
    best = df["decision_score"].idxmax()

    result = {
        "recommended_combination": best,
        "decision_score": float(df.loc[best, "decision_score"]),
        "rationale": (
            f"Highest weighted score ({accuracy_weight:.0%} accuracy, "
            f"{calibration_weight:.0%} calibration-penalty) among the "
            f"{len(df)} combination(s) evaluated so far."
        ),
        "scored_combinations": df["decision_score"].round(4).to_dict(),
    }

    if stats_df is not None and len(stats_df) > 0:
        sig_rows = stats_df[stats_df["best_combination"] == best]
        result["statistical_significance_vs_runner_ups"] = {
            row["compared_to"]: {
                "significant_at_0.05": bool(row["significant_at_0.05"]),
                "paired_t_pvalue": round(float(row["paired_t_pvalue"]), 4),
                "wilcoxon_pvalue": round(float(row["wilcoxon_pvalue"]), 4),
            }
            for _, row in sig_rows.iterrows()
        }
        n_significant = sum(v["significant_at_0.05"] for v in result["statistical_significance_vs_runner_ups"].values())
        result["significance_summary"] = (
            f"Recommended combination's accuracy advantage is statistically "
            f"significant (p<0.05) over {n_significant}/{len(sig_rows)} other "
            f"evaluated combinations. With a small paired test set on this demo "
            f"dataset, a non-significant result often just reflects limited "
            f"sample size, not equivalent performance."
        )

    path = config.METRICS_DIR / "decision_layer_recommendation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Decision layer recommendation saved to %s", path)
    return result


if __name__ == "__main__":
    import torch
    from . import data_acquisition, splitting, augmentation, dataset
    from .models.cnn2d import CNN2D
    from .uncertainty.mc_dropout import mc_dropout_predict_dataset

    manifest = data_acquisition.build_manifest()
    split_result = splitting.subject_stratified_split(manifest)
    loaders = dataset.build_dataloaders(
        manifest, split_result,
        train_transform=augmentation.get_train_transform(),
        eval_transform=augmentation.get_eval_transform(),
    )

    model = CNN2D(num_classes=len(config.CLASS_NAMES), dropout_p=config.DROPOUT_P)
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / "cnn2d_best.pt", map_location="cpu"))

    mc_results = mc_dropout_predict_dataset(model, loaders["test"], T=config.MC_DROPOUT_SAMPLES)
    report = run_full_evaluation(mc_results)
    print(json.dumps(report, indent=2))