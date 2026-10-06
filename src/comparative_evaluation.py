"""
Comparative Model Base Selection (Algorithm 9 + Decision Layer)
===================================================================
Trains all 3 backbones (CNN2D, DenseNet2D, ViT2D) x all 3 uncertainty
methods (MC Dropout, Deep Ensembles, BEDL) = 9 combinations, evaluates
every one under identical conditions, runs the statistical comparator,
and selects a recommended combination.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch

from . import config, data_acquisition, splitting, augmentation, dataset, train, evaluate, checkpoint_utils
from .models.cnn2d import CNN2D
from .models.densenet2d import DenseNet2D
from .models.vit2d import ViT2D
from .uncertainty.mc_dropout import mc_dropout_predict_dataset
from .uncertainty import deep_ensembles
from .uncertainty import bedl as bedl_mod

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BACKBONES = {
    "CNN2D": lambda: CNN2D(num_classes=len(config.CLASS_NAMES), dropout_p=config.DROPOUT_P),
    "DenseNet2D": lambda: DenseNet2D(num_classes=len(config.CLASS_NAMES), dropout_p=config.DROPOUT_P),
    "ViT2D": lambda: ViT2D(num_classes=len(config.CLASS_NAMES), dropout_p=config.DROPOUT_P),
}


def run_comparative_study(loaders: dict, manifest=None, split_result: dict = None,
                           force_retrain: bool = False) -> dict:
    """Train every backbone once, then attach MC Dropout, Deep Ensembles,
    and BEDL to each -- 9 combinations total -- evaluate every one, run
    the statistical comparator against the best, and return everything
    needed downstream (checkpoints, reports, comparison table, stats,
    decision).

    If `manifest`/`split_result` are given, class weights are computed
    once from the manifest (no DataLoader iteration) and reused for every
    training call. If `force_retrain` is False (default), any combination
    whose checkpoint already exists on disk is loaded instead of retrained
    (Section 10 of the optimization pass)."""

    reports = []
    checkpoints = {}
    timings = {}
    all_results = {}   # combination_name -> raw prediction dict (for stats_df)

    class_weights = None
    if manifest is not None and split_result is not None and config.CLASS_BALANCE_STRATEGY == "loss":
        class_weights = train.class_weights_from_manifest(
            manifest, split_result, num_classes=len(config.CLASS_NAMES)
        )

    for name, model_fn in BACKBONES.items():
        # --- Train the plain backbone once, reuse for MC Dropout ---
        logger.info("=" * 60)
        logger.info("Training backbone: %s", name)
        logger.info("=" * 60)
        model = model_fn()
        ckpt_path = config.CHECKPOINT_DIR / f"{name.lower()}_best.pt"

        t0 = time.time()
        if ckpt_path.exists() and not force_retrain:
            logger.info("Checkpoint already exists at %s; reusing (pass "
                        "force_retrain=True to retrain).", ckpt_path)
        else:
            train.train_model(model, loaders["train"], loaders["val"], checkpoint_path=ckpt_path,
                               class_weights=class_weights)
        train_time = time.time() - t0

        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        checkpoints[f"{name}+MC_Dropout"] = {"path": ckpt_path}

        t0 = time.time()
        mc_results = mc_dropout_predict_dataset(model, loaders["test"], T=config.MC_DROPOUT_SAMPLES)
        infer_time = (time.time() - t0) / max(len(mc_results["y_true"]), 1)

        combo_name = f"{name}+MC_Dropout"
        report = evaluate.evaluate_combination(mc_results, combo_name, make_plots=True)
        report["train_time_sec"] = round(train_time, 2)
        report["inference_time_sec_per_sample"] = round(infer_time, 4)
        reports.append(report)
        timings[combo_name] = {"train_time_sec": train_time, "inference_time_sec_per_sample": infer_time}
        all_results[combo_name] = mc_results

        # --- Deep Ensembles for this backbone ---
        logger.info("=" * 60)
        logger.info("Training Deep Ensemble (backbone: %s, M=%d)", name, config.ENSEMBLE_SIZE)
        logger.info("=" * 60)
        ens_ckpt_dir = config.CHECKPOINT_DIR / "ensemble" / name.lower()
        t0 = time.time()
        ensemble_ckpts = deep_ensembles.train_ensemble(
            model_fn, loaders["train"], loaders["val"],
            M=config.ENSEMBLE_SIZE, epochs=config.EPOCHS,
            checkpoint_dir=ens_ckpt_dir, force_retrain=force_retrain,
        )
        ens_train_time = time.time() - t0

        t0 = time.time()
        ens_results = deep_ensembles.deep_ensemble_predict_dataset(model_fn, ensemble_ckpts, loaders["test"])
        ens_infer_time = (time.time() - t0) / max(len(ens_results["y_true"]), 1)

        combo_name = f"{name}+Deep_Ensembles"
        checkpoints[combo_name] = {"paths": ensemble_ckpts}
        report = evaluate.evaluate_combination(ens_results, combo_name, make_plots=True)
        report["train_time_sec"] = round(ens_train_time, 2)
        report["inference_time_sec_per_sample"] = round(ens_infer_time, 4)
        reports.append(report)
        timings[combo_name] = {"train_time_sec": ens_train_time, "inference_time_sec_per_sample": ens_infer_time}
        all_results[combo_name] = ens_results

        # --- BEDL for this backbone ---
        logger.info("=" * 60)
        logger.info("Training BEDL (backbone: %s)", name)
        logger.info("=" * 60)
        bedl_model = model_fn()
        bedl_ckpt_path = config.CHECKPOINT_DIR / f"{name.lower()}_bedl_best.pt"
        t0 = time.time()
        bedl_mod.train_bedl(bedl_model, loaders["train"], loaders["val"], checkpoint_path=bedl_ckpt_path,
                             class_weights=class_weights, force_retrain=force_retrain)
        bedl_train_time = time.time() - t0

        bedl_model.load_state_dict(torch.load(bedl_ckpt_path, map_location="cpu"))
        combo_name = f"{name}+BEDL"
        checkpoints[combo_name] = {"path": bedl_ckpt_path}

        t0 = time.time()
        bedl_results = bedl_mod.bedl_predict_dataset(bedl_model, loaders["test"])
        bedl_infer_time = (time.time() - t0) / max(len(bedl_results["y_true"]), 1)

        report = evaluate.evaluate_combination(bedl_results, combo_name, make_plots=True)
        report["train_time_sec"] = round(bedl_train_time, 2)
        report["inference_time_sec_per_sample"] = round(bedl_infer_time, 4)
        reports.append(report)
        timings[combo_name] = {"train_time_sec": bedl_train_time, "inference_time_sec_per_sample": bedl_infer_time}
        all_results[combo_name] = bedl_results

    # --- Compare, run stats, and select ---
    comparison_df = evaluate.compare_combinations(reports)
    stats_df = evaluate.statistical_comparison(all_results)
    decision = evaluate.select_best_combination(comparison_df, stats_df=stats_df)

    return {
        "comparison_df": comparison_df,
        "stats_df": stats_df,
        "decision": decision,
        "checkpoints": checkpoints,
        "reports": reports,
        "timings": timings,
        "all_results": all_results,
    }


def plot_comparison_bars(comparison_df: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    comparison_df["accuracy"].plot(kind="bar", ax=axes[0], color="#4c72b0")
    axes[0].set_title("Accuracy by combination")
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_ylim(0, 1)

    comparison_df["expected_calibration_error"].plot(kind="bar", ax=axes[1], color="#dd8452")
    axes[1].set_title("Expected Calibration Error by combination\n(lower is better)")
    axes[1].set_ylabel("ECE")
    axes[1].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    manifest = data_acquisition.build_manifest()
    data_acquisition.save_manifest(manifest)
    split_result = splitting.subject_stratified_split(manifest)
    splitting.save_splits(split_result)
    loaders = dataset.build_dataloaders(
        manifest, split_result,
        train_transform=augmentation.get_train_transform(),
        eval_transform=augmentation.get_eval_transform(),
    )

    results = run_comparative_study(loaders, manifest=manifest, split_result=split_result)
    print("\nComparison table:")
    print(results["comparison_df"].round(4))
    print("\nStatistical comparison:")
    print(results["stats_df"].round(4))
    print("\nDecision:")
    print(json.dumps(results["decision"], indent=2))

    plot_comparison_bars(results["comparison_df"], config.PLOTS_DIR / "combination_comparison.png")