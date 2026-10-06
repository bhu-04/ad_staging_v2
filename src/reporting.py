"""Comprehensive reporting for the Team-4 results document.

This module converts the raw predictions produced by the 3x3 benchmark into
all tables/statistics/figures requested in TEAM - 4 Results.docx.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, roc_curve, auc,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config

METHOD_LABELS = {
    "MC_Dropout": "MC Dropout",
    "Deep_Ensembles": "Deep Ensemble",
    "BEDL": "Evidential DL",
}
ARCH_LABELS = {
    "CNN2D": "3D CNN",
    "DenseNet2D": "DenseNet-121",
    "ViT2D": "ViT/Swin",
}


def _pretty_combo(combo: str) -> tuple[str, str]:
    arch, method = combo.split("+")
    return ARCH_LABELS.get(arch, arch), METHOD_LABELS.get(method, method)


def _nll(y_true, p_mean):
    p = np.clip(p_mean[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return float(-np.log(p).mean())


def _classwise(y_true, y_pred, p_mean):
    rows = []
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(config.CLASS_NAMES)))
    for i, cls in enumerate(config.CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        try:
            binary_true = (y_true == i).astype(int)
            auc_i = roc_auc_score(binary_true, p_mean[:, i])
        except ValueError:
            auc_i = np.nan
        rows.append({
            "class": cls,
            "precision": precision,
            "sensitivity_recall": recall,
            "specificity": specificity,
            "f1": f1,
            "auc": auc_i,
        })
    return pd.DataFrame(rows)


def _reliability_data(y_true, p_mean, n_bins=10):
    conf = p_mean.max(axis=1)
    pred = p_mean.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            rows.append({
                "bin_center": (lo + hi) / 2,
                "accuracy": correct[mask].mean(),
                "confidence": conf[mask].mean(),
                "count": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def _bootstrap_difference(x, y, n_boot=2000, seed=42):
    """Paired bootstrap p-value/effect for two per-sample metrics."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    diff = x - y
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    se = float(boot.std(ddof=1))
    z = observed / (se + 1e-12)
    p = float(2 * stats.norm.sf(abs(z)))
    effect = float(observed / (diff.std(ddof=1) + 1e-12))
    return observed, z, p, effect


def dataset_distribution(manifest, split_result):
    path_to_class = dict(zip(manifest["filepath"], manifest["class_label"]))
    rows = []
    for split in ["train", "val", "test"]:
        paths = split_result["splits"][split]
        counts = pd.Series([path_to_class[p] for p in paths]).value_counts()
        row = {"Dataset Split": split.title()}
        total = len(paths)
        for cls in config.CLASS_NAMES:
            row[cls] = int(counts.get(cls, 0))
        row["Total"] = total
        rows.append(row)
    total_row = {"Dataset Split": "Total"}
    for cls in config.CLASS_NAMES:
        total_row[cls] = sum(r[cls] for r in rows)
    total_row["Total"] = sum(r["Total"] for r in rows)
    rows.append(total_row)
    return pd.DataFrame(rows)


def baseline_table(manifest, split_result, model_classes, checkpoints, loaders):
    """Evaluate deterministic backbone checkpoints once for the baseline table."""
    import torch
    from . import evaluate
    rows = []
    results = {}
    for arch, cls in model_classes.items():
        model = cls(num_classes=len(config.CLASS_NAMES), dropout_p=config.DROPOUT_P)
        ckpt = config.CHECKPOINT_DIR / f"{arch.lower()}_best.pt"
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        res = evaluate.deterministic_predict_dataset(model, loaders["test"])
        results[arch] = res
        m = evaluate.compute_discriminative_metrics(res["y_true"], res["y_pred"], res["p_mean"])
        cm = confusion_matrix(res["y_true"], res["y_pred"], labels=np.arange(len(config.CLASS_NAMES)))
        balanced = np.mean([
            cm[i, i] / cm[i].sum() if cm[i].sum() else 0.0
            for i in range(len(config.CLASS_NAMES))
        ])
        rows.append({
            "Architecture": ARCH_LABELS.get(arch, arch),
            "Accuracy (%)": 100 * m["accuracy"],
            "Precision": m["precision_macro"],
            "Recall": m["recall_macro"],
            "Macro-F1": m["f1_macro"],
            "Balanced Acc.": balanced,
            "ROC-AUC": m["roc_auc_macro_ovr"],
        })
    df = pd.DataFrame(rows).set_index("Architecture")
    return df, results


def build_all_tables(study, manifest, split_result, model_classes, loaders):
    comparison = study["comparison_df"].copy()
    raw = study["all_results"]
    timings = study["timings"]

    # Baseline
    baseline_df, baseline_results = baseline_table(
        manifest, split_result, model_classes, study["checkpoints"], loaders
    )

    # 3x3 architecture-UQ table
    rows = []
    for combo, r in comparison.iterrows():
        arch, method = _pretty_combo(combo)
        rows.append({
            "Architecture": arch,
            "UQ Method": method,
            "Accuracy (%)": 100 * r["accuracy"],
            "Macro-F1": r["f1_macro"],
            "ROC-AUC": r["roc_auc_macro_ovr"],
            "ECE": r["expected_calibration_error"],
            "Brier": r["brier_score"],
            "Predictive Entropy": r["mean_predictive_entropy"],
        })
    arch_uq_df = pd.DataFrame(rows)

    # Best UQ for each architecture: maximize macro-F1, accuracy tie-break.
    best_by_arch = {}
    for arch in ["CNN2D", "DenseNet2D", "ViT2D"]:
        candidates = [c for c in comparison.index if c.startswith(arch + "+")]
        best_by_arch[arch] = comparison.loc[candidates].sort_values(
            ["f1_macro", "accuracy"], ascending=False
        ).index[0]

    classwise_frames = []
    for arch, combo in best_by_arch.items():
        r = raw[combo]
        cw = _classwise(r["y_true"], r["y_pred"], r["p_mean"])
        arch_label, method_label = _pretty_combo(combo)
        cw.insert(0, "Architecture + UQ", f"{arch_label} + {method_label}")
        classwise_frames.append(cw)
    classwise_df = pd.concat(classwise_frames, ignore_index=True)

    # Uncertainty quality + calibration
    uq_rows = []
    correct_rows = []
    for combo, r in comparison.iterrows():
        arch, method = _pretty_combo(combo)
        rr = raw[combo]
        y_true, y_pred, p = rr["y_true"], rr["y_pred"], rr["p_mean"]
        correct = y_pred == y_true
        uncertainty = rr["variance"]
        uq_rows.append({
            "Architecture": arch,
            "UQ Method": method,
            "ECE": r["expected_calibration_error"],
            "Brier Score": r["brier_score"],
            "Predictive Entropy": r["mean_predictive_entropy"],
            "Predictive Variance/Disagreement": r["mean_predictive_variance"],
            "NLL": _nll(y_true, p),
        })
        cu = float(uncertainty[correct].mean()) if correct.any() else np.nan
        iu = float(uncertainty[~correct].mean()) if (~correct).any() else np.nan
        diff = iu - cu
        if np.isfinite(cu) and np.isfinite(iu):
            t, pv = stats.ttest_ind(uncertainty[~correct], uncertainty[correct], equal_var=False)
        else:
            t, pv = np.nan, np.nan
        correct_rows.append({
            "Architecture": arch,
            "UQ Method": method,
            "Correct Prediction Uncertainty": cu,
            "Incorrect Prediction Uncertainty": iu,
            "Difference": diff,
            "p-value": pv,
        })
    uq_df = pd.DataFrame(uq_rows)
    correct_df = pd.DataFrame(correct_rows)

    # Computational efficiency
    comp_rows = []
    for combo in comparison.index:
        arch_key, method_key = combo.split("+")
        arch, method = _pretty_combo(combo)
        model = model_classes[arch_key](num_classes=len(config.CLASS_NAMES), dropout_p=config.DROPOUT_P)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        base_infer = timings[f"{arch_key}+MC_Dropout"]["inference_time_sec_per_sample"]
        this_infer = timings[combo]["inference_time_sec_per_sample"]
        overhead = ((this_infer / max(base_infer, 1e-12)) - 1) * 100
        comp_rows.append({
            "Architecture": arch,
            "UQ Method": method,
            "Parameters (M)": params_m,
            "Training Time": timings[combo]["train_time_sec"],
            "Inference Time/Subject": this_infer,
            "UQ Overhead": overhead,
        })
    comp_eff_df = pd.DataFrame(comp_rows)

    # Statistical Comparison: bootstrap mean ± SD and 95% CI on the held-out test set.
    statistical_rows = []
    rng = np.random.default_rng(config.RANDOM_SEED)
    for combo, r in raw.items():
        y_true, y_pred, p_mean = r["y_true"], r["y_pred"], r["p_mean"]
        n = len(y_true)
        boot_acc, boot_f1, boot_auc, boot_ece, boot_brier = [], [], [], [], []
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            yt, yp, pp = y_true[idx], y_pred[idx], p_mean[idx]
            boot_acc.append(accuracy_score(yt, yp))
            boot_f1.append(precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)[2])
            try:
                boot_auc.append(roc_auc_score(yt, pp, multi_class="ovr", average="macro"))
            except ValueError:
                boot_auc.append(np.nan)
            conf = pp.max(axis=1)
            corr = (yp == yt).astype(float)
            edges = np.linspace(0, 1, 11)
            ece = 0.0
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (conf > lo) & (conf <= hi)
                if mask.any(): ece += mask.mean() * abs(corr[mask].mean() - conf[mask].mean())
            boot_ece.append(ece)
            oh = np.eye(len(config.CLASS_NAMES))[yt]
            boot_brier.append(np.mean(np.sum((pp - oh) ** 2, axis=1)))
        def ms_ci(x):
            x = np.asarray(x, float); x = x[np.isfinite(x)]
            if x.size == 0:
                return float("nan"), float("nan"), float("nan"), float("nan")
            sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
            return float(x.mean()), sd, float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))
        aa, asd, alo, ahi = ms_ci(boot_acc)
        ff, fsd, _, _ = ms_ci(boot_f1)
        aucm, aucsd, _, _ = ms_ci(boot_auc)
        ee, esd, _, _ = ms_ci(boot_ece)
        bb, bsd, _, _ = ms_ci(boot_brier)
        arch, method = _pretty_combo(combo)
        statistical_rows.append({
            "Architecture": arch, "UQ Method": method,
            "Accuracy Mean±SD": f"{aa:.4f} ± {asd:.4f}",
            "F1 Mean±SD": f"{ff:.4f} ± {fsd:.4f}",
            "AUC Mean±SD": f"{aucm:.4f} ± {aucsd:.4f}",
            "ECE Mean±SD": f"{ee:.4f} ± {esd:.4f}",
            "Brier Mean±SD": f"{bb:.4f} ± {bsd:.4f}",
            "95% CI Accuracy": f"[{alo:.4f}, {ahi:.4f}]",
        })
    statistical_df = pd.DataFrame(statistical_rows)

    # Statistical comparison between architectures, per UQ method.
    arch_sig_rows = []
    for method in ["MC_Dropout", "BEDL", "Deep_Ensembles"]:
        combos = [f"CNN2D+{method}", f"DenseNet2D+{method}", f"ViT2D+{method}"]
        pairs = [(combos[0], combos[1]), (combos[0], combos[2]), (combos[1], combos[2])]
        pvals = []
        temp = []
        for a, b in pairs:
            xa = (raw[a]["y_pred"] == raw[a]["y_true"]).astype(float)
            xb = (raw[b]["y_pred"] == raw[b]["y_true"]).astype(float)
            t, p = stats.ttest_rel(xa, xb)
            d = (xa - xb).mean() / ((xa - xb).std(ddof=1) + 1e-12)
            pvals.append(p)
            temp.append((a, b, t, p, d))
        adjusted = _holm(pvals)
        for (a, b, t, p, d), ap in zip(temp, adjusted):
            arch_sig_rows.append({
                "UQ Method": METHOD_LABELS[method],
                "Comparison": f"{ARCH_LABELS[a.split('+')[0]]} vs {ARCH_LABELS[b.split('+')[0]]}",
                "Metric": "Accuracy",
                "Test Statistic": t,
                "p-value": p,
                "Adjusted p": ap,
                "Effect Size": d,
                "Significant?": bool(ap < 0.05),
            })
    arch_sig_df = pd.DataFrame(arch_sig_rows)

    # Statistical comparison between UQ methods within architecture on ECE.
    uq_sig_rows = []
    for arch_key in ["CNN2D", "DenseNet2D", "ViT2D"]:
        methods = ["MC_Dropout", "BEDL", "Deep_Ensembles"]
        pairs = [(methods[0], methods[1]), (methods[0], methods[2]), (methods[1], methods[2])]
        pvals, temp = [], []
        for ma, mb in pairs:
            ra, rb = raw[f"{arch_key}+{ma}"], raw[f"{arch_key}+{mb}"]
            # Per-sample calibration error: |confidence - correctness|.
            xa = np.abs(ra["p_mean"].max(axis=1) - (ra["y_pred"] == ra["y_true"]).astype(float))
            xb = np.abs(rb["p_mean"].max(axis=1) - (rb["y_pred"] == rb["y_true"]).astype(float))
            diff, z, p, d = _bootstrap_difference(xa, xb)
            pvals.append(p)
            temp.append((ma, mb, z, p, d))
        adjusted = _holm(pvals)
        for (ma, mb, z, p, d), ap in zip(temp, adjusted):
            uq_sig_rows.append({
                "Architecture": ARCH_LABELS[arch_key],
                "Comparison": f"{METHOD_LABELS[ma]} vs {METHOD_LABELS[mb]}",
                "Metric": "ECE",
                "p-value": p,
                "Adjusted p": ap,
                "Effect Size": d,
                "Significant?": bool(ap < 0.05),
            })
    uq_sig_df = pd.DataFrame(uq_sig_rows)

    # Two-way repeated-measures ANOVA for accuracy, calibration error and Brier.
    anova_df = _two_way_anova(raw)

    # Final ranking of all nine combinations.
    rank_df = _final_ranking(comparison, comp_eff_df)

    return {
        "dataset_distribution": dataset_distribution(manifest, split_result),
        "baseline": baseline_df,
        "baseline_results": baseline_results,
        "architecture_uq": arch_uq_df,
        "best_by_architecture": best_by_arch,
        "classwise": classwise_df,
        "uq_quality": uq_df,
        "correct_incorrect": correct_df,
        "computational_efficiency": comp_eff_df,
        "statistical_comparison": statistical_df,
        "architecture_significance": arch_sig_df,
        "uq_significance": uq_sig_df,
        "architecture_uq_anova": anova_df,
        "ranking": rank_df,
    }


def _holm(pvals):
    pvals = np.asarray(pvals, float)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    m = len(pvals)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def _two_way_anova(raw):
    """Repeated-measures two-way ANOVA using each test subject as the block."""
    from statsmodels.stats.anova import AnovaRM
    rows = []
    for combo, r in raw.items():
        arch, method = combo.split("+")
        correct = (r["y_pred"] == r["y_true"]).astype(float)
        conf = r["p_mean"].max(axis=1)
        cal_gap = np.abs(conf - correct)
        brier_per_sample = np.sum((r["p_mean"] - np.eye(len(config.CLASS_NAMES))[r["y_true"]]) ** 2, axis=1)
        for i in range(len(correct)):
            rows.append({
                "subject": i,
                "architecture": arch,
                "uq": method,
                "accuracy": correct[i],
                "ECE": cal_gap[i],
                "Brier": brier_per_sample[i],
            })
    df = pd.DataFrame(rows)
    out = []
    for metric in ["accuracy", "ECE", "Brier"]:
        a = AnovaRM(df, depvar=metric, subject="subject", within=["architecture", "uq"]).fit()
        table = a.anova_table
        for source, row in table.iterrows():
            stat = float(row["F Value"])
            p = float(row["Pr > F"])
            df_num = float(row["Num DF"])
            df_den = float(row["Den DF"])
            # Partial eta squared from F and dfs.
            eta = (stat * df_num) / (stat * df_num + df_den) if np.isfinite(stat) else np.nan
            if source == "architecture": interpretation = "Backbone effect"
            elif source == "uq": interpretation = "UQ effect"
            else: interpretation = "Interaction"
            out.append({
                "Source": source.replace(":", " × "),
                "Metric": metric,
                "Test Statistic": stat,
                "df": f"{df_num:.0f},{df_den:.0f}",
                "p-value": p,
                "Effect Size": eta,
                "Interpretation": interpretation,
            })
    return pd.DataFrame(out)


def _final_ranking(comparison, comp_eff):
    df = comparison.copy()
    cols = {
        "Accuracy": df["accuracy"],
        "F1 ↑": df["f1_macro"],
        "AUC ↑": df["roc_auc_macro_ovr"].fillna(df["roc_auc_macro_ovr"].mean()),
        "ECE ↓": df["expected_calibration_error"],
        "Brier ↓": df["brier_score"],
        "Inference Cost ↓": df["inference_time_sec_per_sample"],
    }
    score = pd.Series(0.0, index=df.index)
    weights = {"Accuracy": .30, "F1 ↑": .20, "AUC ↑": .15, "ECE ↓": .15, "Brier ↓": .10, "Inference Cost ↓": .10}
    for key, vals in cols.items():
        lo, hi = vals.min(), vals.max()
        norm = (vals - lo) / (hi - lo + 1e-12)
        if key in ["ECE ↓", "Brier ↓", "Inference Cost ↓"]:
            norm = 1 - norm
        score += weights[key] * norm
    df["final_score"] = score
    df = df.sort_values("final_score", ascending=False)
    rows = []
    for rank, (combo, r) in enumerate(df.iterrows(), 1):
        arch, method = _pretty_combo(combo)
        rows.append({
            "Rank": rank,
            "Architecture": arch,
            "UQ Method": method,
            "Accuracy": r["accuracy"],
            "F1 ↑": r["f1_macro"],
            "AUC ↑": r["roc_auc_macro_ovr"],
            "ECE ↓": r["expected_calibration_error"],
            "Brier ↓": r["brier_score"],
            "Inference Cost ↓": r["inference_time_sec_per_sample"],
            "Final Score": r["final_score"],
            "Final Recommendation": "Best overall" if rank == 1 else "",
        })
    return pd.DataFrame(rows)


def save_tables(tables, out_dir=None):
    out_dir = Path(out_dir or config.METRICS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, df in tables.items():
        if not isinstance(df, pd.DataFrame):
            continue
        df.to_csv(out_dir / f"{key}.csv", index=False)
    # Single workbook containing the complete requested result set.
    xlsx = out_dir / "TEAM4_complete_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for key, df in tables.items():
            if isinstance(df, pd.DataFrame):
                sheet = key[:31]
                df.to_excel(writer, sheet_name=sheet, index=False)
    return xlsx


def save_summary_json(tables, study, out_path=None):
    out_path = Path(out_path or config.METRICS_DIR / "TEAM4_complete_results.json")
    payload = {
        "recommended_combination": str(tables["ranking"].iloc[0]["Architecture"] + "+" + tables["ranking"].iloc[0]["UQ Method"]),
        "selection_rule": "Weighted multi-objective score: Accuracy 30%, Macro-F1 20%, ROC-AUC 15%, ECE 15%, Brier 10%, inference cost 10%.",
        "best_by_architecture": {k: str(v) for k, v in tables["best_by_architecture"].items()},
        "dataset_distribution": tables["dataset_distribution"].to_dict(orient="records"),
        "ranking": tables["ranking"].to_dict(orient="records"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=lambda x: None if pd.isna(x) else x)
    return out_path


def _safe(name):
    return name.replace("+", "_").replace(" ", "_")


def generate_figures(tables, study, out_dir=None):
    out_dir = Path(out_dir or config.PLOTS_DIR / "TEAM4")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = study["all_results"]
    comparison = study["comparison_df"]

    # 1. 3x3 Architecture-UQ performance heatmap.
    metric = comparison["f1_macro"].copy()
    matrix = pd.DataFrame(index=[ARCH_LABELS[x] for x in ["CNN2D", "DenseNet2D", "ViT2D"]], columns=[METHOD_LABELS[x] for x in ["MC_Dropout", "BEDL", "Deep_Ensembles"]], dtype=float)
    for combo, val in metric.items():
        a, m = combo.split("+")
        matrix.loc[ARCH_LABELS[a], METHOD_LABELS[m]] = val
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix.values, aspect="auto")
    ax.set_xticks(range(3), matrix.columns)
    ax.set_yticks(range(3), matrix.index)
    ax.set_title("3×3 Architecture–UQ Performance Heatmap (Macro-F1)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix.iloc[i,j]:.3f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Macro-F1")
    fig.tight_layout(); fig.savefig(out_dir / "architecture_uq_heatmap.png", dpi=180); plt.close(fig)

    # 2. Calibration/reliability diagrams for all nine combinations.
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), sharex=True, sharey=True)
    for ax, combo in zip(axes.flat, comparison.index):
        r = raw[combo]
        rel = _reliability_data(r["y_true"], r["p_mean"])
        ax.plot([0,1], [0,1], linestyle="--", linewidth=1)
        if len(rel):
            ax.plot(rel["confidence"], rel["accuracy"], marker="o")
        ax.set_title(" + ".join(_pretty_combo(combo)), fontsize=9)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.grid(alpha=.2)
    fig.supxlabel("Confidence"); fig.supylabel("Accuracy")
    fig.suptitle("Calibration / Reliability Diagrams")
    fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(out_dir / "reliability_diagrams_9.png", dpi=180); plt.close(fig)

    # 3. ECE + Brier comparison.
    uq = tables["uq_quality"]
    x = np.arange(len(uq))
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(x-.18, uq["ECE"], .36, label="ECE")
    ax.bar(x+.18, uq["Brier Score"], .36, label="Brier")
    ax.set_xticks(x, [f"{a}\n{m}" for a,m in zip(uq["Architecture"], uq["UQ Method"])], rotation=35, ha="right")
    ax.set_title("ECE + Brier Score Comparison (lower is better)"); ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "ece_brier_comparison.png", dpi=180); plt.close(fig)

    # 4. Correct vs incorrect uncertainty distribution.
    ci = tables["correct_incorrect"].copy()
    data_correct, data_wrong, labels = [], [], []
    for combo in comparison.index:
        r = raw[combo]
        correct = r["y_pred"] == r["y_true"]
        data_correct.append(r["variance"][correct])
        data_wrong.append(r["variance"][~correct])
        labels.append("\n".join(_pretty_combo(combo)))
    fig, ax = plt.subplots(figsize=(13,6))
    positions = np.arange(len(labels))
    ax.boxplot(data_correct, positions=positions-.18, widths=.3, patch_artist=False, showfliers=False)
    ax.boxplot(data_wrong, positions=positions+.18, widths=.3, patch_artist=False, showfliers=False)
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylabel("Predictive uncertainty"); ax.set_title("Correct vs Incorrect Prediction Uncertainty Distribution")
    ax.legend([plt.Line2D([0],[0], color="black"), plt.Line2D([0],[0], color="black")], ["Correct", "Incorrect"])
    fig.tight_layout(); fig.savefig(out_dir / "correct_vs_incorrect_uncertainty.png", dpi=180); plt.close(fig)

    # 5. CN vs MCI vs AD uncertainty distribution using best UQ per architecture.
    fig, ax = plt.subplots(figsize=(10,6))
    positions, vals, labs = [], [], []
    p = 1
    for arch, combo in tables["best_by_architecture"].items():
        r = raw[combo]
        for cls_idx, cls in enumerate(config.CLASS_NAMES):
            mask = r["y_true"] == cls_idx
            vals.append(r["variance"][mask]); positions.append(p); labs.append(f"{ARCH_LABELS[arch]}\n{cls}"); p += 1
    ax.boxplot(vals, positions=positions, widths=.65, showfliers=False)
    ax.set_xticks(positions, labs, rotation=35, ha="right")
    ax.set_ylabel("Predictive uncertainty"); ax.set_title("CN vs MCI vs AD Uncertainty Distribution")
    fig.tight_layout(); fig.savefig(out_dir / "class_uncertainty_distribution.png", dpi=180); plt.close(fig)

    # 6. Risk-coverage curve for best combination.
    best_combo = study["decision"]["recommended_combination"]
    if best_combo not in raw:
        best_combo = tables["ranking"].iloc[0]["Architecture"] + "+" + tables["ranking"].iloc[0]["UQ Method"]
        best_combo = {v:k for k,v in {"3D CNN":"CNN2D", "DenseNet-121":"DenseNet2D", "ViT/Swin":"ViT2D"}.items()}.get(best_combo, best_combo)
    r = raw[best_combo]
    order = np.argsort(r["variance"])
    correct = (r["y_pred"] == r["y_true"])[order].astype(float)
    cov = np.linspace(1/len(correct), 1, len(correct))
    risk = 1 - np.cumsum(correct) / np.arange(1, len(correct)+1)
    fig, ax = plt.subplots(figsize=(7,5)); ax.plot(cov, risk)
    ax.set_xlabel("Coverage"); ax.set_ylabel("Risk (error rate)"); ax.set_title("Risk–Coverage / Selective Prediction Curve"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out_dir / "risk_coverage_curve.png", dpi=180); plt.close(fig)

    # 7. Best pair confusion matrix + ROC curves.
    fig, ax = plt.subplots(figsize=(6,5))
    cm = confusion_matrix(r["y_true"], r["y_pred"], labels=np.arange(len(config.CLASS_NAMES)))
    im = ax.imshow(cm)
    ax.set_xticks(range(3), config.CLASS_NAMES); ax.set_yticks(range(3), config.CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Best Pair Confusion Matrix\n{best_combo}")
    for i in range(3):
        for j in range(3): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out_dir / "best_confusion_matrix.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,6))
    for i, cls in enumerate(config.CLASS_NAMES):
        y_bin = (r["y_true"] == i).astype(int)
        if len(np.unique(y_bin)) < 2: continue
        fpr, tpr, _ = roc_curve(y_bin, r["p_mean"][:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title(f"ROC Curves — {best_combo}"); ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "best_roc_curves.png", dpi=180); plt.close(fig)

    # 8. Architecture × UQ interaction plot.
    fig, ax = plt.subplots(figsize=(8,5))
    for method in ["MC Dropout", "Evidential DL", "Deep Ensemble"]:
        sub = tables["architecture_uq"][tables["architecture_uq"]["UQ Method"] == method]
        sub = sub.set_index("Architecture").reindex(["3D CNN","DenseNet-121","ViT/Swin"])
        ax.plot(sub.index, sub["Accuracy (%)"], marker="o", label=method)
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Architecture × UQ Interaction Plot"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out_dir / "architecture_uq_interaction.png", dpi=180); plt.close(fig)

    return out_dir