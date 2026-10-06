#!/usr/bin/env python
"""
Phase 4 / E2 -- Normalization ablation (128x128 vs the 96x96 Phase 3 baseline)
=============================================================================

    python run_phase4_e1.py --preflight     # checks only: no training, nothing written
    python run_phase4_e1.py                 # train CNN2D + DenseNet2D at 128x128
    python run_phase4_e1.py --report-only   # rebuild the comparison table only
    python run_phase4_e1.py --models cnn2d --resume

Design
------
* Reuses run_phase3_baseline.py helpers and src.train.train_model unchanged.
* Only the image size differs from Phase 3 (96x96 -> 128x128). Config is checked
  against the saved Phase 3 config_snapshot.json of each model.
* 96x96 is comparison-only: read from outputs/v2/baseline/*/val_metrics.json
  (validation metrics only; Phase 3 results are never modified).
* TEST SPLIT IS NEVER ACCESSED: test rows are dropped from the manifest right
  after the CSV assignment checks; no test image path is stat'ed, hashed,
  loaded, or evaluated, and no test metrics file is read.
* Checkpoints (*.pt) are gitignored; metadata/metrics/config are not.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

import run_phase3_baseline as p3
from src import augmentation, checkpoint_utils, config, dataset, evaluate, train

logger = logging.getLogger("phase4_e1")

RESOLUTION = (96, 96)
MODELS = ["cnn2d", "densenet2d"]
EXP_DIR = ROOT / "outputs" / "v2" / "experiments" / "normalization"
REGISTRY = ROOT / "outputs" / "v2" / "experiments" / "experiment_registry.jsonl"
COMPARISON_MD = ROOT / "outputs" / "v2" / "experiments" / "resolution" / "e2_normalization_comparison.md"
FINAL_CSV = "outputs/final_400/final_400.csv"
SPLIT_CSV = "outputs/splits/split_assignments.csv"

# Phase 3 results are read-only inputs here; prove they are unchanged after the run.
p3.SNAPSHOT_DIRS.append("outputs/v2/baseline")


# ---------------------------------------------------------------------------
# Config consistency with Phase 3
# ---------------------------------------------------------------------------
def check_against_phase3(base_cfg: dict) -> None:
    """Every Phase 3 setting except image_size must match the saved baselines."""
    for n in MODELS:
        d = p3.BASE / n
        for fn in ("config_snapshot.json", "val_metrics.json", "status.json"):
            if not (d / fn).exists():
                raise SystemExit(f"[ABORT] Phase 3 baseline result missing: {(d / fn).relative_to(ROOT)}")
        if json.loads((d / "status.json").read_text()).get("status") != "complete":
            raise SystemExit(f"[ABORT] Phase 3 baseline for {n} is not complete")
        snap = json.loads((d / "config_snapshot.json").read_text())
        if list(snap["image_size"]) != [96, 96]:
            raise SystemExit(f"[ABORT] Phase 3 {n} image_size is {snap['image_size']}, expected [96, 96]")
        for k in p3.EXPECTED_BASELINE:
            if k == "image_size":
                continue
            a, b = base_cfg[k], snap[k]
            if (list(a) if isinstance(a, tuple) else a) != (list(b) if isinstance(b, tuple) else b):
                raise SystemExit(f"[ABORT] config mismatch vs Phase 3 {n}: {k} = {a!r} (E1) vs {b!r} (baseline)")


# ---------------------------------------------------------------------------
# Locked train/val only (test rows are dropped before any file access)
# ---------------------------------------------------------------------------
def load_trainval():
    final_df = pd.read_csv(ROOT / FINAL_CSV)
    split_df = pd.read_csv(ROOT / SPLIT_CSV)
    if len(final_df) != 400 or len(split_df) != 400:
        raise SystemExit("[ABORT] locked CSVs do not contain 400 rows")
    meta = ["filepath", "class_name", "class_id", "subject_id", "session", "scan", "slice_idx", "sha256"]
    m = final_df[meta].merge(split_df[["filepath", "class_name", "class_id", "subject_id", "split"]],
                             on="filepath", how="inner", validate="one_to_one", suffixes=("", "_split"))
    if len(m) != 400:
        raise SystemExit("[ABORT] final_400 / split_assignments filepaths do not match 1:1")
    for a in ("class_name", "class_id", "subject_id"):
        if not (m[a] == m[a + "_split"]).all():
            raise SystemExit(f"[ABORT] final_400 and split disagree on {a}")
    for cid, cname in enumerate(config.CLASS_NAMES):
        if not ((m.class_id == cid) == (m.class_name == cname)).all():
            raise SystemExit("[ABORT] class mapping differs from config.CLASS_NAMES")
    # --- integrity checks on CSV assignment metadata only (no image file is touched) ---
    counts = m.split.value_counts().to_dict()
    if counts != p3.EXPECTED_SPLIT_COUNTS:
        raise SystemExit(f"[ABORT] split counts {counts} != {p3.EXPECTED_SPLIT_COUNTS}")
    if (m.groupby("subject_id").split.nunique() != 1).any():
        raise SystemExit("[ABORT] subject appears in more than one partition")
    mod = m[m.class_name == "Moderate Demented"].groupby("subject_id").split.agg(lambda x: set(x))
    if {k: next(iter(v)) for k, v in mod.items()} != p3.EXPECTED_MODERATE:
        raise SystemExit(f"[ABORT] Moderate assignment differs from locked V1: {dict(mod)}")

    m = m[m.split.isin(["train", "val"])].drop(columns=[c for c in m.columns if c.endswith("_split")]).copy()
    assert not (m.split == "test").any()   # from here on the test split does not exist in memory

    m["rel_filepath"] = m["filepath"]
    m["filepath"] = m["rel_filepath"].map(lambda x: str(p3.resolve_path(x)))
    m["class_idx"] = m["class_id"].astype(int)
    m["class_label"] = m["class_name"]
    missing = [p for p in m.filepath if not Path(p).exists()]
    if missing:
        raise SystemExit(f"[ABORT] {len(missing)} train/val image(s) not found under {config.DATA_ROOT}; "
                         f"first: {missing[0]}")
    bad = [r.filepath for r in m.itertuples() if p3.sha_file(Path(r.filepath)) != r.sha256]
    if bad:
        raise SystemExit(f"[ABORT] {len(bad)} train/val image(s) differ from final_400.csv sha256; first: {bad[0]}")

    # build_dataloaders requires a "test" key; an empty list yields an empty dataset (nothing is read).
    split_result = {"strategy": "locked-subject-level",
                    "splits": {"train": m.loc[m.split == "train", "filepath"].tolist(),
                               "val": m.loc[m.split == "val", "filepath"].tolist(),
                               "test": []}}
    return m, split_result


def build_loaders(manifest, split_result, normalization="zscore"):
    # Windows DataLoader workers are spawned as fresh processes. Keep E2 on the
    # main process so the controlled preprocessing override applies reliably.
    old_workers = config.NUM_WORKERS
    old_persistent = config.PERSISTENT_WORKERS
    config.NUM_WORKERS = 0
    config.PERSISTENT_WORKERS = False

    original_preprocess = dataset.preprocessing.preprocess_slice
    original_cached = dataset.preprocessing.get_cached_slice

    def e2_preprocess(filepath, target_size=None):
        target_size = target_size if target_size is not None else RESOLUTION
        arr = dataset.preprocessing.load_slice_as_array(filepath)
        arr = dataset.preprocessing.resize_array(arr, target_size)
        if normalization == "minmax":
            lo, hi = float(arr.min()), float(arr.max())
            if hi > lo:
                arr = (arr - lo) / (hi - lo)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
            return arr.astype(np.float32)
        return dataset.preprocessing.normalize_intensity(arr).astype(np.float32)

    def e2_cached(filepath, target_size=None):
        # Bypass the shared disk cache because cache contents are normalization-
        # specific and the existing cache key does not encode normalization.
        return e2_preprocess(filepath, RESOLUTION)

    dataset.preprocessing.preprocess_slice = e2_preprocess
    dataset.preprocessing.get_cached_slice = e2_cached
    try:
        loaders = dataset.build_dataloaders(
            manifest,
            split_result,
            train_transform=augmentation.get_train_transform(),
            eval_transform=augmentation.get_eval_transform(),
        )
        assert len(loaders["test"].dataset) == 0, "test loader must be empty in E2"
        return loaders
    finally:
        dataset.preprocessing.preprocess_slice = original_preprocess
        dataset.preprocessing.get_cached_slice = original_cached
        config.NUM_WORKERS = old_workers
        config.PERSISTENT_WORKERS = old_persistent


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def append_registry(rec: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, "a", encoding="utf-8") as f:   # append-only; earlier records are never rewritten
        f.write(json.dumps(rec, default=p3._json_default) + "\n")


def present_macro(per_class: dict, key: str) -> float:
    vals = [v[key] for v in per_class.values() if v["support"] > 0]
    return float(sum(vals) / len(vals))


# ---------------------------------------------------------------------------
# One model
# ---------------------------------------------------------------------------
def run_one(name: str, manifest, split_result, base_cfg: dict, git: dict, env: dict, hashes: dict, normalization: str) -> dict:
    mdir = EXP_DIR / normalization / name
    mdir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in mdir.iterdir() if p.name != ".gitkeep"]
    if existing:
        raise SystemExit(f"[ABORT] {mdir} is not empty. E1 never overwrites results; "
                         "archive the directory or use --resume to skip completed models.")
    fh = logging.FileHandler(mdir / "train.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    exp_id = f"V2-NORM-{normalization.upper()}-{name.upper()}"
    parent = f"V2-B0-{name.upper()}"
    t0 = time.time()
    old_metrics_dir = config.METRICS_DIR
    status = {"model": name, "experiment_id": exp_id, "started": p3.now(), "status": "running"}
    try:
        seed = base_cfg["random_seed"]
        p3.set_seed(seed)
        loaders = build_loaders(manifest, split_result, normalization=normalization)
        class_w = train.class_weights_from_manifest(manifest, split_result, p3.NUM_CLASSES)
        model = p3.build_model(name)
        n_params = int(sum(p.numel() for p in model.parameters()))

        cfg = {**base_cfg, "experiment_id": exp_id, "parent_experiment": parent, "experiment": "E2-normalization", "normalization_variant": normalization,
               "model": p3.MODEL_LABEL[name], "n_parameters": n_params,
               "class_weights": [float(w) for w in class_w.cpu()], "class_order": list(config.CLASS_NAMES),
               "dataset": FINAL_CSV, "split": SPLIT_CSV, "dataset_sha256": hashes["dataset"],
               "split_file_sha256": hashes["split"],
               "split_sizes_used": {"train": len(split_result["splits"]["train"]),
                                    "val": len(split_result["splits"]["val"])},
               "val_split_hash": checkpoint_utils.hash_split(manifest, split_result, "val"),
               "test_split_accessed": False, "changed_vs_phase3": {"image_size": [[96, 96], list(RESOLUTION)]},
               "git": git, "environment": env, "started": status["started"]}
        p3.write_json(cfg, mdir / "config_snapshot.json")

        # ---------------- training (VALIDATION only) ----------------
        ckpt = mdir / f"{name}_best.pt"          # *.pt is gitignored
        config.METRICS_DIR = mdir                # training_history.json goes into this run's directory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        hist = train.train_model(
            model, loaders["train"], loaders["val"], epochs=config.EPOCHS, lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY, checkpoint_path=ckpt, device=config.DEVICE,
            max_attempts=config.MAX_ATTEMPTS, base_seed=seed, class_weights=class_w,
            patience=config.EARLY_STOP_PATIENCE, scheduler_kind=config.LR_SCHEDULER)
        config.METRICS_DIR = old_metrics_dir
        train_seconds = time.time() - t0
        peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None

        best_epoch = p3.best_epoch_from_history(hist)
        attempts_used = int(hist["seed"] - seed + 1)
        meta = checkpoint_utils.build_metadata(name, config, split_result, manifest, hist, pretrained=False)
        meta.pop("test_split_hash", None)        # test split deliberately not touched in E1
        meta["split_sizes"] = cfg["split_sizes_used"]
        meta.update({"experiment_id": exp_id, "parent_experiment": parent, "best_epoch": best_epoch,
                     "attempts_used": attempts_used, "test_split_accessed": False,
                     "checkpoint_sha256_at_save": p3.sha_file(ckpt)})
        checkpoint_utils.save_metadata(ckpt, meta)

        frozen = {"frozen_at": p3.now(), "checkpoint": ckpt.name, "checkpoint_sha256": p3.sha_file(ckpt),
                  "selection_metric": "validation macro-F1 (tie-break: val accuracy, then lower val loss)",
                  "best_epoch": best_epoch, "attempts_used": attempts_used, "seed_of_selected_attempt": hist["seed"],
                  "best_val_macro_f1": hist["best_val_macro_f1"], "best_val_acc": hist["best_val_acc"],
                  "best_val_loss": hist["best_val_loss"], "degenerate": hist["degenerate"]}
        p3.write_json(frozen, mdir / "frozen_checkpoint.json")

        # ---------------- validation re-evaluation of the frozen checkpoint ----------------
        model = p3.build_model(name)
        model.load_state_dict(p3.load_state(ckpt, config.DEVICE))
        vpred = evaluate.deterministic_predict_dataset(model, loaders["val"], device=config.DEVICE)
        v_f1 = float(f1_score(vpred["y_true"], vpred["y_pred"], average="macro", zero_division=0))
        v_acc = float(accuracy_score(vpred["y_true"], vpred["y_pred"]))
        if abs(v_f1 - hist["best_val_macro_f1"]) > 1e-6 or abs(v_acc - hist["best_val_acc"]) > 1e-6:
            raise RuntimeError("Reloaded checkpoint does not reproduce recorded validation metrics "
                               f"(f1 {v_f1:.6f} vs {hist['best_val_macro_f1']:.6f}; "
                               f"acc {v_acc:.6f} vs {hist['best_val_acc']:.6f}).")
        val_m = p3.full_metrics(vpred["y_true"], vpred["y_pred"], vpred["p_mean"], with_roc=False)
        val_m.update({"role": "VALIDATION (model selection)", "selection_macro_f1_as_computed_by_train_py": v_f1,
                      "val_loss_at_best_epoch": hist["best_val_loss"], "best_epoch": best_epoch,
                      "checkpoint_reproduces_training_metrics": True,
                      "note": "Moderate Demented is absent from validation; macro_f1_present_classes is the 3-class value."})
        p3.write_json(val_m, mdir / "val_metrics.json")
        p3.predict_table(loaders["val"], vpred).to_csv(mdir / "val_predictions.csv", index=False)
        evaluate.plot_confusion_matrix(vpred["y_true"], vpred["y_pred"], mdir / "val_confusion_matrix.png")

        summary = {**status, "status": "complete", "finished": p3.now(), "train_seconds": round(train_seconds, 1),
                   "peak_gpu_mem_mb": None if peak_mb is None else round(peak_mb, 1),
                   "best_epoch": best_epoch, "epochs_run_last_attempt": hist["epochs_run"],
                   "attempts_used": attempts_used, "degenerate": hist["degenerate"],
                   "checkpoint_sha256": frozen["checkpoint_sha256"], "n_parameters": n_params}
        p3.write_json(summary, mdir / "status.json")
        append_registry({
            "experiment_id": exp_id, "parent_experiment": parent, "timestamp": summary["finished"], "seed": seed,
            "dataset_hash": hashes["dataset"], "split_hash": hashes["split"], "val_split_hash": cfg["val_split_hash"],
            "architecture": p3.MODEL_LABEL[name], "resolution": list(RESOLUTION),
            "normalization": normalization, "augmentation": base_cfg["train_augmentation"],
            "class_balance": base_cfg["class_balance_strategy"], "pretrained": False, "optimizer": "AdamW",
            "learning_rate": config.LEARNING_RATE, "batch_size": config.BATCH_SIZE, "epochs": config.EPOCHS,
            "best_epoch": best_epoch, "best_val_accuracy": hist["best_val_acc"],
            "best_val_macro_f1": hist["best_val_macro_f1"],
            "val_macro_f1_present_classes": val_m["macro_f1_present_classes"],
            "val_macro_precision_present_classes": present_macro(val_m["per_class"], "precision"),
            "val_macro_recall_present_classes": present_macro(val_m["per_class"], "recall"),
            "val_loss": hist["best_val_loss"], "per_class_metrics": val_m["per_class"],
            "ece": val_m["ece_10bins"], "brier": val_m["brier"], "train_seconds": summary["train_seconds"],
            "peak_gpu_mem_mb": summary["peak_gpu_mem_mb"], "degenerate": hist["degenerate"],
            "checkpoint": str((mdir / ckpt.name).relative_to(ROOT).as_posix()) + " (gitignored)",
            "test_split_accessed": False, "status": "complete"})
        return summary
    except BaseException as e:  # noqa: BLE001
        config.METRICS_DIR = old_metrics_dir
        (mdir / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        failed = {**status, "status": "failed", "finished": p3.now(), "error": f"{type(e).__name__}: {e}"}
        p3.write_json(failed, mdir / "status.json")
        append_registry({"experiment_id": exp_id, "parent_experiment": parent, "timestamp": failed["finished"],
                         "architecture": p3.MODEL_LABEL[name], "resolution": list(RESOLUTION),
                         "status": "failed", "error": failed["error"], "test_split_accessed": False})
        logger.error("%s FAILED:\n%s", name, traceback.format_exc())
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        return {"model": name, "status": "failed", "error": failed["error"]}
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()


# ---------------------------------------------------------------------------
# Comparison (validation only; Phase 3 test_metrics.json is never read)
# ---------------------------------------------------------------------------
def build_comparison() -> None:
    def load(d: Path):
        if not (d / "val_metrics.json").exists() or not (d / "status.json").exists():
            return None
        v = json.loads((d / "val_metrics.json").read_text())
        s = json.loads((d / "status.json").read_text())
        return (v, s) if s.get("status") == "complete" else None

    def f(x, nd=4):
        return "n/a" if x is None else f"{x:.{nd}f}"

    lines = [
        "# E2 — Normalization ablation (validation only)\n",
        "96×96, locked split, seed 42. Baseline = per-image z-score; "
        "alternative = per-image min-max [0,1]. Test split was not accessed.\n",
        "| Model | Normalization | Val acc | Sel. macro-F1 | Macro-F1 (3 present) | Val loss | ECE | Brier | Best ep | Train s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for n in MODELS:
        for norm in ("zscore", "minmax"):
            r = load(EXP_DIR / norm / n)
            if not r:
                lines.append(f"| {p3.MODEL_LABEL[n]} | {norm} | not available |||||||")
                continue
            v, s = r
            lines.append(
                f"| {p3.MODEL_LABEL[n]} | {norm} | {f(v['accuracy'])} | "
                f"{f(v['selection_macro_f1_as_computed_by_train_py'])} | "
                f"{f(v['macro_f1_present_classes'])} | {f(v['val_loss_at_best_epoch'])} | "
                f"{f(v['ece_10bins'])} | {f(v['brier'])} | {s.get('best_epoch')} | "
                f"{s.get('train_seconds')} |"
            )
    lines += [
        "",
        "Caveat: 44 validation images, one seed, and no repeated runs. "
        "Differences are exploratory, not evidence of superiority."
    ]
    COMPARISON_MD.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def preflight(manifest, split_result, snap_before: dict) -> None:
    print(f"== E2 normalization preflight | image size {RESOLUTION} | "
          f"train/val rows {len(manifest)} | test rows in memory: 0 ==")
    for norm in ("zscore", "minmax"):
        ld = build_loaders(manifest, split_result, normalization=norm)
        for nm in ("train", "val"):
            x, _ = next(iter(ld[nm]))
            assert x.ndim == 4 and x.shape[1] == 1 and tuple(x.shape[2:]) == RESOLUTION, x.shape
            assert torch.isfinite(x).all()
            print(f"  {norm:7s} {nm} batch {tuple(x.shape)} OK (test loader empty: {len(ld['test'].dataset) == 0})")
    for n in MODELS:
        m = p3.build_model(n).eval()
        with torch.no_grad():
            out = m(torch.randn(2, 1, *RESOLUTION))
        assert tuple(out.shape) == (2, p3.NUM_CLASSES)
    print("  CNN2D/DenseNet2D forward at 96x96: OK")
    for norm in ("zscore", "minmax"):
        for n in MODELS:
            d = EXP_DIR / norm / n
            if d.exists() and any(p.name != ".gitkeep" for p in d.iterdir()):
                raise SystemExit(f"[ABORT] output dir not empty: {d.relative_to(ROOT)}")
    rel = (EXP_DIR / "zscore" / "cnn2d" / "cnn2d_best.pt").relative_to(ROOT).as_posix()
    if subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT).returncode != 0:
        raise SystemExit("[ABORT] checkpoint path is NOT gitignored")
    assert not any(p3.diff_snapshots(snap_before, p3.snapshot_tree()).values())
    print("== E2 PREFLIGHT PASSED (nothing written, test split untouched) ==")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 / E2 normalization ablation")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    ap.add_argument("--normalization", choices=["zscore", "minmax", "both"], default="both")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    base_cfg = p3.apply_and_check_baseline_config()
    if list(config.IMAGE_SIZE) != [96, 96]:
        config.IMAGE_SIZE = (96, 96)
    p3.verify_locked_hashes()
    snap_before = p3.snapshot_tree()
    manifest, split_result = load_trainval()
    config.IMAGE_SIZE = (96, 96)

    if args.report_only:
        build_comparison()
        print(f"Comparison written to {COMPARISON_MD}")
        return

    if args.preflight:
        preflight(manifest, split_result, snap_before)
        return

    hashes = {"dataset": p3.sha_file(ROOT / FINAL_CSV), "split": p3.sha_file(ROOT / SPLIT_CSV)}
    git, env = p3.git_info(), p3.env_info()
    failures = []

    norms = ("zscore", "minmax") if args.normalization == "both" else (args.normalization,)
    for norm in norms:
        for n in args.models:
            sj = EXP_DIR / norm / n / "status.json"
            if args.resume and sj.exists() and json.loads(sj.read_text()).get("status") == "complete":
                print(f"== {norm}/{p3.MODEL_LABEL[n]}: already complete, skipping ==")
                continue
            print(f"\n==================== {p3.MODEL_LABEL[n]} @ 96x96 | {norm} ====================")
            r = run_one(n, manifest, split_result, base_cfg, git, env, hashes, norm)
            if r.get("status") != "complete":
                failures.append(f"{norm}/{p3.MODEL_LABEL[n]}: {r.get('error')}")

    build_comparison()
    p3.verify_locked_hashes()
    diff = p3.diff_snapshots(snap_before, p3.snapshot_tree())
    print(f"\nComparison: {COMPARISON_MD}\nRegistry: {REGISTRY}")
    if any(diff.values()):
        raise SystemExit(f"[ALERT] V1 / smoke-test / Phase 3 files changed: {diff}")
    print("V1 and Phase 3 artifacts unchanged.")
    if failures:
        raise SystemExit("[DONE WITH FAILURES] " + "; ".join(failures))
    print("E2 complete. STOP here — do not start E3.")


if __name__ == "__main__":
    main()