#!/usr/bin/env python
"""
Phase 3 -- V2 BASELINE runner (CNN2D, DenseNet2D, ViT2D)
=========================================================

Run from the repository root:

    python run_phase3_baseline.py --preflight      # no training, no test data touched
    python run_phase3_baseline.py                  # full Phase 3 baseline
    python run_phase3_baseline.py --report-only    # rebuild reports from saved results

What it does
------------
1. Verifies the locked V1 files against the hashes recorded in the smoke test
   (refuses to run on any mismatch) and snapshots every file under
   outputs/final_400, outputs/splits and outputs/v2/smoke_tests.
2. Loads the locked 400-image dataset + locked subject-level split
   (273 / 44 / 83). Nothing is regenerated.
3. For each model: trains with the repo's own src.train.train_model (AdamW,
   class-weighted CE, validation macro-F1 checkpointing, early stopping),
   freezes the checkpoint, re-evaluates it on VALIDATION to confirm the
   reloaded weights reproduce the selection metrics, and only then evaluates
   the frozen checkpoint ONCE on the test split.
4. Re-verifies all V1 hashes, then writes baseline_report.md / .json.

The test split is never used for selection, early stopping, tuning or
thresholds. No Phase 4 / UQ / subject-level aggregation is performed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from src import config, augmentation, checkpoint_utils, dataset, evaluate, train

logger = logging.getLogger("phase3")

# ---------------------------------------------------------------------------
# Locked values
# ---------------------------------------------------------------------------
LOCKED_SHA256 = {  # from outputs/v2/smoke_tests/smoke_test_report.json
    "outputs/final_400/final_400.csv":
        "cd62ce417cc29a7f56682366e2fbf83f337c3ec7dd5efe906a4d4c478e2a0f59",
    "outputs/splits/split_assignments.csv":
        "70246e6e39f4694b52c7a80f4871f1f48d831d8d04a92007ca8d52b06a4981b9",
}
# No locked hash was recorded for this file; it is only checked for
# "unchanged during this run".
UNHASHED_LOCKED = ["outputs/splits/moderate_subject_cv.csv"]
SNAPSHOT_DIRS = ["outputs/final_400", "outputs/splits", "outputs/v2/smoke_tests"]

EXPECTED_SPLIT_COUNTS = {"train": 273, "val": 44, "test": 83}
EXPECTED_MODERATE = {"OAS1_0351": "train", "OAS1_0308": "test"}

# Documented V2 baseline (demo profile in src/config.py + README/plan).
EXPECTED_BASELINE = {
    "mode": "demo", "image_size": (96, 96), "epochs": 25, "max_attempts": 2,
    "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4,
    "dropout_p": 0.35, "early_stop_patience": 6, "lr_scheduler": "plateau",
    "class_balance_strategy": "loss", "random_seed": 42,
}

MODEL_NAMES = ["cnn2d", "densenet2d", "vit2d"]
BASE = ROOT / "outputs" / "v2" / "baseline"
NUM_CLASSES = len(config.CLASS_NAMES)
LABELS = list(range(NUM_CLASSES))


# ---------------------------------------------------------------------------
# Hashing / V1 integrity
# ---------------------------------------------------------------------------
def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_variants(path: Path) -> dict:
    """Raw hash plus hash of the CRLF-normalised bytes.

    The smoke-test hashes were recorded on a Windows checkout (CRLF); the
    Git blobs are LF. The content is identical but the raw bytes differ, so
    a locked value may legitimately match either representation.
    """
    raw = Path(path).read_bytes()
    crlf = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return {"raw": _sha_bytes(raw), "crlf_normalised": _sha_bytes(crlf)}


def verify_locked_hashes() -> dict:
    out = {}
    for rel, locked in LOCKED_SHA256.items():
        p = ROOT / rel
        if not p.exists():
            raise SystemExit(f"[ABORT] locked V1 file missing: {rel}")
        v = hash_variants(p)
        if v["raw"] == locked:
            via = "raw"
        elif v["crlf_normalised"] == locked:
            via = "crlf_normalised"
        else:
            raise SystemExit(
                f"[ABORT] V1 hash mismatch for {rel}\n"
                f"  locked          : {locked}\n  raw             : {v['raw']}\n"
                f"  crlf_normalised : {v['crlf_normalised']}\n"
                "Refusing to run: the locked V1 artifact has changed content.")
        out[rel] = {"locked": locked, "matched_via": via, **v}
    return out


def snapshot_tree() -> dict:
    snap = {}
    for d in SNAPSHOT_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file():
                snap[p.relative_to(ROOT).as_posix()] = sha_file(p)
    for rel in UNHASHED_LOCKED:
        p = ROOT / rel
        if p.exists():
            snap[rel] = sha_file(p)
    return snap


def diff_snapshots(before: dict, after: dict) -> dict:
    return {
        "modified": sorted(k for k in before if k in after and before[k] != after[k]),
        "removed": sorted(k for k in before if k not in after),
        "added": sorted(k for k in after if k not in before),
    }


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not serialisable: {type(o)}")


def git_info() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"unavailable ({e})"
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "remote": run("config", "--get", "remote.origin.url"),
            "status_porcelain": run("status", "--porcelain").splitlines()}


def env_info() -> dict:
    import sklearn
    import torchvision
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "device": str(config.DEVICE),
            "gpu": torch.cuda.get_device_name(0) if config.DEVICE.type == "cuda" else None,
            "cpu_threads": config.CPU_THREADS, "num_workers": config.NUM_WORKERS}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_state(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # torch < 1.13
        return torch.load(path, map_location=device)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def apply_and_check_baseline_config() -> dict:
    config.configure("demo")
    actual = {
        "mode": config.MODE, "image_size": tuple(config.IMAGE_SIZE), "epochs": config.EPOCHS,
        "max_attempts": config.MAX_ATTEMPTS, "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE, "weight_decay": config.WEIGHT_DECAY,
        "dropout_p": config.DROPOUT_P, "early_stop_patience": config.EARLY_STOP_PATIENCE,
        "lr_scheduler": config.LR_SCHEDULER, "class_balance_strategy": config.CLASS_BALANCE_STRATEGY,
        "random_seed": config.RANDOM_SEED,
    }
    bad = {k: (actual[k], v) for k, v in EXPECTED_BASELINE.items() if actual[k] != v}
    if bad:
        raise SystemExit("[ABORT] config differs from the documented baseline "
                         f"(actual, expected): {bad}\nCheck env overrides "
                         "(CLASS_BALANCE_STRATEGY, LR_SCHEDULER) and src/config.py.")
    actual.update({
        "optimizer": "AdamW", "loss": "CrossEntropyLoss(weight=inverse-frequency class weights)",
        "scheduler_params": "ReduceLROnPlateau(mode=min, factor=0.5, patience=3) on val_loss",
        "checkpoint_selection": "val macro-F1 > val accuracy > lower val loss (src/train.py)",
        "preprocessing": "grayscale -> bilinear resize -> per-image zero-mean/unit-variance",
        "train_augmentation": "RandomHorizontalFlip(p=0.5) + RandomRotation(+/-10 deg, bilinear, fill=0)",
        "eval_augmentation": "none (deterministic)",
        "preprocess_cache": bool(config.USE_PREPROCESS_CACHE),
        "vit": {"patch_size": config.VIT_PATCH_SIZE, "embed_dim": config.VIT_EMBED_DIM,
                "depth": config.VIT_DEPTH, "num_heads": config.VIT_NUM_HEADS,
                "mlp_dim": config.VIT_MLP_DIM},
        "pretrained": False,
    })
    return actual


# ---------------------------------------------------------------------------
# Locked dataset
# ---------------------------------------------------------------------------
def resolve_path(rel: str) -> Path:
    """final_400.csv stores Windows-style paths relative to data/ (e.g.
    'oasis_subset\\non demented\\x.jpg'). Normalise separators so this works
    on any OS."""
    parts = PureWindowsPath(str(rel)).parts
    if PureWindowsPath(str(rel)).is_absolute():
        return Path(str(rel).replace("\\", "/"))
    if parts[0] != "oasis_subset":
        raise ValueError(f"unexpected path root in locked CSV: {rel}")
    return config.DATA_ROOT.joinpath(*parts[1:])


def load_locked(verify_images: bool = True):
    final_df = pd.read_csv(ROOT / "outputs/final_400/final_400.csv")
    split_df = pd.read_csv(ROOT / "outputs/splits/split_assignments.csv")
    if len(final_df) != 400 or len(split_df) != 400:
        raise SystemExit("[ABORT] locked CSVs do not contain 400 rows")

    meta = ["filepath", "class_name", "class_id", "subject_id", "session", "scan", "slice_idx"]
    m = final_df[meta + ["sha256"]].merge(
        split_df[["filepath", "class_name", "class_id", "subject_id", "split"]],
        on="filepath", how="inner", validate="one_to_one", suffixes=("", "_split"))
    if len(m) != 400:
        raise SystemExit("[ABORT] final_400 / split_assignments filepaths do not match 1:1")
    for a, b in [("class_name", "class_name_split"), ("class_id", "class_id_split"),
                 ("subject_id", "subject_id_split")]:
        if not (m[a] == m[b]).all():
            raise SystemExit(f"[ABORT] final_400 and split disagree on {a}")
    m = m.drop(columns=["class_name_split", "class_id_split", "subject_id_split"])

    for cid, cname in enumerate(config.CLASS_NAMES):
        if not ((m.class_id == cid) == (m.class_name == cname)).all():
            raise SystemExit("[ABORT] class_id <-> class_name mapping differs from config.CLASS_NAMES")
    if not (m.groupby("class_id").size() == 100).all():
        raise SystemExit("[ABORT] not 100 images per class")

    counts = m.split.value_counts().to_dict()
    if counts != EXPECTED_SPLIT_COUNTS:
        raise SystemExit(f"[ABORT] split counts {counts} != {EXPECTED_SPLIT_COUNTS}")
    if (m.groupby("subject_id").split.nunique() != 1).any():
        raise SystemExit("[ABORT] subject appears in more than one partition")
    mod = m[m.class_name == "Moderate Demented"].groupby("subject_id").split.agg(lambda x: set(x))
    if {k: next(iter(v)) for k, v in mod.items()} != EXPECTED_MODERATE:
        raise SystemExit(f"[ABORT] Moderate assignment differs from locked V1: {dict(mod)}")

    m["rel_filepath"] = m["filepath"]
    m["filepath"] = m["rel_filepath"].map(lambda x: str(resolve_path(x)))
    m["class_idx"] = m["class_id"].astype(int)
    m["class_label"] = m["class_name"]

    missing = [p for p in m.filepath if not Path(p).exists()]
    if missing:
        raise SystemExit(f"[ABORT] {len(missing)} locked image(s) not found under "
                         f"{config.DATA_ROOT}; first: {missing[0]}")
    if verify_images:
        bad = [r.filepath for r in m.itertuples() if sha_file(Path(r.filepath)) != r.sha256]
        if bad:
            raise SystemExit(f"[ABORT] {len(bad)} image(s) differ from the sha256 recorded in "
                             f"final_400.csv (not the locked images); first: {bad[0]}")

    split_result = {"strategy": "locked-subject-level",
                    "splits": {s: m.loc[m.split == s, "filepath"].tolist() for s in ("train", "val", "test")}}
    return m, split_result


def crosscheck_data_acquisition(manifest: pd.DataFrame) -> dict:
    """Compare against the repo's own loader where it works (Windows paths)."""
    try:
        from src import data_acquisition
        tr, va, te = data_acquisition.load_locked_experiment()
    except Exception as e:  # noqa: BLE001
        return {"status": "skipped", "reason": f"{type(e).__name__}: {e}"}
    ok = all(set(df.filepath) == set(manifest.loc[manifest.split == s, "filepath"])
             for s, df in (("train", tr), ("val", va), ("test", te)))
    if not ok:
        raise SystemExit("[ABORT] data_acquisition.load_locked_experiment() disagrees with the runner's loader")
    return {"status": "identical_to_data_acquisition"}


def describe_dataset(manifest: pd.DataFrame) -> dict:
    d = {}
    for s in ("train", "val", "test"):
        g = manifest[manifest.split == s]
        d[s] = {"images": int(len(g)), "subjects": int(g.subject_id.nunique()),
                "images_per_class": {c: int((g.class_name == c).sum()) for c in config.CLASS_NAMES},
                "subjects_per_class": {c: int(g[g.class_name == c].subject_id.nunique())
                                       for c in config.CLASS_NAMES}}
    return d


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_model(name: str):
    from src.models.cnn2d import CNN2D
    from src.models.densenet2d import DenseNet2D
    from src.models.vit2d import ViT2D
    p = config.DROPOUT_P
    if name == "cnn2d":
        return CNN2D(num_classes=NUM_CLASSES, dropout_p=p)
    if name == "densenet2d":
        return DenseNet2D(num_classes=NUM_CLASSES, dropout_p=p, in_channels=1)  # compact, from scratch
    if name == "vit2d":
        return ViT2D(num_classes=NUM_CLASSES, dropout_p=p)
    raise ValueError(name)


MODEL_LABEL = {"cnn2d": "CNN2D", "densenet2d": "DenseNet2D", "vit2d": "ViT2D"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def best_epoch_from_history(h: dict) -> int:
    """Replicates the improvement rule in src/train.py::_train_once
    (history does not store best_epoch)."""
    bf, ba, bl, best = -1.0, -1.0, float("inf"), 0
    for i, (f, a, l) in enumerate(zip(h["val_macro_f1"], h["val_acc"], h["val_loss"]), 1):
        if f > bf or (f == bf and a > ba) or (f == bf and a == ba and l < bl):
            bf, ba, bl, best = f, a, l, i
    return best


def full_metrics(y_true, y_pred, p, with_roc: bool) -> dict:
    p64 = np.asarray(p, dtype=np.float64)
    p64 = p64 / p64.sum(axis=1, keepdims=True)
    pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    present = [i for i in LABELS if sup[i] > 0]
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision_all4": float(pr.mean()), "macro_recall_all4": float(rc.mean()),
        "macro_f1_all4": float(f1.mean()),
        "macro_f1_present_classes": float(f1[present].mean()),
        "classes_present_in_true": [config.CLASS_NAMES[i] for i in present],
        "per_class": {config.CLASS_NAMES[i]: {"precision": float(pr[i]), "recall": float(rc[i]),
                                              "f1": float(f1[i]), "support": int(sup[i])} for i in LABELS},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "ece_10bins": evaluate.expected_calibration_error(y_true, p64),
        "brier": evaluate.multiclass_brier_score(y_true, p64),
        "roc_auc_macro_ovr": None, "roc_auc_note": None,
    }
    if with_roc:
        if len(present) == NUM_CLASSES:
            try:
                out["roc_auc_macro_ovr"] = float(roc_auc_score(y_true, p64, multi_class="ovr",
                                                               average="macro", labels=LABELS))
                out["roc_auc_note"] = ("computed, but the Moderate class comes from a single test subject; "
                                       "treat as descriptive only")
            except ValueError as e:
                out["roc_auc_note"] = f"not computable: {e}"
        else:
            out["roc_auc_note"] = "not valid: class(es) absent from y_true"
    else:
        out["roc_auc_note"] = "not reported on validation (Moderate Demented absent from validation split)"
    return out


def predict_table(loader, pred: dict) -> pd.DataFrame:
    m = loader.dataset.manifest
    df = pd.DataFrame({"filepath": m["filepath"].values, "subject_id": m["subject_id"].values,
                       "class_name": m["class_name"].values, "y_true": pred["y_true"],
                       "y_pred": pred["y_pred"]})
    for i, c in enumerate(config.CLASS_NAMES):
        df[f"p_{i}_{c.replace(' ', '_')}"] = pred["p_mean"][:, i]
    if not (df["y_true"].values == m["class_idx"].values).all():
        raise RuntimeError("prediction order does not match loader dataset order")
    return df


# ---------------------------------------------------------------------------
# One model
# ---------------------------------------------------------------------------
def run_one(name: str, manifest, split_result, base_cfg: dict, git: dict, env: dict) -> dict:
    mdir = BASE / name
    mdir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in mdir.iterdir() if p.name != ".gitkeep"]
    if existing:
        raise SystemExit(f"[ABORT] {mdir} is not empty. Phase 3 never overwrites results; "
                         "move/archive the directory (or use --resume to skip completed models).")

    fh = logging.FileHandler(mdir / "train.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)
    t0 = time.time()
    old_metrics_dir = config.METRICS_DIR
    status = {"model": name, "started": now(), "status": "running"}
    try:
        seed = base_cfg["random_seed"]
        set_seed(seed)
        loaders = dataset.build_dataloaders(manifest, split_result,
                                            train_transform=augmentation.get_train_transform(),
                                            eval_transform=augmentation.get_eval_transform())
        class_w = train.class_weights_from_manifest(manifest, split_result, NUM_CLASSES)
        model = build_model(name)
        n_params = int(sum(p.numel() for p in model.parameters()))

        cfg = {**base_cfg, "experiment_id": f"V2-B0-{name.upper()}", "model": MODEL_LABEL[name],
               "n_parameters": n_params, "class_weights": [float(w) for w in class_w.cpu()],
               "class_order": list(config.CLASS_NAMES),
               "dataset": "outputs/final_400/final_400.csv", "split": "outputs/splits/split_assignments.csv",
               "split_sizes": {k: len(v) for k, v in split_result["splits"].items()},
               "val_split_hash": checkpoint_utils.hash_split(manifest, split_result, "val"),
               "test_split_hash": checkpoint_utils.hash_split(manifest, split_result, "test"),
               "git": git, "environment": env, "started": status["started"]}
        write_json(cfg, mdir / "config_snapshot.json")

        # ---------------- training (VALIDATION only) ----------------
        ckpt = mdir / f"{name}_best.pt"
        config.METRICS_DIR = mdir   # train_model writes training_history.json here, not to the shared dir
        hist = train.train_model(
            model, loaders["train"], loaders["val"], epochs=config.EPOCHS, lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY, checkpoint_path=ckpt, device=config.DEVICE,
            max_attempts=config.MAX_ATTEMPTS, base_seed=seed, class_weights=class_w,
            patience=config.EARLY_STOP_PATIENCE, scheduler_kind=config.LR_SCHEDULER)
        config.METRICS_DIR = old_metrics_dir
        train_seconds = time.time() - t0

        best_epoch = best_epoch_from_history(hist)
        attempts_used = int(hist["seed"] - seed + 1)
        meta = checkpoint_utils.build_metadata(name, config, split_result, manifest, hist, pretrained=False)
        meta.update({"experiment_id": cfg["experiment_id"], "best_epoch": best_epoch,
                     "attempts_used": attempts_used, "checkpoint_sha256_at_save": sha_file(ckpt)})
        checkpoint_utils.save_metadata(ckpt, meta)

        # ---------------- FREEZE checkpoint (before any test access) ----------------
        frozen = {"frozen_at": now(), "checkpoint": ckpt.name, "checkpoint_sha256": sha_file(ckpt),
                  "selection_metric": "validation macro-F1 (tie-break: val accuracy, then lower val loss)",
                  "best_epoch": best_epoch, "attempts_used": attempts_used, "seed_of_selected_attempt": hist["seed"],
                  "best_val_macro_f1": hist["best_val_macro_f1"], "best_val_acc": hist["best_val_acc"],
                  "best_val_loss": hist["best_val_loss"], "degenerate": hist["degenerate"]}
        write_json(frozen, mdir / "frozen_checkpoint.json")

        # ---------------- validation re-evaluation of the frozen checkpoint ----------------
        model = build_model(name)
        model.load_state_dict(load_state(ckpt, config.DEVICE))
        vpred = evaluate.deterministic_predict_dataset(model, loaders["val"], device=config.DEVICE)
        v_sel_f1 = float(f1_score(vpred["y_true"], vpred["y_pred"], average="macro", zero_division=0))
        v_acc = float(accuracy_score(vpred["y_true"], vpred["y_pred"]))
        if abs(v_sel_f1 - hist["best_val_macro_f1"]) > 1e-6 or abs(v_acc - hist["best_val_acc"]) > 1e-6:
            raise RuntimeError(
                "Reloaded checkpoint does not reproduce the recorded validation metrics "
                f"(f1 {v_sel_f1:.6f} vs {hist['best_val_macro_f1']:.6f}; acc {v_acc:.6f} vs "
                f"{hist['best_val_acc']:.6f}). Test evaluation withheld.")
        val_m = full_metrics(vpred["y_true"], vpred["y_pred"], vpred["p_mean"], with_roc=False)
        val_m.update({
            "role": "VALIDATION (model selection)",
            "selection_macro_f1_as_computed_by_train_py": v_sel_f1,
            "selection_macro_f1_note": (
                "src/train.py calls f1_score(average='macro') without labels=, so the average is over "
                "labels present in y_true UNION y_pred. Validation has no Moderate images: if the model "
                "never predicts Moderate this is a 3-class mean; if it does, an extra F1=0 class lowers it. "
                "See macro_f1_present_classes for the 3-class value."),
            "val_loss_at_best_epoch": hist["best_val_loss"], "best_epoch": best_epoch,
            "checkpoint_reproduces_training_metrics": True})
        write_json(val_m, mdir / "val_metrics.json")
        predict_table(loaders["val"], vpred).to_csv(mdir / "val_predictions.csv", index=False)
        evaluate.plot_confusion_matrix(vpred["y_true"], vpred["y_pred"], mdir / "val_confusion_matrix.png")

        # ---------------- TEST: once, frozen checkpoint ----------------
        test_path = mdir / "test_metrics.json"
        if test_path.exists():
            raise RuntimeError("test_metrics.json already exists; test evaluation is single-use")
        tpred = evaluate.deterministic_predict_dataset(model, loaders["test"], device=config.DEVICE)
        test_m = full_metrics(tpred["y_true"], tpred["y_pred"], tpred["p_mean"], with_roc=True)
        test_m.update({"role": "TEST (final held-out evaluation; not used for any decision)",
                       "checkpoint_sha256": frozen["checkpoint_sha256"],
                       "moderate_caveat": ("Moderate Demented test images all come from one subject "
                                           "(OAS1_0308); training Moderate images from one other subject "
                                           "(OAS1_0351). Moderate test metrics are single-subject.")})
        write_json(test_m, test_path)
        predict_table(loaders["test"], tpred).to_csv(mdir / "test_predictions.csv", index=False)
        evaluate.plot_confusion_matrix(tpred["y_true"], tpred["y_pred"], mdir / "test_confusion_matrix.png")
        evaluate.plot_reliability_diagram(tpred["y_true"], tpred["p_mean"], mdir / "test_reliability_diagram.png")

        total = time.time() - t0
        summary = {**status, "status": "complete", "finished": now(), "train_seconds": round(train_seconds, 1),
                   "total_seconds": round(total, 1), "best_epoch": best_epoch, "epochs_run_last_attempt": hist["epochs_run"],
                   "attempts_used": attempts_used, "degenerate": hist["degenerate"],
                   "checkpoint_sha256": frozen["checkpoint_sha256"], "n_parameters": n_params}
        write_json(summary, mdir / "status.json")
        return summary
    except BaseException as e:  # noqa: BLE001
        config.METRICS_DIR = old_metrics_dir
        tb = traceback.format_exc()
        (mdir / "FAILED.txt").write_text(tb, encoding="utf-8")
        write_json({**status, "status": "failed", "finished": now(), "error": f"{type(e).__name__}: {e}"},
                   mdir / "status.json")
        logger.error("%s FAILED:\n%s", name, tb)
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        return {"model": name, "status": "failed", "error": f"{type(e).__name__}: {e}"}
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def collect_results() -> dict:
    res = {}
    for n in MODEL_NAMES:
        d = BASE / n
        r = {"status": "not_run"}
        if (d / "status.json").exists():
            r = json.loads((d / "status.json").read_text())
        for fn, key in [("config_snapshot.json", "config"), ("frozen_checkpoint.json", "frozen"),
                        ("val_metrics.json", "val"), ("test_metrics.json", "test"),
                        ("training_history.json", "history")]:
            if (d / fn).exists():
                r[key] = json.loads((d / fn).read_text())
        res[n] = r
    return res


def build_reports(ctx: dict) -> None:
    res = collect_results()
    report = {"phase": "Phase 3 - V2 baseline", "generated": now(), **ctx, "models": res}
    write_json(report, BASE / "baseline_report.json")

    cfg = ctx["baseline_config"]
    L = []
    A = L.append
    A("# Phase 3 — V2 Baseline Report\n")
    A(f"Generated: {report['generated']}  \nGit commit: `{ctx['git']['commit']}` "
      f"(branch `{ctx['git']['branch']}`; working-tree changes at start: {len(ctx['git']['status_porcelain'])})\n")
    A("**VALIDATION = model selection. TEST = final held-out evaluation; no decision was made from test results.**\n")
    A("## A. Baseline configuration (identical for all three models)\n")
    A("| Item | Value |\n|---|---|")
    for k in ["mode", "image_size", "random_seed", "optimizer", "learning_rate", "weight_decay", "batch_size",
              "epochs", "early_stop_patience", "lr_scheduler", "scheduler_params", "loss", "class_balance_strategy",
              "dropout_p", "max_attempts", "checkpoint_selection", "preprocessing", "train_augmentation",
              "eval_augmentation", "pretrained"]:
        A(f"| {k} | {cfg[k]} |")
    A(f"| ViT | {cfg['vit']} |")
    A(f"| Device | {ctx['environment']['device']} ({ctx['environment'].get('gpu')}) |")
    A("")
    A("## B. Dataset / split (locked V1)\n")
    A("| Split | Images | Subjects | " + " | ".join(config.CLASS_NAMES) + " |\n|---|---:|---:|" + "---:|" * NUM_CLASSES)
    for s, d in ctx["dataset"].items():
        A(f"| {s} | {d['images']} | {d['subjects']} | " +
          " | ".join(f"{d['images_per_class'][c]} ({d['subjects_per_class'][c]} subj)" for c in config.CLASS_NAMES) + " |")
    A("\nModerate Demented: OAS1_0351 (60 images) → train; OAS1_0308 (40 images) → test; **0 Moderate subjects in validation**.\n")

    for letter, n in zip("CDE", MODEL_NAMES):
        r = res[n]
        A(f"## {letter}. {MODEL_LABEL[n]}\n")
        if r.get("status") != "complete":
            A(f"**Status: {r.get('status')}** {r.get('error', '')}\n")
            continue
        fz, v, t = r["frozen"], r["val"], r["test"]
        A(f"Parameters: {r['n_parameters']:,} · best epoch: {r['best_epoch']} (last epoch reached: "
          f"{r['epochs_run_last_attempt']}) · attempts used: {r['attempts_used']} · degenerate: {r['degenerate']} · "
          f"train time: {r['train_seconds']}s · checkpoint sha256: `{fz['checkpoint_sha256'][:16]}…`\n")
        A("**Validation (selection)**\n")
        A(f"- accuracy {_f(v['accuracy'])} · selection macro-F1 (as computed by train.py) {_f(v['selection_macro_f1_as_computed_by_train_py'])} "
          f"· macro-F1 over the 3 classes present {_f(v['macro_f1_present_classes'])} · loss {_f(v['val_loss_at_best_epoch'])}")
        A(f"- ECE {_f(v['ece_10bins'])} · Brier {_f(v['brier'])} · ROC-AUC: {v['roc_auc_note']}\n")
        A("**Test (frozen checkpoint, evaluated once)**\n")
        A(f"- accuracy {_f(t['accuracy'])} · macro-P {_f(t['macro_precision_all4'])} · macro-R {_f(t['macro_recall_all4'])} "
          f"· macro-F1 {_f(t['macro_f1_all4'])} · ROC-AUC {_f(t['roc_auc_macro_ovr'])} · ECE {_f(t['ece_10bins'])} · Brier {_f(t['brier'])}\n")
        A("| Class | Val P | Val R | Val F1 | Val n | Test P | Test R | Test F1 | Test n |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c in config.CLASS_NAMES:
            a, b = v["per_class"][c], t["per_class"][c]
            A(f"| {c} | {_f(a['precision'],3)} | {_f(a['recall'],3)} | {_f(a['f1'],3)} | {a['support']} | "
              f"{_f(b['precision'],3)} | {_f(b['recall'],3)} | {_f(b['f1'],3)} | {b['support']} |")
        A("\nTest confusion matrix (rows = true, cols = predicted; order: " + ", ".join(config.CLASS_NAMES) + ")\n")
        A("```\n" + "\n".join(" ".join(f"{x:4d}" for x in row) for row in t["confusion_matrix"]) + "\n```\n")

    A("## F. Validation-selection procedure\n")
    A("Within each run, the checkpoint is the epoch with the highest validation macro-F1; ties are broken by validation "
      "accuracy, then lower validation loss (`src/train.py`). Training stops after 6 epochs without improvement. If the "
      "best checkpoint is degenerate (single-class predictions or accuracy ≤ majority baseline) the repo retries once "
      "with a new seed, dropout 0.25 and 1.5× epochs; the best attempt by validation macro-F1 is kept. The checkpoint "
      "is frozen (sha256 recorded in `frozen_checkpoint.json`), reloaded, and verified to reproduce the recorded "
      "validation metrics before the test split is read.\n")
    A("## G. Final test results (summary)\n")
    A("| Model | Test acc | Test macro-F1 | Test ECE | Test Brier | Val acc | Val sel. macro-F1 | Best epoch |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for n in MODEL_NAMES:
        r = res[n]
        if r.get("status") == "complete":
            A(f"| {MODEL_LABEL[n]} | {_f(r['test']['accuracy'])} | {_f(r['test']['macro_f1_all4'])} | {_f(r['test']['ece_10bins'])} | "
              f"{_f(r['test']['brier'])} | {_f(r['val']['accuracy'])} | {_f(r['val']['selection_macro_f1_as_computed_by_train_py'])} | {r['best_epoch']} |")
        else:
            A(f"| {MODEL_LABEL[n]} | {r.get('status')} | | | | | | |")
    A("\nThis table is descriptive. It is not a model ranking: 44 validation / 83 test images from 27 / 28 subjects, a "
      "single seed, and a single-subject Moderate test class do not support claims that any model is superior.\n")
    A("## H. Runtime\n")
    for n in MODEL_NAMES:
        r = res[n]
        A(f"- {MODEL_LABEL[n]}: train {r.get('train_seconds', 'n/a')}s, total {r.get('total_seconds', 'n/a')}s")
    A("\n## I. Implementation issues encountered\n")
    for x in ctx["issues"] or ["None recorded."]:
        A(f"- {x}")
    A("\n## J. Deviations / notes on the documented baseline\n")
    for x in ctx["deviations"]:
        A(f"- {x}")
    A("\n## K. V1 integrity\n")
    h = ctx["v1_integrity"]
    A(f"- Locked-hash check before run: **{h['before_locked_hash_check']}**; after run: **{h['after_locked_hash_check']}**")
    for rel, d in h["locked_hashes_after"].items():
        A(f"  - `{rel}`: locked `{d['locked'][:16]}…`, matched via {d['matched_via']}")
    A(f"- Files snapshotted (final_400, splits, smoke_tests): {h['files_snapshotted']}; "
      f"modified: {h['diff']['modified']}, removed: {h['diff']['removed']}, added: {h['diff']['added']}")
    A(f"- Result: **{'V1 and smoke-test artifacts unchanged' if h['unchanged'] else 'CHANGE DETECTED — see JSON'}**\n")
    A("## L. Limitations\n")
    for x in ctx["limitations"]:
        A(f"- {x}")
    (BASE / "baseline_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


DEVIATIONS = [
    "Phase 3 uses the 'demo' profile in src/config.py (96×96, 25 epochs, patience 6, AdamW lr 1e-3 / wd 1e-4, batch 32, "
    "dropout 0.35, ReduceLROnPlateau, loss-weighted CE, seed 42). `run_pipeline.py --mode demo` only performs a forward-pass "
    "demonstration and does not train, so this dedicated runner calls src.train.train_model directly. The 'full' profile "
    "(64×64, 30 epochs) is NOT the baseline.",
    "src/train.py::train_model writes a single shared training_history.json to config.METRICS_DIR; the runner redirects "
    "METRICS_DIR to each model's own directory during training so nothing shared is overwritten.",
    "src/train.py computes validation macro-F1 with f1_score(average='macro') and no labels=. Validation has no Moderate "
    "images, so the selection metric averages over a prediction-dependent class set (3 or 4 classes). This is the documented "
    "repository behaviour and was kept unchanged; macro-F1 over the 3 present classes is reported alongside it.",
    "run_pipeline.run_demo builds DenseNet2D with dropout_p=0.3, whereas config.DROPOUT_P = 0.35; the baseline uses 0.35 for "
    "all models. DenseNet2D is the compact from-scratch network (the pretrained DenseNet121 path is not used).",
    "train_model's degenerate-checkpoint retry (MAX_ATTEMPTS=2) can change seed (+1), dropout (-0.1) and epochs (×1.5) for a "
    "second attempt. It is part of the existing configuration; attempts used and the seed of the selected attempt are recorded.",
    "train.py does not store best_epoch or write checkpoint metadata; the runner derives best_epoch with the same improvement "
    "rule and writes the .meta.json sidecar via checkpoint_utils.",
    "Smoke-test V1 hashes were recorded on a CRLF (Windows) checkout; Git blobs are LF. The runner accepts a locked hash if the "
    "raw bytes or the CRLF-normalised bytes match, and records which.",
    "Single seed (42); no repeated runs, so run-to-run variance is not estimated.",
]
LIMITATIONS = [
    "Moderate Demented comes from only two independent subjects. Validation has none, so validation metrics say nothing about "
    "Moderate performance; the Moderate test result is a single-subject result (OAS1_0308) after training on one other subject (OAS1_0351).",
    "Slices from the same subject are not independent; all reported metrics are slice-level, not patient-level. No subject-level aggregation was performed in Phase 3.",
    "Validation (44 images / 27 subjects) and test (83 images / 28 subjects) are small; small numeric differences between models are not evidence of superiority.",
    "2D slices from the OASIS-derived dataset only; no clinical validity is claimed.",
    "Test ROC-AUC is computed only because all four classes appear in the test split; it inherits the single-subject Moderate limitation.",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 V2 baseline runner")
    ap.add_argument("--preflight", action="store_true", help="checks only: no training, no test data access")
    ap.add_argument("--report-only", action="store_true", help="rebuild reports from saved results")
    ap.add_argument("--resume", action="store_true", help="skip models whose status.json says complete")
    ap.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    base_cfg = apply_and_check_baseline_config()
    git, env = git_info(), env_info()

    print("== Verifying locked V1 hashes ==")
    h_before = verify_locked_hashes()
    for k, v in h_before.items():
        print(f"  OK {k} (matched via {v['matched_via']})")
    snap_before = snapshot_tree()

    manifest, split_result = load_locked(verify_images=True)
    xcheck = crosscheck_data_acquisition(manifest)
    dset = describe_dataset(manifest)
    print(f"== Locked dataset OK: {dset['train']['images']}/{dset['val']['images']}/{dset['test']['images']} "
          f"(train/val/test); data_acquisition cross-check: {xcheck['status']} ==")
    print(f"== Device: {config.DEVICE} | image size {base_cfg['image_size']} | seed {base_cfg['random_seed']} ==")

    if args.preflight:
        for n in MODEL_NAMES:
            m = build_model(n)
            x = torch.randn(2, 1, *config.IMAGE_SIZE)
            m.eval()
            with torch.no_grad():
                out = m(x)
            assert tuple(out.shape) == (2, NUM_CLASSES)
            print(f"  {MODEL_LABEL[n]:11s} forward OK, {sum(p.numel() for p in m.parameters()):,} params")
        set_seed(base_cfg["random_seed"])
        ld = dataset.build_dataloaders(manifest, split_result, augmentation.get_train_transform(),
                                       augmentation.get_eval_transform())
        xb, yb = next(iter(ld["train"]))
        xv, _ = next(iter(ld["val"]))   # test loader intentionally not touched
        print(f"  train batch {tuple(xb.shape)}, val batch {tuple(xv.shape)}, "
              f"train class weights {[round(float(w), 3) for w in train.class_weights_from_manifest(manifest, split_result, NUM_CLASSES).cpu()]}")
        diff = diff_snapshots(snap_before, snapshot_tree())
        assert not any(diff.values()), diff
        print("== PREFLIGHT PASSED (nothing written, no training, test split untouched) ==")
        return

    BASE.mkdir(parents=True, exist_ok=True)
    issues = []
    if not args.report_only:
        for n in args.models:
            sj = BASE / n / "status.json"
            if args.resume and sj.exists() and json.loads(sj.read_text()).get("status") == "complete":
                print(f"== {MODEL_LABEL[n]}: already complete, skipping ==")
                continue
            print(f"\n==================== {MODEL_LABEL[n]} ====================")
            r = run_one(n, manifest, split_result, base_cfg, git, env)
            if r.get("status") != "complete":
                issues.append(f"{MODEL_LABEL[n]} failed: {r.get('error')} (traceback in outputs/v2/baseline/{n}/FAILED.txt)")

    h_after = verify_locked_hashes()
    snap_after = snapshot_tree()
    diff = diff_snapshots(snap_before, snap_after)
    unchanged = not any(diff.values())
    ctx = {"baseline_config": base_cfg, "git": git, "environment": env, "dataset": dset,
           "data_loader_crosscheck": xcheck, "issues": issues, "deviations": DEVIATIONS, "limitations": LIMITATIONS,
           "v1_integrity": {"before_locked_hash_check": "PASS", "after_locked_hash_check": "PASS",
                            "locked_hashes_before": h_before, "locked_hashes_after": h_after,
                            "files_snapshotted": len(snap_before), "diff": diff, "unchanged": unchanged}}
    build_reports(ctx)
    print(f"\nReports written to {BASE / 'baseline_report.md'} and baseline_report.json")
    if not unchanged:
        raise SystemExit(f"[ALERT] V1/smoke-test files changed during the run: {diff}")
    if issues:
        raise SystemExit("[DONE WITH FAILURES] " + "; ".join(issues))
    print("Phase 3 complete. STOP here — do not start Phase 4.")


if __name__ == "__main__":
    main()