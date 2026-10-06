"""Leakage-safe SUBJECT-LEVEL split for final_400.csv (seed 42).

Strategy: per class, subjects are shuffled (seeded) and assigned whole to
train/val/test (~70/15/15 by subject count; min 1 subject in val and test).
Moderate Demented has only 2 subjects (OAS1_0308, OAS1_0351), so it CANNOT
appear independently in all three partitions: the subject with more images
goes to train, the other to test; val has NO Moderate. A supplementary
2-fold subject-held-out CV is produced for Moderate.
No image-level splitting is ever done.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import config

FINAL_CSV = config.OUTPUT_ROOT / "final_400" / "final_400.csv"
OUT_DIR = config.OUTPUT_ROOT / "splits"
KEEP = ["filepath", "class_name", "class_id", "subject_id", "session", "scan", "slice_idx"]
MOD_CLASS = "Moderate Demented"
MOD_SUBJECTS = ["OAS1_0308", "OAS1_0351"]


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_split(df: pd.DataFrame, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assign = {}
    for cname in config.CLASS_NAMES:  # fixed order -> deterministic
        g = df[df.class_name == cname]
        if cname == MOD_CLASS:
            counts = g.groupby("subject_id").size()
            order = sorted(counts.index, key=lambda s: (-counts[s], s))
            assign[order[0]] = "train"
            assign[order[1]] = "test"
            continue
        subs = sorted(g.subject_id.unique())
        rng.shuffle(subs)
        n = len(subs)
        n_val = max(1, round(n * config.SPLIT_RATIOS["val"]))
        n_test = max(1, round(n * config.SPLIT_RATIOS["test"]))
        for i, s in enumerate(subs):
            assign[s] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    out = df[KEEP].copy()
    out["split"] = out.subject_id.map(assign)
    return out


def moderate_cv(df: pd.DataFrame) -> pd.DataFrame:
    m = df[df.class_name == MOD_CLASS][KEEP]
    rows = []
    for fold, held in enumerate(MOD_SUBJECTS, 1):
        t = m.copy()
        t["fold"] = fold
        t["heldout_subject"] = held
        t["role"] = np.where(t.subject_id == held, "heldout_test", "train")
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def validate(df: pd.DataFrame, a: pd.DataFrame) -> dict:
    key = ["class_name", "subject_id", "session", "scan", "slice_idx"]
    sp = a.groupby("subject_id").split.nunique()
    mod = a[a.class_name == MOD_CLASS].groupby("subject_id").split.agg(lambda x: sorted(set(x)))
    r = {
        "n_images_400": len(a) == 400,
        "100_per_class": bool((a.class_name.value_counts() == 100).all()),
        "all_filepaths_once": sorted(a.filepath) == sorted(df.filepath),
        "no_duplicate_filepath": not a.filepath.duplicated().any(),
        "no_duplicate_class_subject_session_scan_slice": not a.duplicated(key).any(),
        "no_subject_in_multiple_partitions": bool((sp == 1).all()),
        "all_four_classes_present": set(a.class_name) == set(config.CLASS_NAMES),
        "split_values_valid": set(a.split) <= {"train", "val", "test"},
        "moderate_subjects_exactly_once": all(s in mod.index and len(mod[s]) == 1 for s in MOD_SUBJECTS),
    }
    r["ALL_PASSED"] = all(r.values())
    return r


def run() -> dict:
    h0 = _sha(FINAL_CSV)
    df = pd.read_csv(FINAL_CSV)
    a = build_split(df)
    a2 = build_split(df)
    cv = moderate_cv(df)
    v = validate(df, a)
    v["deterministic_rerun_identical"] = a.equals(a2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a.to_csv(OUT_DIR / "split_assignments.csv", index=False)
    cv.to_csv(OUT_DIR / "moderate_subject_cv.csv", index=False)

    order = ["train", "val", "test"]
    img = a.pivot_table(index="class_name", columns="split", values="filepath", aggfunc="count", fill_value=0).reindex(index=config.CLASS_NAMES, columns=order, fill_value=0)
    sub = a.pivot_table(index="class_name", columns="split", values="subject_id", aggfunc="nunique", fill_value=0).reindex(index=config.CLASS_NAMES, columns=order, fill_value=0)
    summ = pd.concat([img.add_suffix("_images"), sub.add_suffix("_subjects")], axis=1)
    summ.loc["TOTAL"] = [a[a.split == s].shape[0] for s in order] + [a[a.split == s].subject_id.nunique() for s in order]
    summ.reset_index().rename(columns={"index": "class_name"}).to_csv(OUT_DIR / "split_summary.csv", index=False)
    ss = a.groupby(["subject_id", "class_name", "split"]).size().rename("n_images").reset_index()
    ss.to_csv(OUT_DIR / "subject_split_summary.csv", index=False)

    v["final_400_unchanged"] = _sha(FINAL_CSV) == h0
    v["ALL_PASSED"] = all(x for k, x in v.items() if k != "ALL_PASSED")
    mod_assign = {s: a[a.subject_id == s].split.iloc[0] for s in MOD_SUBJECTS}
    mod_n = {s: int((a.subject_id == s).sum()) for s in MOD_SUBJECTS}
    folds = {f"fold{i}": {"heldout": h, "train": [s for s in MOD_SUBJECTS if s != h]} for i, h in enumerate(MOD_SUBJECTS, 1)}
    limits = [
        "Moderate Demented has only 2 subjects: no conventional 3-way subject-independent split exists for it.",
        f"Validation contains NO Moderate images; test Moderate comes from a single subject ({[s for s in MOD_SUBJECTS if mod_assign[s]=='test'][0]}), so Moderate test metrics are single-subject and high-variance.",
        "Moderate train comes from a single subject: the model sees little Moderate anatomical variability.",
        "Val-based model selection/early stopping/calibration cannot assess Moderate.",
        "Image ratios deviate from 70/15/15; subject independence prioritised.",
        "Moderate 2-fold CV is supplementary; each fold tests on one subject only. Other classes keep main-split assignments.",
    ]
    rep = {"strategy": "subject-level, class-stratified (seeded shuffle, seed=42); Moderate assigned deterministically (more-image subject->train, other->test)",
           "seed": config.RANDOM_SEED, "image_counts": img.to_dict("index"), "subject_counts": sub.to_dict("index"),
           "totals_images": {s: int((a.split == s).sum()) for s in order},
           "totals_subjects": {s: int(a[a.split == s].subject_id.nunique()) for s in order},
           "moderate_assignment": mod_assign, "moderate_image_counts": mod_n, "moderate_cv": folds,
           "validation": v, "limitations": limits}
    (OUT_DIR / "split_report.json").write_text(json.dumps(rep, indent=2, default=int))
    md = ["# Split report", f"**Strategy:** {rep['strategy']}", "", "## Counts", summ.to_markdown(), "",
          "## Moderate Demented", *[f"- {s}: {mod_assign[s]} ({mod_n[s]} images)" for s in MOD_SUBJECTS],
          *[f"- {k}: held out {x['heldout']}, train {x['train']}" for k, x in folds.items()], "",
          "## Validation", *[f"- {k}: {x}" for k, x in v.items()], "", "## Limitations", *[f"- {l}" for l in limits]]
    (OUT_DIR / "split_report.md").write_text("\n".join(md))
    return rep


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ["totals_images", "totals_subjects", "image_counts", "subject_counts", "moderate_assignment", "moderate_image_counts", "validation"]}, indent=1, default=int))