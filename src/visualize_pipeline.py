"""
Visual Output Pipeline (input -> preprocessed -> classified -> final output)
==============================================================================
This is not one of the project document's 12 numbered modules -- it's a
direct response to the requirement that this project's outputs be shown as
actual images, not only metric tables and score plots. It sits downstream
of everything else already built: it loads one raw slice, runs it through
the real preprocessing, the real trained model(s), and the REAL
uncertainty mechanism of whichever combination is being visualized (Monte
Carlo Dropout, Deep Ensembles, or BEDL -- not always MC Dropout), and
renders the four requested stages as one figure per sample:

  1. Input image        - the raw slice exactly as it exists on disk.
  2. Preprocessed image - post Algorithm-1-equivalent processing (resized,
                           intensity-normalized), rescaled only for display
                           (the model itself consumes the normalized array).
  3. Classified image   - the slice with a Grad-CAM saliency overlay, showing
                           which anatomical regions drove the prediction,
                           captioned with predicted vs. true class.
  4. Final output       - a decision-support style summary panel: predicted
                           stage, confidence, per-class probability bars,
                           predictive uncertainty, and a plain-language
                           flag when the case looks like it needs review.

BUG FIX: earlier versions of this module (and of run_pipeline.py's caller)
hardcoded MC Dropout on a single backbone checkpoint for panel 4,
regardless of which combination the Decision Layer actually recommended.
If the recommended combination was e.g. "CNN2D+Deep_Ensembles", the visual
demo silently used the plain CNN2D+MC_Dropout model instead -- which can
(and did, in one run) collapse to predicting the majority class for every
single sample, making the demo look far worse than the actually-recommended
combination. `visualize_sample`/`visualize_samples_per_class` now take an
explicit `predict_fn(x) -> dict(p_mean, y_pred, variance, entropy)`, so the
caller supplies the real prediction mechanism of whatever combination is
being shown (MC Dropout, Deep Ensembles average, or BEDL) instead of this
module silently assuming MC Dropout. See `build_predict_fn()` below and
run_pipeline.py's STAGE 5 for how the right one gets selected.

Grad-CAM is a value-add for interpretability; it is a standard, well-known
technique (gradient-weighted class-activation mapping) and does not require
any additional training. It always runs against ONE representative model
(the backbone itself for MC Dropout/BEDL, or ensemble member 0 for Deep
Ensembles, since Grad-CAM is inherently single-model) -- this is noted in
the figure only implicitly via the caption; the FINAL OUTPUT panel's
probabilities/uncertainty are what actually reflect the full combination.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config, preprocessing
from .uncertainty.mc_dropout import mc_dropout_predict


# ---------------------------------------------------------------------------
# Prediction-mechanism dispatch -- the actual fix
# ---------------------------------------------------------------------------
def build_predict_fn(combination_name: str, checkpoints: dict, model_classes: dict):
    """Given a combination name like 'CNN2D+Deep_Ensembles' and the
    checkpoint registry `comparative_evaluation.run_comparative_study()`
    returns, build (predict_fn, gradcam_model):
      predict_fn(x)   -> dict(p_mean, y_pred, variance, entropy), using the
                          REAL mechanism for this combination.
      gradcam_model     -> one loaded nn.Module to run Grad-CAM against.

    This is the single place that decides how to visualize a combination,
    so run_pipeline.py never has to special-case backbone/method pairs
    itself and can't silently fall back to the wrong one.
    """
    from .uncertainty import deep_ensembles, bedl as bedl_mod

    backbone_name, method = combination_name.split("+")
    model_cls = model_classes[backbone_name]
    model_fn = lambda: model_cls(num_classes=len(config.CLASS_NAMES), dropout_p=config.DROPOUT_P)
    ckpt_info = checkpoints[combination_name]

    if method == "MC_Dropout":
        model = model_fn()
        model.load_state_dict(torch.load(ckpt_info["path"], map_location="cpu"))
        predict_fn = lambda x: mc_dropout_predict(model, x, T=config.MC_DROPOUT_SAMPLES)
        return predict_fn, model

    if method == "Deep_Ensembles":
        ckpt_paths = ckpt_info["paths"]
        predict_fn = lambda x: deep_ensembles.deep_ensemble_predict(model_fn, ckpt_paths, x)
        # Grad-CAM needs one concrete model -- use ensemble member 0.
        gradcam_model = model_fn()
        gradcam_model.load_state_dict(torch.load(ckpt_paths[0], map_location="cpu"))
        return predict_fn, gradcam_model

    if method == "BEDL":
        model = model_fn()
        model.load_state_dict(torch.load(ckpt_info["path"], map_location="cpu"))
        predict_fn = lambda x: bedl_mod.bedl_predict(model, x)
        return predict_fn, model

    raise ValueError(f"Unknown uncertainty method '{method}' in combination '{combination_name}'")


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def grad_cam(model: torch.nn.Module, x: torch.Tensor, target_class: int = None,
             layer_name: str = "features") -> tuple:
    """Gradient-weighted Class Activation Mapping against the named feature
    layer (the last conv block's output, before global pooling, for both
    CNN2D and DenseNet2D -- both expose this as `model.features`)."""
    model.eval()
    activations = {}

    def fwd_hook(module, inp, out):
        out.retain_grad()
        activations["value"] = out

    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(fwd_hook)

    logits = model(x)
    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    model.zero_grad()
    score = logits[0, target_class]
    score.backward()
    handle.remove()

    act = activations["value"]           # (1, C, H', W')
    grad = act.grad                      # (1, C, H', W')
    weights = grad.mean(dim=(2, 3), keepdim=True)     # global-average-pooled gradients
    cam = F.relu((weights * act).sum(dim=1))          # (1, H', W')
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear",
                         align_corners=False).squeeze().detach().numpy()
    return cam, target_class, float(torch.softmax(logits, dim=1)[0, target_class].item())


# ---------------------------------------------------------------------------
# Full 4-panel figure for one sample
# ---------------------------------------------------------------------------
def visualize_sample(filepath: str, gradcam_model: torch.nn.Module,
                      predict_fn=None,
                      true_class_idx: int = None,
                      out_path: Path = None,
                      layer_name: str = "features") -> dict:
    """Produce and save the 4-panel figure (input / preprocessed /
    classified+Grad-CAM / final decision-support output) for one image,
    and return the underlying numbers so they can also be inspected
    programmatically.

    `predict_fn(x) -> dict(p_mean, y_pred, variance, entropy)` supplies the
    REAL prediction mechanism for panel 4 (MC Dropout / Deep Ensembles /
    BEDL). If not given, defaults to MC Dropout on `gradcam_model` for
    backwards compatibility -- but callers driving this from a comparative
    study should always pass the actual combination's predict_fn (see
    build_predict_fn() above).
    """
    if predict_fn is None:
        predict_fn = lambda x: mc_dropout_predict(gradcam_model, x, T=config.MC_DROPOUT_SAMPLES)

    # --- Panel 1: input image, exactly as stored on disk ---
    raw_arr = preprocessing.load_slice_as_array(filepath)

    # --- Panel 2: preprocessed image (what the model actually receives) ---
    std_arr = preprocessing.preprocess_slice(filepath)
    x = torch.from_numpy(std_arr).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)

    # --- Panel 3: classified image with Grad-CAM (single representative model) ---
    cam, pred_idx_gradcam, _ = grad_cam(gradcam_model, x.clone(), layer_name=layer_name)

    # --- Panel 4: final decision-support output (REAL combination mechanism) ---
    pred_out = predict_fn(x)
    p_mean = pred_out["p_mean"][0].detach().numpy()
    pred_idx = int(pred_out["y_pred"][0].item())
    top1_conf = float(p_mean[pred_idx])
    entropy = float(pred_out["entropy"][0].item())
    variance = float(pred_out["variance"][0].item())

    pred_label = config.CLASS_NAMES[pred_idx]
    true_label = config.CLASS_NAMES[true_class_idx] if true_class_idx is not None else None

    # simple, documented review-flag threshold -- not a validated clinical
    # threshold, just a legible demonstration of how uncertainty could gate
    # a downstream decision.
    max_entropy = np.log(len(config.CLASS_NAMES))
    needs_review = (entropy / max_entropy) > 0.5

    # --- Render the figure ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(raw_arr, cmap="gray")
    axes[0].set_title("1. Input Image")
    axes[0].axis("off")

    disp_std = (std_arr - std_arr.min()) / (std_arr.max() - std_arr.min() + 1e-8)
    axes[1].imshow(disp_std, cmap="gray")
    axes[1].set_title("2. Preprocessed\n(resized, normalized)")
    axes[1].axis("off")

    axes[2].imshow(disp_std, cmap="gray")
    axes[2].imshow(cam, cmap="jet", alpha=0.45)
    title_line2 = f"Pred: {pred_label} ({top1_conf:.0%})"
    if true_label is not None:
        title_line2 += f" | True: {true_label}"
    axes[2].set_title(f"3. Classified (Grad-CAM)\n{title_line2}")
    axes[2].axis("off")

    axes[3].axis("off")

    # BUG FIX: previously bar_ax/text were positioned using axes[3]'s
    # position BEFORE fig.tight_layout() ran. tight_layout() only knows
    # about the 4 subplot axes -- it has no idea bar_ax exists (it's added
    # manually via add_axes, not part of the subplot grid) -- so it shifts
    # axes[0..3] to fit the suptitle/titles while bar_ax stays frozen at
    # the stale pre-layout coordinates. That's exactly what produced the
    # "4. Final Output" title landing on top of panel 3 and the bars
    # drifting out of alignment. Fix: run tight_layout() (with room
    # reserved for the suptitle) and force a draw FIRST, so axes[3]'s
    # position is final, THEN place bar_ax/text relative to it.
    fig.suptitle(Path(filepath).name, fontsize=10, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.canvas.draw()  # finalize layout so get_position() below is accurate

    panel4_pos = axes[3].get_position()
    bar_ax = fig.add_axes([panel4_pos.x0, panel4_pos.y0 + 0.30,
                            panel4_pos.width, panel4_pos.height * 0.55])
    bar_colors = ["#2ca02c" if i == pred_idx else "#4c72b0" for i in range(len(config.CLASS_NAMES))]
    bar_ax.barh(config.CLASS_NAMES, p_mean, color=bar_colors)
    bar_ax.set_xlim(0, 1)
    bar_ax.set_xlabel("Predictive probability")
    bar_ax.set_title("4. Final Output", loc="left")

    flag_text = "⚠ Flagged for review (high uncertainty)" if needs_review else "✓ Confident prediction"
    flag_color = "#b22222" if needs_review else "#2ca02c"
    summary = (
        f"Predicted stage: {pred_label}\n"
        f"Confidence: {top1_conf:.1%}\n"
        f"Predictive entropy: {entropy:.3f} (max {max_entropy:.3f})\n"
        f"Predictive variance: {variance:.4f}"
    )
    text_top = panel4_pos.y0 + 0.24
    fig.text(panel4_pos.x0, text_top, summary, fontsize=9, va="top", color="black")
    fig.text(panel4_pos.x0, panel4_pos.y0 + 0.04, flag_text, fontsize=9.5, va="bottom",
              color=flag_color, weight="bold")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "filepath": filepath,
        "predicted_class": pred_label,
        "true_class": true_label,
        "confidence": top1_conf,
        "p_mean": p_mean.tolist(),
        "entropy": entropy,
        "variance": variance,
        "needs_review": bool(needs_review),
        "figure_path": str(out_path) if out_path else None,
    }


def visualize_samples_per_class(manifest, gradcam_model, predict_fn=None, n_per_class: int = 1,
                                 out_dir: Path = config.PLOTS_DIR / "sample_pipeline") -> list:
    """Convenience wrapper: pick n_per_class test-like samples per class
    and render the 4-panel figure for each, using `predict_fn` for the
    real combination mechanism if supplied (see visualize_sample)."""
    out_dir = Path(out_dir)
    results = []
    for class_label in config.CLASS_NAMES:
        rows = manifest[manifest["class_label"] == class_label].sample(
            n=min(n_per_class, (manifest["class_label"] == class_label).sum()),
            random_state=config.RANDOM_SEED,
        )
        for _, row in rows.iterrows():
            out_path = out_dir / f"{class_label}_{Path(row['filepath']).stem}.png"
            result = visualize_sample(
                row["filepath"], gradcam_model, predict_fn=predict_fn,
                true_class_idx=row["class_idx"],
                out_path=out_path,
            )
            results.append(result)
    return results


if __name__ == "__main__":
    from . import data_acquisition
    from .models.cnn2d import CNN2D

    manifest = data_acquisition.build_manifest()
    model = CNN2D(num_classes=len(config.CLASS_NAMES), dropout_p=config.DROPOUT_P)
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / "cnn2d_best.pt", map_location="cpu"))

    results = visualize_samples_per_class(manifest, model, n_per_class=1)
    for r in results:
        print(f"{Path(r['filepath']).name}: pred={r['predicted_class']} "
              f"true={r['true_class']} conf={r['confidence']:.1%} "
              f"review={r['needs_review']} -> {r['figure_path']}")