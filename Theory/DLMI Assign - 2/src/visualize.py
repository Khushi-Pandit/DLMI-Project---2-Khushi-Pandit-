"""
visualize.py
------------
All plotting utilities for the BUS classification project.
  - confusion_matrix_plot()
  - metrics_comparison_bar()
  - roc_curves_multiclass()
  - feature_importance_plot()
  - learning_curve_plot()
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from itertools import cycle

from sklearn.metrics        import (confusion_matrix, roc_curve, auc)
from sklearn.preprocessing  import label_binarize

CLASS_NAMES = ["Benign", "Malignant", "Normal"]
PALETTE     = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# ─────────────────────────────────────────────────────────────────
# 1. Confusion Matrix Heatmap
# ─────────────────────────────────────────────────────────────────

def confusion_matrix_plot(y_true, y_pred, title: str, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────
# 2. Metrics Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────

def metrics_comparison_bar(results_list: list, save_path: str):
    """
    results_list : list of dicts with keys
                   {model_name, accuracy, balanced_acc, macro_f1, weighted_f1, auc_ovr}
    """
    metrics = ["accuracy", "balanced_acc", "macro_f1", "weighted_f1"]
    labels  = [r["model_name"] for r in results_list]
    x       = np.arange(len(labels))
    width   = 0.18

    fig, ax = plt.subplots(figsize=(max(10, 2*len(labels)), 6))

    for i, metric in enumerate(metrics):
        vals = [r[metric] for r in results_list]
        rects = ax.bar(x + i * width, vals, width,
                       label=metric.replace("_", " ").title(),
                       color=PALETTE[i], edgecolor="black", linewidth=0.5)
        for rect, v in zip(rects, vals):
            ax.text(rect.get_x() + rect.get_width()/2,
                    rect.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison – BUS Dataset", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────
# 3. Multi-class ROC Curves (OVR)
# ─────────────────────────────────────────────────────────────────

def roc_curves_multiclass(model, X_test, y_test,
                           model_name: str, save_path: str):
    n_classes = len(CLASS_NAMES)
    y_bin     = label_binarize(y_test, classes=list(range(n_classes)))
    y_score   = model.predict_proba(X_test)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i]        = auc(fpr[i], tpr[i])

    # micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc["micro"]              = auc(fpr["micro"], tpr["micro"])

    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = cycle(PALETTE)

    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f"{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})")

    ax.plot(fpr["micro"], tpr["micro"], color="black",
            linestyle="--", lw=1.5,
            label=f"Micro-avg (AUC = {roc_auc['micro']:.3f})")
    ax.plot([0,1],[0,1], "k:", lw=1)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves – {model_name}", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────
# 4. Feature Importance (tree-based models)
# ─────────────────────────────────────────────────────────────────

def feature_importance_plot(model, feature_names: list, save_path: str,
                             title: str = "Feature Importances"):
    if not hasattr(model, "feature_importances_"):
        print("[visualize] Model has no feature_importances_ – skipping.")
        return

    importances = model.feature_importances_
    idx         = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars    = ax.barh(sorted_names[::-1], sorted_vals[::-1],
                      color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Importance Score")
    ax.set_title(title, fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────
# 5. Learning Curve
# ─────────────────────────────────────────────────────────────────

def learning_curve_plot(train_sizes, train_scores, val_scores,
                         title: str, save_path: str):
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    val_mean   = np.mean(val_scores,   axis=1)
    val_std    = np.std(val_scores,    axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", color="#4C72B0", label="Train")
    ax.fill_between(train_sizes,
                    train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color="#4C72B0")
    ax.plot(train_sizes, val_mean, "s-", color="#DD8452", label="Validation")
    ax.fill_between(train_sizes,
                    val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color="#DD8452")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("F1-Score (macro)")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Saved → {save_path}")