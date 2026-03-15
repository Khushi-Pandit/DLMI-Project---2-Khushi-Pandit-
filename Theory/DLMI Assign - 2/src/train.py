"""
train.py
--------
End-to-end training pipeline for BUS imbalanced classification.

Pipeline:
  1. Generate / load data
  2. Apply oversampling (SMOTE) + augmentation (Gaussian Noise)
  3. Train 5 models
  4. Evaluate on held-out test set
  5. Save results, plots, and metrics CSV
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_loader    import generate_bus_dataset, load_and_split
from oversampling   import (apply_smote, apply_adasyn,
                             apply_borderline_smote, apply_random_oversample,
                             plot_class_distribution)
from augmentation   import augment_pipeline
from focal_loss     import compute_alpha_weights
from models         import (train_random_forest, train_gradient_boosting,
                             train_mlp, train_focal_rf, train_focal_gbm,
                             evaluate_model)
from visualize      import (confusion_matrix_plot, metrics_comparison_bar,
                             roc_curves_multiclass, feature_importance_plot)
from sklearn.model_selection import learning_curve
from visualize import learning_curve_plot

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)


def run_pipeline():
    print("\n" + "="*65)
    print("  BUS IMBALANCED CLASSIFICATION PIPELINE")
    print("="*65)

    # ── 1. Data ────────────────────────────────────────────────────
    print("\n[Step 1] Generating / loading dataset …")
    generate_bus_dataset(n_samples=2000, save_dir=DATA_DIR)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = \
        load_and_split(os.path.join(DATA_DIR, "bus_dataset.csv"))

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ── 2. Oversampling ────────────────────────────────────────────
    print("\n[Step 2] Oversampling comparison …")
    _, y_smote  = apply_smote(X_train, y_train)
    _, y_adasyn = apply_adasyn(X_train, y_train)
    _, y_border = apply_borderline_smote(X_train, y_train)
    _, y_random = apply_random_oversample(X_train, y_train)

    plot_class_distribution(
        y_train,
        {"SMOTE": y_smote, "ADASYN": y_adasyn,
         "BorderlineSMOTE": y_border, "RandomOS": y_random},
        save_path=os.path.join(RESULTS_DIR, "oversampling_distribution.png")
    )

    # Use SMOTE as the main oversampler
    X_os, y_os = apply_smote(X_train, y_train)

    # ── 3. Augmentation ────────────────────────────────────────────
    print("\n[Step 3] Augmentation (Gaussian Noise on SMOTE-balanced data) …")
    X_aug, y_aug = augment_pipeline(X_os, y_os, method="gaussian")
    print(f"  Final training set: {X_aug.shape}")

    # ── 4. Model training ─────────────────────────────────────────
    print("\n[Step 4] Training models …")

    models = {}

    # 4a. RF baseline (no oversampling)
    print("\n  → RandomForest (baseline, no oversampling)")
    models["RF_Baseline"] = train_random_forest(X_train, y_train)

    # 4b. RF + SMOTE
    print("\n  → RandomForest + SMOTE")
    models["RF_SMOTE"] = train_random_forest(X_os, y_os, class_weight=None)

    # 4c. RF + SMOTE + Augmentation
    print("\n  → RandomForest + SMOTE + Augmentation")
    models["RF_SMOTE_Aug"] = train_random_forest(X_aug, y_aug, class_weight=None)

    # 4d. FocalRF + SMOTE + Augmentation
    print("\n  → RandomForest + FocalLoss + SMOTE + Augmentation")
    models["FocalRF"] = train_focal_rf(X_aug, y_aug, gamma=2.0)

    # 4e. FocalGBM + SMOTE + Augmentation
    print("\n  → GradientBoosting + FocalLoss + SMOTE + Augmentation")
    models["FocalGBM"] = train_focal_gbm(X_aug, y_aug, gamma=2.0)

    # ── 5. Evaluation ─────────────────────────────────────────────
    print("\n[Step 5] Evaluating on test set …")
    all_results = []

    for name, model in models.items():
        res = evaluate_model(model, X_test, y_test, name)
        all_results.append(res)

        # Confusion matrix
        y_pred = model.predict(X_test)
        confusion_matrix_plot(
            y_test, y_pred,
            title=f"Confusion Matrix – {name}",
            save_path=os.path.join(RESULTS_DIR, f"cm_{name}.png")
        )

        # ROC curves
        roc_curves_multiclass(
            model, X_test, y_test,
            model_name=name,
            save_path=os.path.join(RESULTS_DIR, f"roc_{name}.png")
        )

    # Feature importance (best model = FocalGBM)
    feature_importance_plot(
        models["FocalGBM"], feature_names,
        save_path=os.path.join(RESULTS_DIR, "feature_importance_FocalGBM.png"),
        title="Feature Importances – FocalGBM"
    )

    # Metrics comparison bar chart
    metrics_comparison_bar(
        all_results,
        save_path=os.path.join(RESULTS_DIR, "metrics_comparison.png")
    )

    # Learning curve for best model
    print("\n[Step 5b] Learning curve for FocalRF …")
    from sklearn.metrics import make_scorer, f1_score as f1
    scorer = make_scorer(f1, average="macro")
    train_sizes, train_sc, val_sc = learning_curve(
        train_random_forest(X_aug, y_aug, class_weight=None),
        X_aug, y_aug,
        cv=3, scoring=scorer,
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )
    learning_curve_plot(
        train_sizes, train_sc, val_sc,
        title="Learning Curve – RF (SMOTE + Augmentation)",
        save_path=os.path.join(RESULTS_DIR, "learning_curve.png")
    )

    # ── 6. Save metrics CSV ───────────────────────────────────────
    print("\n[Step 6] Saving metrics …")
    metrics_keys = ["model_name", "accuracy", "balanced_acc",
                    "macro_f1", "weighted_f1", "auc_ovr"]
    summary = [{k: r[k] for k in metrics_keys} for r in all_results]
    df_summary = pd.DataFrame(summary)
    csv_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nMetrics summary:\n{df_summary.to_string(index=False)}")
    print(f"\nSaved to {csv_path}")

    # ── 7. Save JSON summary ──────────────────────────────────────
    for r in all_results:
        r.pop("confusion_matrix", None)
        r.pop("classification_report", None)
    json_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*65)
    print("  PIPELINE COMPLETE")
    print(f"  Results → {RESULTS_DIR}")
    print("="*65)
    return df_summary


if __name__ == "__main__":
    run_pipeline()