"""
models.py
---------
Classification models for BUS dataset:
  1. RandomForestClassifier  (baseline)
  2. GradientBoostingClassifier (strong tree baseline)
  3. MLPClassifier            (neural baseline)
  4. FocalWeightedRF          (RF + focal-loss sample weights)
  5. FocalWeightedGBM         (GBM + focal-loss sample weights)

All models expose a common interface: train() / evaluate()
"""

import numpy as np
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     roc_auc_score, f1_score, accuracy_score,
                                     balanced_accuracy_score)
from focal_loss              import FocalLossNumpy, compute_alpha_weights

RANDOM_STATE = 42
CLASS_NAMES  = ["Benign", "Malignant", "Normal"]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: evaluate any fitted sklearn model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    y_pred       = model.predict(X_test)
    y_prob       = model.predict_proba(X_test)

    acc          = accuracy_score(y_test, y_pred)
    bal_acc      = balanced_accuracy_score(y_test, y_pred)
    macro_f1     = f1_score(y_test, y_pred, average="macro")
    weighted_f1  = f1_score(y_test, y_pred, average="weighted")
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    cm  = confusion_matrix(y_test, y_pred)
    cr  = classification_report(y_test, y_pred,
                                 target_names=CLASS_NAMES, digits=4)

    results = {
        "model_name"   : model_name,
        "accuracy"     : round(acc, 4),
        "balanced_acc" : round(bal_acc, 4),
        "macro_f1"     : round(macro_f1, 4),
        "weighted_f1"  : round(weighted_f1, 4),
        "auc_ovr"      : round(auc, 4) if not np.isnan(auc) else "N/A",
        "confusion_matrix" : cm,
        "classification_report" : cr,
    }

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Balanced Acc     : {bal_acc:.4f}")
    print(f"  Macro F1         : {macro_f1:.4f}")
    print(f"  Weighted F1      : {weighted_f1:.4f}")
    print(f"  AUC (OVR macro)  : {auc:.4f}")
    print(f"\n{cr}")
    print(f"  Confusion Matrix:\n{cm}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 1. Baseline RandomForest
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train,
                        class_weight="balanced") -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print(f"[RF] Trained on {X_train.shape[0]} samples.")
    return rf


# ─────────────────────────────────────────────────────────────────────────────
# 2. Baseline GradientBoosting
# ─────────────────────────────────────────────────────────────────────────────

def train_gradient_boosting(X_train, y_train) -> GradientBoostingClassifier:
    gbm = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    gbm.fit(X_train, y_train)
    print(f"[GBM] Trained on {X_train.shape[0]} samples.")
    return gbm


# ─────────────────────────────────────────────────────────────────────────────
# 3. MLP Classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_mlp(X_train, y_train) -> MLPClassifier:
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )
    mlp.fit(X_train, y_train)
    print(f"[MLP] Trained on {X_train.shape[0]} samples. Iters={mlp.n_iter_}")
    return mlp


# ─────────────────────────────────────────────────────────────────────────────
# 4. RF + Focal-Loss sample weights
# ─────────────────────────────────────────────────────────────────────────────

def train_focal_rf(X_train, y_train,
                   gamma: float = 2.0) -> RandomForestClassifier:
    """
    Two-stage training:
      Stage 1 – fit a quick RF to get predicted probabilities.
      Stage 2 – compute focal-loss sample weights from Stage 1 probs.
      Stage 3 – re-fit RF using those weights.
    """
    # Stage 1: warm-up RF (no class weight)
    rf_warm = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_warm.fit(X_train, y_train)
    y_prob_warm = rf_warm.predict_proba(X_train)

    # Stage 2: focal weights
    alpha   = compute_alpha_weights(y_train)
    fl      = FocalLossNumpy(gamma=gamma, alpha=alpha)
    sw      = fl.compute_sample_weights(y_train, y_prob_warm)
    fl_val  = fl.loss(y_train, y_prob_warm)
    print(f"[FocalRF] Warm-up focal loss: {fl_val:.4f}")

    # Stage 3: final RF
    rf_final = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_final.fit(X_train, y_train, sample_weight=sw)
    print(f"[FocalRF] Trained on {X_train.shape[0]} samples (gamma={gamma}).")
    return rf_final


# ─────────────────────────────────────────────────────────────────────────────
# 5. GBM + Focal-Loss sample weights
# ─────────────────────────────────────────────────────────────────────────────

def train_focal_gbm(X_train, y_train,
                    gamma: float = 2.0) -> GradientBoostingClassifier:
    # Warm-up
    gbm_warm = GradientBoostingClassifier(
        n_estimators=50, random_state=RANDOM_STATE
    )
    gbm_warm.fit(X_train, y_train)
    y_prob_warm = gbm_warm.predict_proba(X_train)

    alpha  = compute_alpha_weights(y_train)
    fl     = FocalLossNumpy(gamma=gamma, alpha=alpha)
    sw     = fl.compute_sample_weights(y_train, y_prob_warm)

    gbm_final = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    gbm_final.fit(X_train, y_train, sample_weight=sw)
    print(f"[FocalGBM] Trained on {X_train.shape[0]} samples (gamma={gamma}).")
    return gbm_final


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import generate_bus_dataset, load_and_split

    generate_bus_dataset(save_dir="../data")
    X_train, X_val, X_test, y_train, y_val, y_test, _, _ = load_and_split("../data/bus_dataset.csv")

    rf  = train_random_forest(X_train, y_train)
    evaluate_model(rf, X_test, y_test, "RandomForest (baseline)")

    frf = train_focal_rf(X_train, y_train)
    evaluate_model(frf, X_test, y_test, "RandomForest + FocalLoss")