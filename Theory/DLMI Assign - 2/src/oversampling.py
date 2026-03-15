"""
oversampling.py
---------------
Oversampling strategies for imbalanced BUS dataset:
  1. SMOTE            – Synthetic Minority Over-sampling Technique
  2. ADASYN           – Adaptive Synthetic Sampling
  3. BorderlineSMOTE  – Focuses on borderline samples
  4. RandomOverSampler – Baseline random duplication
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from collections import Counter

from imblearn.over_sampling import (
    SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
)

RANDOM_STATE = 42
CLASS_NAMES  = {0: "Benign", 1: "Malignant", 2: "Normal"}


def apply_smote(X_train, y_train):
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[SMOTE]          Before: {Counter(y_train)} → After: {Counter(y_res)}")
    return X_res, y_res


def apply_adasyn(X_train, y_train):
    try:
        ada = ADASYN(random_state=RANDOM_STATE, n_neighbors=5)
        X_res, y_res = ada.fit_resample(X_train, y_train)
        print(f"[ADASYN]         Before: {Counter(y_train)} → After: {Counter(y_res)}")
    except RuntimeError as e:
        print(f"[ADASYN] Fallback to SMOTE — ADASYN failed: {e}")
        X_res, y_res = apply_smote(X_train, y_train)
    return X_res, y_res


def apply_borderline_smote(X_train, y_train):
    bs = BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = bs.fit_resample(X_train, y_train)
    print(f"[BorderlineSMOTE] Before: {Counter(y_train)} → After: {Counter(y_res)}")
    return X_res, y_res


def apply_random_oversample(X_train, y_train):
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    print(f"[RandomOverSampler] Before: {Counter(y_train)} → After: {Counter(y_res)}")
    return X_res, y_res


def plot_class_distribution(y_before, y_after_dict, save_path: str):
    """
    Bar chart comparing class distribution before and after each oversampling method.
    """
    fig, axes = plt.subplots(1, len(y_after_dict) + 1,
                             figsize=(5 * (len(y_after_dict) + 1), 5))
    fig.suptitle("Class Distribution: Before vs After Oversampling", fontsize=14, y=1.02)

    def _bar(ax, counter, title):
        labels = [CLASS_NAMES[k] for k in sorted(counter.keys())]
        values = [counter[k] for k in sorted(counter.keys())]
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Sample Count")
        ax.set_ylim(0, max(values) * 1.25)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(val), ha="center", va="bottom", fontsize=9)

    _bar(axes[0], Counter(y_before), "Original")

    for ax, (name, y_after) in zip(axes[1:], y_after_dict.items()):
        _bar(ax, Counter(y_after), name)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[oversampling] Distribution plot saved → {save_path}")


def get_best_oversampled(X_train, y_train, method: str = "smote"):
    """
    Returns oversampled data using the specified method.
    Default = SMOTE (best general-purpose).
    """
    methods = {
        "smote":     apply_smote,
        "adasyn":    apply_adasyn,
        "borderline": apply_borderline_smote,
        "random":    apply_random_oversample,
    }
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(methods.keys())}")
    return methods[method](X_train, y_train)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import generate_bus_dataset, load_and_split

    generate_bus_dataset(save_dir="../data")
    X_train, X_val, X_test, y_train, y_val, y_test, _, _ = load_and_split("../data/bus_dataset.csv")

    print("\n--- Oversampling Comparison ---")
    _, y_smote      = apply_smote(X_train, y_train)
    _, y_adasyn     = apply_adasyn(X_train, y_train)
    _, y_border     = apply_borderline_smote(X_train, y_train)
    _, y_random     = apply_random_oversample(X_train, y_train)

    plot_class_distribution(
        y_train,
        {"SMOTE": y_smote, "ADASYN": y_adasyn, "BorderlineSMOTE": y_border, "RandomOS": y_random},
        save_path="../results/oversampling_distribution.png"
    )