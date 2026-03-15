"""
data_loader.py
--------------
Synthetic BUS (Breast UltraSound) dataset generator + loader.
Mimics the real BUS dataset class distribution:
  - benign   : majority class
  - malignant: minority class
  - normal   : minority class
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# 1.  Generate synthetic BUS-like feature dataset
# ---------------------------------------------------------------------------
def generate_bus_dataset(n_samples: int = 2000, save_dir: str = "../data") -> pd.DataFrame:
    """
    Generate a synthetic tabular BUS dataset.
    Features represent:
        - texture, contrast, energy, homogeneity (GLCM)
        - mean_intensity, std_intensity
        - area, perimeter, circularity  (shape)
        - BI-RADS-like score
    Classes: 0=benign (60%), 1=malignant (25%), 2=normal (15%)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Class proportions
    n_benign    = int(n_samples * 0.60)
    n_malignant = int(n_samples * 0.25)
    n_normal    = n_samples - n_benign - n_malignant

    def make_class(n, means, stds):
        return np.column_stack([
            np.random.normal(m, s, n) for m, s in zip(means, stds)
        ])

    feature_names = [
        "texture", "contrast", "energy", "homogeneity",
        "mean_intensity", "std_intensity",
        "area", "perimeter", "circularity", "birads_score"
    ]

    # Benign: smooth, well-defined
    benign = make_class(n_benign,
        means=[25, 0.30, 0.65, 0.85, 120, 18, 850, 110, 0.78, 2.1],
        stds= [ 5, 0.08, 0.10, 0.07,  20,  5, 200,  25, 0.08, 0.5])

    # Malignant: irregular, heterogeneous
    malignant = make_class(n_malignant,
        means=[45, 0.70, 0.30, 0.50,  90, 32, 600,  95, 0.52, 4.2],
        stds= [ 8, 0.12, 0.08, 0.10,  25,  8, 250,  30, 0.12, 0.6])

    # Normal: very uniform
    normal = make_class(n_normal,
        means=[15, 0.15, 0.80, 0.92, 140, 10, 950, 120, 0.88, 1.0],
        stds= [ 4, 0.05, 0.07, 0.04,  15,  3, 150,  18, 0.05, 0.3])

    X = np.vstack([benign, malignant, normal])
    y = np.array([0]*n_benign + [1]*n_malignant + [2]*n_normal)

    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y

    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    csv_path = os.path.join(save_dir, "bus_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"[data_loader] Dataset saved → {csv_path}")
    print(f"[data_loader] Class distribution:\n{df['label'].value_counts().rename({0:'benign',1:'malignant',2:'normal'})}\n")
    return df


# ---------------------------------------------------------------------------
# 2.  Load & split
# ---------------------------------------------------------------------------
def load_and_split(csv_path: str = "../data/bus_dataset.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    feature_names = df.drop("label", axis=1).columns.tolist()
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names


if __name__ == "__main__":
    df = generate_bus_dataset()
    splits = load_and_split()
    X_train, X_val, X_test, y_train, y_val, y_test, _, _ = splits
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")