"""
augmentation.py
---------------
Tabular data augmentation for BUS feature vectors:
  1. GaussianNoise    – adds small Gaussian noise to minority class samples
  2. FeatureMixup    – convex interpolation between two minority samples
  3. SMOTE-based      – handled in oversampling.py; referenced here for completeness

For image-based pipelines (when raw ultrasound images are available),
an image_augment() function is also provided using OpenCV-compatible transforms.
"""

import numpy as np
from collections import Counter

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# 1. Gaussian Noise Augmentation
# ---------------------------------------------------------------------------
class GaussianNoiseAugmenter:
    """
    Augment minority class samples by adding Gaussian noise.
    Useful when the minority class has very few genuine samples.
    """

    def __init__(self, noise_std: float = 0.05, augment_factor: int = 2):
        self.noise_std     = noise_std
        self.augment_factor = augment_factor

    def fit_resample(self, X, y, minority_classes: list = None):
        if minority_classes is None:
            counts = Counter(y)
            max_count = max(counts.values())
            minority_classes = [c for c, n in counts.items() if n < max_count]

        X_aug, y_aug = list(X), list(y)

        for cls in minority_classes:
            idx = np.where(y == cls)[0]
            samples = X[idx]

            for _ in range(self.augment_factor):
                noise   = np.random.normal(0, self.noise_std, samples.shape)
                X_new   = samples + noise
                X_aug.extend(X_new)
                y_aug.extend([cls] * len(X_new))

        X_aug = np.array(X_aug)
        y_aug = np.array(y_aug)

        # Shuffle
        perm  = np.random.permutation(len(y_aug))
        print(f"[GaussianNoise] Before: {Counter(y)} → After: {Counter(y_aug)}")
        return X_aug[perm], y_aug[perm]


# ---------------------------------------------------------------------------
# 2. Feature Mixup
# ---------------------------------------------------------------------------
class MixupAugmenter:
    """
    Mixup: interpolate between two samples of the same minority class.
    lambda ~ Beta(alpha, alpha); new_sample = lambda*x1 + (1-lambda)*x2
    """

    def __init__(self, alpha: float = 0.4, augment_factor: int = 2):
        self.alpha          = alpha
        self.augment_factor = augment_factor

    def fit_resample(self, X, y, minority_classes: list = None):
        if minority_classes is None:
            counts = Counter(y)
            max_count = max(counts.values())
            minority_classes = [c for c, n in counts.items() if n < max_count]

        X_aug, y_aug = list(X), list(y)

        for cls in minority_classes:
            idx     = np.where(y == cls)[0]
            samples = X[idx]
            n       = len(samples)

            for _ in range(self.augment_factor * n):
                i1, i2 = np.random.choice(n, 2, replace=False)
                lam     = np.random.beta(self.alpha, self.alpha)
                x_new   = lam * samples[i1] + (1 - lam) * samples[i2]
                X_aug.append(x_new)
                y_aug.append(cls)

        X_aug = np.array(X_aug)
        y_aug = np.array(y_aug)
        perm  = np.random.permutation(len(y_aug))
        print(f"[Mixup]         Before: {Counter(y)} → After: {Counter(y_aug)}")
        return X_aug[perm], y_aug[perm]


# ---------------------------------------------------------------------------
# 3. Combined pipeline helper
# ---------------------------------------------------------------------------
def augment_pipeline(X_train, y_train, method: str = "gaussian"):
    """
    Convenience wrapper.
    method: 'gaussian' | 'mixup'
    """
    if method == "gaussian":
        aug = GaussianNoiseAugmenter(noise_std=0.05, augment_factor=2)
    elif method == "mixup":
        aug = MixupAugmenter(alpha=0.4, augment_factor=2)
    else:
        raise ValueError(f"Unknown augmentation method '{method}'")

    return aug.fit_resample(X_train, y_train)


# ---------------------------------------------------------------------------
# 4. Image augmentation reference (for raw ultrasound images)
# ---------------------------------------------------------------------------
IMAGE_AUGMENTATION_CONFIG = {
    "description": "Albumentations / torchvision transforms for raw US images",
    "transforms": [
        "RandomHorizontalFlip(p=0.5)",
        "RandomRotation(degrees=(-15, 15))",
        "RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)",
        "GaussianBlur(blur_limit=(3,7), p=0.3)",
        "ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03, p=0.3)",
        "GridDistortion(p=0.2)",
        "Normalize(mean=[0.485], std=[0.229])",  # single channel US
    ],
    "note": (
        "Apply only to training set. "
        "ElasticTransform + GridDistortion simulate probe pressure variation. "
        "Use albumentations library: pip install albumentations"
    )
}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import generate_bus_dataset, load_and_split

    generate_bus_dataset(save_dir="../data")
    X_train, _, _, y_train, _, _, _, _ = load_and_split("../data/bus_dataset.csv")

    print("\n--- Augmentation Comparison ---")
    X_g, y_g = augment_pipeline(X_train, y_train, method="gaussian")
    X_m, y_m = augment_pipeline(X_train, y_train, method="mixup")
    print(f"\nGaussian shapes : X={X_g.shape}, y={y_g.shape}")
    print(f"Mixup shapes    : X={X_m.shape}, y={y_m.shape}")