"""
focal_loss.py
-------------
Focal Loss for multi-class imbalanced classification.

FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

  - gamma > 0 reduces loss for easy (well-classified) examples
  - alpha  balances class weights for rare classes

Two implementations are provided:
  1. FocalLossNumpy  – NumPy / scikit-learn compatible (used for RF/GBM wrappers)
  2. FocalLossTorch  – PyTorch nn.Module (used in MLP / deep models)

Also provides compute_class_weights() for cross-entropy baseline comparison.
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────────────────────────
# 1. Class weight helper (used by both focal loss & cross-entropy)
# ─────────────────────────────────────────────────────────────────

def compute_alpha_weights(y_train: np.ndarray) -> np.ndarray:
    """
    Compute per-class alpha weights (inverse frequency).
    Returns array of shape [n_classes].
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced",
                                   classes=classes,
                                   y=y_train)
    alpha = weights / weights.sum()           # normalise to sum = 1
    print(f"[FocalLoss] Alpha weights: { {int(c): round(float(w),4) for c,w in zip(classes,alpha)} }")
    return alpha


# ─────────────────────────────────────────────────────────────────
# 2. NumPy Focal Loss (for custom sample_weight in sklearn)
# ─────────────────────────────────────────────────────────────────

class FocalLossNumpy:
    """
    Computes focal-loss-inspired sample weights that can be passed to
    sklearn estimators via `sample_weight`.

    Higher weight → model focuses on hard-to-classify samples.
    """

    def __init__(self, gamma: float = 2.0, alpha: np.ndarray = None):
        self.gamma = gamma
        self.alpha = alpha          # per-class alpha; None → equal

    def compute_sample_weights(self,
                                y_true: np.ndarray,
                                y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        y_true       : (N,)  true integer labels
        y_pred_proba : (N, C) predicted probabilities from a model

        Returns
        -------
        sample_weights : (N,) higher for harder / minority samples
        """
        n = len(y_true)
        # p_t = predicted probability of the true class
        p_t = y_pred_proba[np.arange(n), y_true]
        p_t = np.clip(p_t, 1e-7, 1.0)

        # alpha_t
        if self.alpha is not None:
            alpha_t = self.alpha[y_true]
        else:
            alpha_t = np.ones(n)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma

        # Normalise so weights sum to N (sklearn convention)
        focal_weight = focal_weight * n / focal_weight.sum()
        return focal_weight

    def loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Scalar focal loss (for logging)."""
        n   = len(y_true)
        p_t = y_pred_proba[np.arange(n), y_true]
        p_t = np.clip(p_t, 1e-7, 1.0)

        if self.alpha is not None:
            alpha_t = self.alpha[y_true]
        else:
            alpha_t = np.ones(n)

        fl = -alpha_t * (1.0 - p_t) ** self.gamma * np.log(p_t)
        return float(fl.mean())


# ─────────────────────────────────────────────────────────────────
# 3. PyTorch Focal Loss (nn.Module)
# ─────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FocalLossTorch(nn.Module):
        """
        Multi-class Focal Loss.
        Usage:
            criterion = FocalLossTorch(alpha=alpha_tensor, gamma=2.0)
            loss = criterion(logits, targets)
        """

        def __init__(self,
                     alpha: "torch.Tensor" = None,
                     gamma: float = 2.0,
                     reduction: str = "mean"):
            super().__init__()
            self.gamma     = gamma
            self.reduction = reduction
            if alpha is not None:
                self.register_buffer("alpha", alpha.float())
            else:
                self.alpha = None

        def forward(self,
                    inputs: "torch.Tensor",
                    targets: "torch.Tensor") -> "torch.Tensor":
            """
            inputs  : (N, C) raw logits
            targets : (N,)   integer class indices
            """
            log_prob = F.log_softmax(inputs, dim=1)            # (N, C)
            prob     = torch.exp(log_prob)                     # (N, C)

            # gather p_t and log_p_t for each true class
            log_p_t  = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
            p_t      = prob.gather(1, targets.unsqueeze(1)).squeeze(1)      # (N,)

            focal_w  = (1.0 - p_t) ** self.gamma

            if self.alpha is not None:
                alpha_t = self.alpha[targets]
                focal_w = alpha_t * focal_w

            loss = -focal_w * log_p_t

            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss

    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False
    print("[focal_loss] PyTorch not installed – FocalLossTorch unavailable.")


# ─────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, ".")
    from data_loader import generate_bus_dataset, load_and_split

    generate_bus_dataset(save_dir="../data")
    X_train, _, _, y_train, _, _, _, _ = load_and_split("../data/bus_dataset.csv")

    alpha = compute_alpha_weights(y_train)

    # Dummy probabilities
    rng    = np.random.default_rng(42)
    y_prob = rng.dirichlet(alpha=[1, 1, 1], size=len(y_train))

    fl     = FocalLossNumpy(gamma=2.0, alpha=alpha)
    sw     = fl.compute_sample_weights(y_train, y_prob)
    scalar = fl.loss(y_train, y_prob)

    print(f"\nSample weights (first 5): {sw[:5].round(4)}")
    print(f"Focal Loss value        : {scalar:.4f}")

    if TORCH_AVAILABLE:
        import torch
        logits  = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        alpha_t = torch.tensor(alpha, dtype=torch.float32)
        crit    = FocalLossTorch(alpha=alpha_t, gamma=2.0)
        print(f"\nTorch Focal Loss (dummy): {crit(logits, targets):.4f}")