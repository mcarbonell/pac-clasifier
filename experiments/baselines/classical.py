"""
Classical ML Baselines (KNN, SVM, MLP) for PAC Comparative Evaluation.
Uses standard scikit-learn implementations with consistent interfaces.
"""

import time
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from typing import Tuple, Optional


def to_numpy(t) -> np.ndarray:
    """Converts torch tensor or array to numpy 2D array."""
    if isinstance(t, torch.Tensor):
        if t.dim() > 2:
            t = t.reshape(t.size(0), -1)
        return t.detach().cpu().numpy()
    arr = np.asarray(t)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


class KNNBaseline:
    """k-Nearest Neighbors baseline classifier."""
    def __init__(self, k: int = 3, metric: str = 'cosine'):
        self.k = k
        self.metric = metric
        self.clf = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm='auto')
        self.is_fitted = False

    def fit(self, x_train, y_train, verbose: bool = False):
        x = to_numpy(x_train)
        y = to_numpy(y_train).ravel()
        if verbose:
            print(f"Fitting KNN (k={self.k}, metric={self.metric}) on {x.shape[0]} samples...")
        self.clf.fit(x, y)
        self.is_fitted = True
        return self

    def predict(self, x_test) -> Tuple[torch.Tensor, torch.Tensor]:
        x = to_numpy(x_test)
        preds = self.clf.predict(x)
        # Placeholder score: 1.0
        scores = np.ones(len(preds), dtype=np.float32)
        return torch.tensor(preds, dtype=torch.long), torch.tensor(scores, dtype=torch.float32)


class SVMBaseline:
    """Support Vector Machine with RBF kernel."""
    def __init__(self, c: float = 10.0, kernel: str = 'rbf'):
        self.c = c
        self.kernel = kernel
        self.clf = SVC(C=c, kernel=kernel, random_state=42)
        self.is_fitted = False

    def fit(self, x_train, y_train, verbose: bool = False):
        x = to_numpy(x_train)
        y = to_numpy(y_train).ravel()
        if verbose:
            print(f"Fitting SVM (kernel={self.kernel}, C={self.c}) on {x.shape[0]} samples...")
        self.clf.fit(x, y)
        self.is_fitted = True
        return self

    def predict(self, x_test) -> Tuple[torch.Tensor, torch.Tensor]:
        x = to_numpy(x_test)
        preds = self.clf.predict(x)
        scores = np.ones(len(preds), dtype=np.float32)
        return torch.tensor(preds, dtype=torch.long), torch.tensor(scores, dtype=torch.float32)


class MLPBaseline:
    """1-Hidden Layer Multi-Layer Perceptron (e.g. 784 -> 256 -> 10)."""
    def __init__(self, hidden_dim: int = 256, max_iter: int = 25, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.seed = seed
        self.clf = MLPClassifier(
            hidden_layer_sizes=(hidden_dim,),
            max_iter=max_iter,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=5,
            verbose=False
        )
        self.is_fitted = False

    def fit(self, x_train, y_train, verbose: bool = False):
        x = to_numpy(x_train)
        y = to_numpy(y_train).ravel()
        if verbose:
            print(f"Fitting MLP (hidden={self.hidden_dim}, max_iter={self.max_iter}) on {x.shape[0]} samples...")
        self.clf.fit(x, y)
        self.is_fitted = True
        return self

    def predict(self, x_test) -> Tuple[torch.Tensor, torch.Tensor]:
        x = to_numpy(x_test)
        preds = self.clf.predict(x)
        probs = self.clf.predict_proba(x)
        max_prob = np.max(probs, axis=1)
        return torch.tensor(preds, dtype=torch.long), torch.tensor(max_prob, dtype=torch.float32)
