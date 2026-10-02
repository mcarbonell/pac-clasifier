"""
Generalized Learning Vector Quantization (GLVQ) Baseline in PyTorch.
Implements Sato & Yamada (1996) "Generalized Learning Vector Quantization".

This provides a direct, rigorous baseline comparing PAC against the canonical
gradient-based prototype classifier family (LVQ).
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GLVQClassifier(nn.Module):
    """
    Generalized Learning Vector Quantization (GLVQ) Classifier.
    
    Parameters
    ----------
    prototypes_per_class : int, default=1
        Number of prototype vectors assigned to each class.
    lr : float, default=0.01
        Learning rate for prototype position updates.
    epochs : int, default=30
        Number of training epochs.
    batch_size : int, default=256
        Mini-batch size for stochastic gradient descent.
    device : str or torch.device, optional
        Computation device.
    """
    def __init__(
        self,
        prototypes_per_class: int = 1,
        lr: float = 0.01,
        epochs: int = 30,
        batch_size: int = 256,
        device: Optional[str] = None
    ):
        super().__init__()
        self.prototypes_per_class = prototypes_per_class
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.prototypes: Optional[nn.Parameter] = None
        self.prototype_labels: Optional[torch.Tensor] = None
        self.num_classes: int = 0
        self.feature_dim: int = 0
        self.is_fitted: bool = False

    def _init_prototypes(self, x: torch.Tensor, y: torch.Tensor):
        """Initializes prototypes near class centroids with small Gaussian perturbation."""
        unique_classes = torch.unique(y)
        self.num_classes = len(unique_classes)
        self.feature_dim = x.shape[1]
        
        proto_list = []
        proto_labels_list = []
        
        for c in unique_classes:
            mask = (y == c)
            class_samples = x[mask]
            centroid = class_samples.mean(dim=0)
            
            for _ in range(self.prototypes_per_class):
                # Small jitter around centroid if multiple prototypes per class
                jitter = torch.randn_like(centroid) * 0.01 if self.prototypes_per_class > 1 else 0.0
                proto_list.append(centroid + jitter)
                proto_labels_list.append(c.item())
                
        proto_tensor = torch.stack(proto_list).to(self.device)
        self.prototypes = nn.Parameter(proto_tensor)
        self.prototype_labels = torch.tensor(proto_labels_list, dtype=torch.long, device=self.device)

    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor, verbose: bool = True):
        """
        Trains GLVQ using mini-batch gradient descent on the Sato & Yamada relative distance loss.
        """
        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)
        if x_train.dim() > 2:
            x_train = x_train.reshape(x_train.size(0), -1)
            
        x_train = x_train.float().to(self.device)
        y_train = y_train.long().to(self.device)
        
        N = x_train.size(0)
        self._init_prototypes(x_train, y_train)
        
        optimizer = torch.optim.Adam([self.prototypes], lr=self.lr)
        
        indices = torch.arange(N, device=self.device)
        
        for epoch in range(self.epochs):
            # Shuffle indices each epoch
            perm = indices[torch.randperm(N, device=self.device)]
            total_loss = 0.0
            num_batches = 0
            
            for start_idx in range(0, N, self.batch_size):
                batch_idx = perm[start_idx:start_idx + self.batch_size]
                xb = x_train[batch_idx]  # (B, D)
                yb = y_train[batch_idx]  # (B,)
                B = xb.size(0)
                
                # Pairwise squared Euclidean distances: (B, P)
                # ||xb - w||^2 = ||xb||^2 - 2 xb w^T + ||w||^2
                dists = torch.cdist(xb, self.prototypes, p=2.0) ** 2  # (B, P)
                
                # Create mask for same-class prototypes vs different-class prototypes
                # proto_labels: (P,) -> broadcast to (B, P)
                same_class_mask = (self.prototype_labels.unsqueeze(0) == yb.unsqueeze(1))  # (B, P)
                diff_class_mask = ~same_class_mask  # (B, P)
                
                # Distance to closest correct prototype d_J
                d_same = dists.clone()
                d_same[~same_class_mask] = float('inf')
                d_J, _ = torch.min(d_same, dim=1)  # (B,)
                
                # Distance to closest incorrect prototype d_K
                d_diff = dists.clone()
                d_diff[~diff_class_mask] = float('inf')
                d_K, _ = torch.min(d_diff, dim=1)  # (B,)
                
                # Relative distance ratio mu = (d_J - d_K) / (d_J + d_K)
                eps = 1e-7
                mu = (d_J - d_K) / (d_J + d_K + eps)  # (B,)
                
                # GLVQ sigmoid loss
                loss = torch.sigmoid(mu).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Fast Feedback: print in early batches of epoch 0
                if epoch == 0 and num_batches <= 5 and verbose:
                    print(f"GLVQ Init | Batch {num_batches:2d} | Loss: {loss.item():.4f}")
                    
            if verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                avg_loss = total_loss / max(1, num_batches)
                with torch.no_grad():
                    train_preds, _ = self.predict(x_train)
                    acc = (train_preds == y_train).float().mean().item()
                print(f"GLVQ Epoch {epoch:3d}/{self.epochs} | Loss: {avg_loss:.4f} | Train Acc: {acc*100:.2f}%")
                
        self.is_fitted = True
        return self

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classifies samples to the nearest prototype.
        Returns:
            predicted_labels: (N,) tensor
            min_distances: (N,) tensor
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
            
        x = x.float().to(self.device)
        dists = torch.cdist(x, self.prototypes, p=2.0)  # (N, P)
        min_dist, nearest_idx = torch.min(dists, dim=1)
        preds = self.prototype_labels[nearest_idx]
        return preds, min_dist
