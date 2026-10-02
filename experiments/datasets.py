"""
Dataset Loaders for PAC Benchmarks.
Supports MNIST, Fashion-MNIST, and Tabular datasets.
"""

import os
import torch
import numpy as np
from typing import Tuple, Optional


def load_mnist(data_dir: str = "data", subsample: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads MNIST dataset (train: 60,000, test: 10,000).
    Returns normalized (x_train, y_train, x_test, y_test).
    """
    from torchvision import datasets, transforms
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.view(-1))  # Flatten to 784D
    ])
    
    train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    x_train = train_set.data.float().view(-1, 784) / 255.0
    y_train = train_set.targets.long()
    
    x_test = test_set.data.float().view(-1, 784) / 255.0
    y_test = test_set.targets.long()
    
    if subsample is not None and subsample < len(x_train):
        perm_train = torch.randperm(len(x_train))[:subsample]
        x_train, y_train = x_train[perm_train], y_train[perm_train]
        test_sub = min(len(x_test), max(200, subsample // 3))
        perm_test = torch.randperm(len(x_test))[:test_sub]
        x_test, y_test = x_test[perm_test], y_test[perm_test]
        
    return x_train, y_train, x_test, y_test


def load_fashion_mnist(data_dir: str = "data", subsample: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads Fashion-MNIST dataset (train: 60,000, test: 10,000).
    Returns normalized (x_train, y_train, x_test, y_test).
    """
    from torchvision import datasets, transforms
    os.makedirs(data_dir, exist_ok=True)
    
    train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True)
    test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True)
    
    x_train = train_set.data.float().view(-1, 784) / 255.0
    y_train = train_set.targets.long()
    
    x_test = test_set.data.float().view(-1, 784) / 255.0
    y_test = test_set.targets.long()
    
    if subsample is not None and subsample < len(x_train):
        perm_train = torch.randperm(len(x_train))[:subsample]
        x_train, y_train = x_train[perm_train], y_train[perm_train]
        test_sub = min(len(x_test), max(200, subsample // 3))
        perm_test = torch.randperm(len(x_test))[:test_sub]
        x_test, y_test = x_test[perm_test], y_test[perm_test]
        
    return x_train, y_train, x_test, y_test


def load_tabular_digits(subsample: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads standard 64-feature Optical Recognition of Handwritten Digits tabular dataset.
    Train: 80%, Test: 20%.
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    
    digits = load_digits()
    x = digits.data.astype(np.float32) / 16.0  # Normalize to [0, 1]
    y = digits.target.astype(np.int64)
    
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    
    x_train = torch.tensor(x_tr, dtype=torch.float32)
    y_train = torch.tensor(y_tr, dtype=torch.long)
    x_test = torch.tensor(x_te, dtype=torch.float32)
    y_test = torch.tensor(y_te, dtype=torch.long)
    
    if subsample is not None and subsample < len(x_train):
        x_train = x_train[:subsample]
        y_train = y_train[:subsample]
        
    return x_train, y_train, x_test, y_test


def get_dataset(name: str = 'mnist', data_dir: str = 'data', quick: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unified dataset getter.
    name: 'mnist', 'fashion', or 'tabular'
    quick: if True, returns a small subsample for rapid validation.
    """
    sub = 500 if quick else None
    name = name.lower()
    
    if name == 'mnist':
        return load_mnist(data_dir=data_dir, subsample=sub)
    elif name in ('fashion', 'fashion_mnist', 'fashion-mnist'):
        return load_fashion_mnist(data_dir=data_dir, subsample=sub)
    elif name in ('tabular', 'digits'):
        return load_tabular_digits(subsample=sub)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Supported: 'mnist', 'fashion', 'tabular'")
