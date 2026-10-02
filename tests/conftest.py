"""
Pytest configuration and shared fixtures for PAC Classifier test suite.
"""

import pytest
import torch


@pytest.fixture
def synthetic_separable_2d():
    """
    Creates a simple 2D synthetic dataset with 3 clearly separable clusters.
    """
    torch.manual_seed(42)
    c1 = torch.randn(30, 2) + torch.tensor([5.0, 5.0])
    c2 = torch.randn(30, 2) + torch.tensor([-5.0, 5.0])
    c3 = torch.randn(30, 2) + torch.tensor([0.0, -5.0])
    
    x = torch.cat([c1, c2, c3], dim=0)
    y = torch.cat([
        torch.zeros(30, dtype=torch.long),
        torch.ones(30, dtype=torch.long),
        torch.full((30,), 2, dtype=torch.long)
    ])
    return x, y


@pytest.fixture
def synthetic_noisy_multiclass():
    """
    Creates synthetic multi-class data with intentional overlap/noise to trigger error bifurcation.
    """
    torch.manual_seed(123)
    n_per_class = 40
    dim = 8
    
    x_list = []
    y_list = []
    for cls in range(4):
        center = torch.zeros(dim)
        center[cls % dim] = 2.0
        samples = torch.randn(n_per_class, dim) * 0.8 + center
        x_list.append(samples)
        y_list.append(torch.full((n_per_class,), cls, dtype=torch.long))
        
    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return x, y
