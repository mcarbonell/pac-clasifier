"""
Unit tests for benchmark baselines (GLVQ, KNN, SVM, MLP) and metrics logger.
"""

import pytest
import torch
from experiments.baselines import GLVQClassifier, KNNBaseline, SVMBaseline, MLPBaseline
from experiments.metrics_logger import BenchmarkTracker, aggregate_and_save_summary
from experiments.datasets import load_tabular_digits


def test_glvq_fit_predict(synthetic_separable_2d):
    """GLVQ baseline should train and predict correctly on synthetic data."""
    x, y = synthetic_separable_2d
    clf = GLVQClassifier(prototypes_per_class=1, lr=0.05, epochs=5, batch_size=16, device='cpu')
    clf.fit(x, y, verbose=False)
    
    assert clf.is_fitted
    preds, dists = clf.predict(x)
    assert preds.shape == y.shape
    acc = (preds == y).float().mean().item()
    assert acc > 0.85


def test_classical_baselines(synthetic_separable_2d):
    """Classical baselines (KNN, SVM, MLP) should fit and predict uniformly."""
    x, y = synthetic_separable_2d
    
    knn = KNNBaseline(k=1)
    knn.fit(x, y)
    preds, _ = knn.predict(x)
    assert (preds == y).float().mean().item() == 1.0
    
    svm = SVMBaseline(c=1.0)
    svm.fit(x, y)
    preds, _ = svm.predict(x)
    assert (preds == y).float().mean().item() == 1.0
    
    mlp = MLPBaseline(hidden_dim=32, max_iter=10)
    mlp.fit(x, y)
    preds, _ = mlp.predict(x)
    assert preds.shape == y.shape


def test_metrics_tracker_and_summary(tmp_path):
    """Metrics tracker should properly record and aggregate metrics."""
    raw_dir = str(tmp_path / "raw")
    sum_dir = str(tmp_path / "sum")
    
    raw_results = []
    for seed in [1, 2]:
        tracker = BenchmarkTracker("test_exp", {"lr": 0.01}, seed)
        tracker.start()
        tracker.record_step(iteration=1, eval_time=0.01, evaluations=10, current_acc=0.95)
        res = tracker.finish(final_objective=0.95)
        tracker.save_raw(output_dir=raw_dir, result_data=res)
        raw_results.append(res)
        
    summary = aggregate_and_save_summary("test_exp", raw_results, output_dir=sum_dir)
    assert summary["num_seeds"] == 2
    assert summary["metrics"]["final_objective"]["mean"] == 0.95
    assert "wall_clock_time" in summary["metrics"]
    assert "internal_overhead_time" in summary["metrics"]
