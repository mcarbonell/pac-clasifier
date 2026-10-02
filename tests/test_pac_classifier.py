"""
Unit tests for PurifyingArchetypeClassifier (PAC).
"""

import pytest
import torch
from pac import PurifyingArchetypeClassifier, PAC
from pac_v2 import PurifyingArchetypeClassifierV2
from pac.morphing import morph_archetype, optimize_morph_fit


def test_unfitted_model_raises():
    """Predicting or querying archetypes before fitting should raise ValueError."""
    clf = PurifyingArchetypeClassifier()
    x = torch.randn(10, 5)
    with pytest.raises(ValueError):
        clf.predict(x)
    with pytest.raises(ValueError):
        clf.get_archetypes()
    with pytest.raises(ValueError):
        clf.get_archetype_info()


def test_fit_predict_synthetic_separable(synthetic_separable_2d):
    """Separable 2D clusters should achieve 100% accuracy quickly."""
    x, y = synthetic_separable_2d
    clf = PurifyingArchetypeClassifier(max_iters=10, target_acc=0.999, device='cpu')
    clf.fit(x, y, verbose=False)
    
    assert clf.is_fitted
    preds, scores = clf.predict(x)
    acc = (preds == y).float().mean().item()
    assert acc == 1.0
    assert preds.shape == y.shape
    assert scores.shape == (x.shape[0],)


@pytest.mark.parametrize("dim", [1, 7, 33, 100])
def test_arbitrary_dimensions(dim):
    """PAC should accept any arbitrary input feature dimension D."""
    torch.manual_seed(99)
    x = torch.randn(50, dim)
    y = torch.randint(0, 3, (50,))
    
    clf = PurifyingArchetypeClassifier(max_iters=5, device='cpu')
    clf.fit(x, y, verbose=False)
    
    arch_tensors, arch_labels = clf.get_archetypes()
    assert arch_tensors.shape[1] == dim
    preds, _ = clf.predict(x)
    assert preds.shape == (50,)


def test_archetype_monotonicity(synthetic_noisy_multiclass):
    """Archetype count should be monotonically non-decreasing over iterations."""
    x, y = synthetic_noisy_multiclass
    clf = PurifyingArchetypeClassifier(max_iters=15, target_acc=1.0, device='cpu')
    clf.fit(x, y, verbose=False)
    
    counts = [stat['num_archetypes'] for stat in clf.generation_stats]
    assert len(counts) > 1
    # Check non-decreasing
    for i in range(1, len(counts)):
        assert counts[i] >= counts[i - 1], f"Archetype count decreased at gen {i}: {counts[i]} < {counts[i-1]}"


def test_confusion_mapping_consistency(synthetic_noisy_multiclass):
    """Confusion mapping should never pair a class with itself (true != pred)."""
    x, y = synthetic_noisy_multiclass
    clf = PurifyingArchetypeClassifier(max_iters=10, bifurcation_mode='confusion', device='cpu')
    clf.fit(x, y, verbose=False)
    
    for cid, (true_l, pred_l) in clf.cluster_confusion_map.items():
        assert true_l != pred_l, f"Cluster {cid} confused class {true_l} with itself!"
        info = clf.get_archetype_info(cid)
        assert info['label'] == true_l
        assert info['confusion'] == (true_l, pred_l)


def test_persistent_errors_tracking(synthetic_noisy_multiclass):
    """Persistent errors diagnostics should correctly track samples never classified correctly."""
    x, y = synthetic_noisy_multiclass
    clf = PurifyingArchetypeClassifier(max_iters=5, device='cpu')
    clf.fit(x, y, verbose=False)
    
    assert clf.never_correct_mask is not None
    assert clf.never_correct_mask.shape == (x.shape[0],)
    assert clf.persistent_error_indices is not None
    if len(clf.persistent_error_indices) > 0:
        assert torch.all(clf.persistent_error_indices >= 0)
        assert torch.all(clf.persistent_error_indices < x.shape[0])


def test_retrocompatibility_v2(synthetic_noisy_multiclass):
    """PurifyingArchetypeClassifierV2 from pac_v2 should work as a drop-in alias."""
    x, y = synthetic_noisy_multiclass
    clf_v2 = PurifyingArchetypeClassifierV2(max_iters=5, device='cpu')
    clf_v2.fit(x, y, verbose=False)
    
    assert clf_v2.is_fitted
    preds, _ = clf_v2.predict(x)
    assert preds.shape == y.shape
    assert clf_v2.bifurcation_mode == 'confusion'


@pytest.mark.parametrize("metric", ['cosine', 'euclidean'])
def test_distance_metrics(synthetic_separable_2d, metric):
    """Both cosine similarity and euclidean distance metrics should be supported."""
    x, y = synthetic_separable_2d
    clf = PurifyingArchetypeClassifier(max_iters=10, distance_metric=metric, device='cpu')
    clf.fit(x, y, verbose=False)
    
    preds, scores = clf.predict(x)
    acc = (preds == y).float().mean().item()
    assert acc > 0.90


def test_confidence_bifurcation_mode(synthetic_noisy_multiclass):
    """Legacy confidence bifurcation mode should operate without errors."""
    x, y = synthetic_noisy_multiclass
    clf = PurifyingArchetypeClassifier(max_iters=5, bifurcation_mode='confidence', device='cpu')
    clf.fit(x, y, verbose=False)
    
    assert clf.is_fitted
    preds, _ = clf.predict(x)
    assert preds.shape == y.shape


def test_morphing_affine_transform():
    """Spatial morphing should properly transform 2D flattened inputs."""
    torch.manual_seed(0)
    # Create simple 4x4 image
    img = torch.zeros(16)
    img[5] = 1.0  # single active pixel
    
    # Translation
    tx = torch.tensor(0.2)
    ty = torch.tensor(0.0)
    theta = torch.tensor(0.0)
    scale = torch.tensor(1.0)
    
    morphed = morph_archetype(img, tx, ty, theta, scale, image_shape=(4, 4))
    assert morphed.shape == (16,)
    assert not torch.isnan(morphed).any()
    
    # Test optimizer
    target = morphed.clone().detach()
    fit_sim = optimize_morph_fit(target, img, image_shape=(4, 4), iterations=5, device=torch.device('cpu'))
    assert fit_sim > 0.0


def test_print_confusion_archetypes_and_info(synthetic_noisy_multiclass, capsys):
    """print_confusion_archetypes and get_archetype_info should output properly."""
    x, y = synthetic_noisy_multiclass
    clf = PurifyingArchetypeClassifier(max_iters=5, device='cpu')
    clf.fit(x, y, verbose=False)
    
    # Test get_archetype_info
    all_info = clf.get_archetype_info()
    assert isinstance(all_info, dict)
    
    # Test print_confusion_archetypes
    clf.print_confusion_archetypes()
    captured = capsys.readouterr()
    assert "PAC CONFUSION ARCHETYPES SUMMARY" in captured.out


def test_predict_with_morphing(synthetic_noisy_multiclass):
    """predict_with_morphing should evaluate and predict labels."""
    torch.manual_seed(10)
    # Create 4x4 data
    x = torch.randn(12, 16)
    y = torch.randint(0, 2, (12,))
    clf = PurifyingArchetypeClassifier(max_iters=3, device='cpu')
    clf.fit(x, y, verbose=False)
    
    preds, sims = clf.predict_with_morphing(x[:3], image_shape=(4, 4), top_k=2, morph_iters=2)
    assert preds.shape == (3,)
    assert sims.shape == (3,)

