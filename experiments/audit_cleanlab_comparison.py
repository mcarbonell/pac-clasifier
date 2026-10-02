"""
Direct One-by-One Item Comparison: PAC Persistent Errors vs Cleanlab Label Issues.
Evaluates cross-method convergence on the full 60,000 MNIST training set.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cleanlab

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.datasets import load_mnist


class FastMNISTNet(nn.Module):
    """Lightweight 2-layer neural network for fast out-of-fold probability estimation."""
    def __init__(self, in_features=784, hidden=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def compute_out_of_fold_probabilities(x_train: torch.Tensor, y_train: torch.Tensor, n_splits=3, epochs=4, batch_size=256, device="cpu"):
    """
    Computes cross-validated out-of-fold predicted probabilities for Cleanlab.
    """
    dev = torch.device(device)
    N = len(y_train)
    oof_probs = np.zeros((N, 10), dtype=np.float32)
    
    # Stratified K-Fold partition indices
    indices = np.arange(N)
    np.random.seed(42)
    shuffled_idx = np.random.permutation(indices)
    folds = np.array_split(shuffled_idx, n_splits)
    
    print(f"[CLEANLAB] Computing {n_splits}-fold out-of-sample predicted probabilities for {N} images...")
    
    for fold_i in range(n_splits):
        t0 = time.time()
        val_idx = folds[fold_i]
        train_idx = np.setdiff1d(indices, val_idx)
        
        x_tr, y_tr = x_train[train_idx].to(dev), y_train[train_idx].to(dev)
        x_val, y_val = x_train[val_idx].to(dev), y_train[val_idx].to(dev)
        
        model = FastMNISTNet().to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        criterion = nn.CrossEntropyLoss()
        
        # Train fold model
        model.train()
        n_tr = len(x_tr)
        for ep in range(epochs):
            perm = torch.randperm(n_tr)
            for b in range(0, n_tr, batch_size):
                b_idx = perm[b:b+batch_size]
                optimizer.zero_grad()
                logits = model(x_tr[b_idx])
                loss = criterion(logits, y_tr[b_idx])
                loss.backward()
                optimizer.step()
                
        # Evaluate validation out-of-fold probabilities
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_probs = F.softmax(val_logits, dim=1).cpu().numpy()
            
        oof_probs[val_idx] = val_probs
        fold_acc = (np.argmax(val_probs, axis=1) == y_train[val_idx].numpy()).mean()
        print(f"   Fold {fold_i+1}/{n_splits} finished in {time.time()-t0:.2f}s | Val Acc: {fold_acc*100:.2f}%")
        
    return oof_probs


def run_cleanlab_comparison():
    print("=" * 80)
    print("ITEM-BY-ITEM AUDIT COMPARISON: PAC PERSISTENT ERRORS vs CLEANLAB (MNIST 60K)")
    print("=" * 80)
    
    # 1. Load MNIST training set
    x_train, y_train, _, _ = load_mnist(data_dir="data")
    N = len(y_train)
    y_np = y_train.numpy()
    
    # 2. Load PAC persistent errors
    pac_json_path = "results/mnist_audit/persistent_errors.json"
    if not os.path.exists(pac_json_path):
        raise FileNotFoundError(f"Missing {pac_json_path}. Run examples/audit_mnist_errors.py first.")
        
    with open(pac_json_path, "r", encoding="utf-8") as f:
        pac_data = json.load(f)
        
    pac_errors = pac_data["errors"]
    pac_indices = set(e["index"] for e in pac_errors)
    pac_dict = {e["index"]: e for e in pac_errors}
    
    print(f"[PAC] Loaded {len(pac_indices)} persistent errors from {pac_json_path}")
    print(f"      PAC Persistent Error Rate: {len(pac_indices) / N * 100:.3f}%")
    
    # 3. Compute Cleanlab label issues
    oof_probs = compute_out_of_fold_probabilities(x_train, y_train, n_splits=3, epochs=4)
    
    print("\n[CLEANLAB] Running find_label_issues() using Confident Learning...")
    t0 = time.time()
    cleanlab_indices = cleanlab.filter.find_label_issues(
        labels=y_np,
        pred_probs=oof_probs,
        return_indices_ranked_by="self_confidence"
    )
    cleanlab_time = time.time() - t0
    cleanlab_set = set(cleanlab_indices)
    
    print(f"[CLEANLAB] Completed in {cleanlab_time:.2f}s")
    print(f"           Cleanlab Label Issues Detected: {len(cleanlab_set)} ({len(cleanlab_set) / N * 100:.3f}%)")
    
    # 4. Compute Intersection and Alignment Metrics
    intersection = pac_indices & cleanlab_set
    union = pac_indices | cleanlab_set
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
    pac_recall_of_cleanlab = len(intersection) / len(cleanlab_set) if len(cleanlab_set) > 0 else 0.0
    cleanlab_precision_on_pac = len(intersection) / len(pac_indices) if len(pac_indices) > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("CONVERGENCE & INTERSECTION METRICS")
    print("=" * 80)
    print(f"PAC Persistent Errors count     : {len(pac_indices)} ({len(pac_indices)/N*100:.2f}%)")
    print(f"Cleanlab Confident Learning count: {len(cleanlab_set)} ({len(cleanlab_set)/N*100:.2f}%)")
    print(f"Exact Overlapping Samples (Intersection): {len(intersection)}")
    print(f"Jaccard Similarity Index         : {jaccard:.4f}")
    print(f"PAC Alignment with Cleanlab      : {pac_recall_of_cleanlab*100:.2f}%")
    print(f"Cleanlab Agreement with PAC      : {cleanlab_precision_on_pac*100:.2f}%")
    
    # 5. Extract Detailed Overlapping Samples
    overlapping_details = []
    for idx in sorted(list(intersection)):
        idx_int = int(idx)
        true_l = int(y_np[idx_int])
        pac_pred = int(pac_dict[idx_int]["pred_label"])
        cleanlab_pred = int(np.argmax(oof_probs[idx_int]))
        cleanlab_conf = float(oof_probs[idx_int, true_l])
        pac_conf = float(pac_dict[idx_int]["confidence"])
        
        overlapping_details.append({
            "index": idx_int,
            "true_label": true_l,
            "pac_predicted": pac_pred,
            "cleanlab_predicted": cleanlab_pred,
            "cleanlab_self_confidence": round(cleanlab_conf, 4),
            "pac_similarity": round(pac_conf, 4)
        })
        
    print(f"\n--- Top 15 Exact Overlapping Mislabeled Samples (Detected by BOTH PAC & Cleanlab) ---")
    print(f"{'Index':<8} | {'True':<5} | {'PAC Pred':<10} | {'Cleanlab Pred':<14} | {'Cleanlab Conf':<14} | {'PAC Sim':<8}")
    print("-" * 75)
    for sample in overlapping_details[:15]:
        print(f"{sample['index']:<8} | {sample['true_label']:<5} | {sample['pac_predicted']:<10} | {sample['cleanlab_predicted']:<14} | {sample['cleanlab_self_confidence']:<14.4f} | {sample['pac_similarity']:<8.4f}")
        
    # Check canonical known MNIST errors
    canonical_indices = [59915, 25678, 51274, 59718, 11570, 43454, 4369]
    print("\n--- Canonical Known MNIST Label Errors ---")
    for c_idx in canonical_indices:
        in_pac = c_idx in pac_indices
        in_cleanlab = c_idx in cleanlab_set
        pac_pred = pac_dict[c_idx]["pred_label"] if in_pac else "N/A"
        cl_pred = int(np.argmax(oof_probs[c_idx])) if in_cleanlab else "N/A"
        print(f"Sample #{c_idx:5d} (True: {y_np[c_idx]}): in PAC={in_pac} (PAC pred: {pac_pred}), in Cleanlab={in_cleanlab} (CL pred: {cl_pred})")

    # 6. Save Report to JSON
    report_data = {
        "dataset": "mnist_train",
        "total_samples": N,
        "pac_persistent_error_count": len(pac_indices),
        "cleanlab_label_issues_count": len(cleanlab_set),
        "exact_intersection_count": len(intersection),
        "jaccard_similarity": round(jaccard, 4),
        "pac_recall_of_cleanlab": round(pac_recall_of_cleanlab, 4),
        "cleanlab_precision_on_pac": round(cleanlab_precision_on_pac, 4),
        "overlapping_samples": overlapping_details
    }
    
    out_json = "results/mnist_audit/cleanlab_vs_pac_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[SAVED] Comparison JSON saved to {out_json}")
    
    # 7. Generate Comparison Visualization Figure
    n_display = min(16, len(overlapping_details))
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.5), dpi=150)
    axes = axes.ravel()
    
    for i in range(n_display):
        item = overlapping_details[i]
        idx = item["index"]
        img = x_train[idx].view(28, 28).numpy()
        
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(
            f"#{idx}\nTrue: {item['true_label']}\nPAC: {item['pac_predicted']} | CL: {item['cleanlab_predicted']}",
            fontsize=8, fontweight='bold', color='red'
        )
        
    plt.suptitle("Coincident Label Errors Identified by BOTH PAC and Cleanlab (MNIST 60K)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = "results/figures/cleanlab_pac_overlap_examples.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"[SAVED] Figure saved to {fig_path}\n")


if __name__ == "__main__":
    run_cleanlab_comparison()
