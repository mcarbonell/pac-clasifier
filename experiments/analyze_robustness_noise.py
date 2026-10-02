"""
Controlled Label Noise Experiment for PAC Classifier.
Injects 0%, 5%, 10%, 15%, 20% symmetric label noise into training data and evaluates:
1. PAC Test Accuracy degradation under noise.
2. Alignment between injected noise rate and detected persistent errors rate.
3. Precision, Recall, and F1 of persistent errors as label noise detectors (Cleanlab baseline comparison).

Outputs saved to results/raw/, results/summary/, and results/figures/.
"""

import os
import sys
import argparse
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Tuple
from pac import PurifyingArchetypeClassifier
from experiments.datasets import get_dataset
from experiments.metrics_logger import get_git_commit_hash, get_hardware_info


def inject_symmetric_noise(y: torch.Tensor, noise_rate: float, num_classes: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly flips labels of a fraction `noise_rate` of samples to another random class.
    Returns:
        noisy_y: tensor of labels with injected noise
        noisy_mask: boolean tensor indicating which samples were flipped
    """
    if noise_rate <= 0.0:
        return y.clone(), torch.zeros(len(y), dtype=torch.bool)
        
    torch.manual_seed(seed)
    n = len(y)
    num_to_flip = int(n * noise_rate)
    
    perm = torch.randperm(n)
    flip_indices = perm[:num_to_flip]
    
    noisy_y = y.clone()
    noisy_mask = torch.zeros(n, dtype=torch.bool)
    noisy_mask[flip_indices] = True
    
    for idx in flip_indices:
        original = y[idx].item()
        # Pick another class uniformly at random
        other_classes = [c for c in range(num_classes) if c != original]
        chosen = other_classes[torch.randint(0, len(other_classes), (1,)).item()]
        noisy_y[idx] = chosen
        
    return noisy_y, noisy_mask


def run_noise_experiment(
    dataset_name: str = "tabular",
    noise_levels: list = [0.0, 0.05, 0.10, 0.15, 0.20],
    seeds: list = [42, 123, 456],
    quick: bool = False,
    max_iters: int = 50,
    device: str = "cpu"
) -> dict:
    """Executes noise injection experiments across multiple noise levels and seeds."""
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/summary", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    print("=" * 75)
    print(f"CONTROLLED LABEL NOISE EXPERIMENT: Dataset={dataset_name.upper()} | Noise={noise_levels}")
    print("=" * 75)

    x_tr, y_tr, x_te, y_te = get_dataset(dataset_name, quick=quick)
    num_classes = len(torch.unique(y_tr))
    n_train = len(y_tr)
    
    print(f"[DATASET] Loaded {dataset_name}: {n_train} train samples, {len(y_te)} test samples, {num_classes} classes.")

    results_by_noise = {str(nl): [] for nl in noise_levels}

    for nl in noise_levels:
        print(f"\n---> Evaluating Injected Noise Rate: {nl * 100:.1f}% across {len(seeds)} seed(s)...")

        for seed in seeds:
            t0 = time.perf_counter()
            y_tr_noisy, is_noisy = inject_symmetric_noise(y_tr, noise_rate=nl, num_classes=num_classes, seed=seed)
            
            clf = PurifyingArchetypeClassifier(
                max_iters=max_iters if not quick else 5,
                bifurcation_mode="confusion",
                distance_metric="cosine",
                device=device
            )
            clf.fit(x_tr, y_tr_noisy, verbose=False)
            train_time = time.perf_counter() - t0

            # Test evaluation on clean test set
            preds_test, _ = clf.predict(x_te)
            test_acc = (preds_test.cpu() == y_te.cpu()).float().mean().item()

            # Persistent error diagnostics
            persistent_mask = clf.never_correct_mask.cpu()
            detected_noise_count = persistent_mask.sum().item()
            detected_rate = detected_noise_count / n_train
            
            # Confusion matrix of noise detection
            tp = (persistent_mask & is_noisy).sum().item()
            fp = (persistent_mask & ~is_noisy).sum().item()
            fn = (~persistent_mask & is_noisy).sum().item()
            tn = (~persistent_mask & ~is_noisy).sum().item()
            
            precision = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / max(1e-7, precision + recall)

            run_data = {
                "noise_level": nl,
                "seed": seed,
                "test_acc": float(round(test_acc, 4)),
                "detected_noise_rate": float(round(detected_rate, 4)),
                "num_archetypes": len(clf.arch_tensors) if clf.arch_tensors is not None else 0,
                "train_time_sec": float(round(train_time, 4)),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": float(round(precision, 4)),
                "recall": float(round(recall, 4)),
                "f1_score": float(round(f1, 4))
            }
            results_by_noise[str(nl)].append(run_data)
            
            print(f"   Seed {seed:4d} | Clean Test Acc: {test_acc*100:.2f}% | Detected Rate: {detected_rate*100:.2f}% | F1: {f1:.3f} | Archetypes: {run_data['num_archetypes']}")

    # Aggregate summaries
    summary_data = {
        "dataset": dataset_name,
        "commit_hash": get_git_commit_hash(),
        "hardware_info": get_hardware_info(),
        "noise_levels": noise_levels,
        "results": {}
    }

    injected_rates = []
    detected_means = []
    test_acc_means = []
    f1_means = []

    for nl_str, runs in results_by_noise.items():
        nl_val = float(nl_str)
        accs = [r["test_acc"] for r in runs]
        detected = [r["detected_noise_rate"] for r in runs]
        f1s = [r["f1_score"] for r in runs]
        archs = [r["num_archetypes"] for r in runs]

        injected_rates.append(nl_val * 100)
        detected_means.append(np.mean(detected) * 100)
        test_acc_means.append(np.mean(accs) * 100)
        f1_means.append(np.mean(f1s))

        summary_data["results"][nl_str] = {
            "test_acc_mean": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs)),
            "detected_rate_mean": float(np.mean(detected)),
            "detected_rate_std": float(np.std(detected)),
            "f1_mean": float(np.mean(f1s)),
            "archetypes_mean": float(np.mean(archs))
        }

    # Save summary JSON
    sum_path = f"results/summary/{dataset_name}_noise_injection_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[SAVED] Summary written to {sum_path}")

    # Plot correlation figure
    plt.figure(figsize=(10, 4.5), dpi=150)

    # Subplot 1: Injected vs Detected Noise
    plt.subplot(1, 2, 1)
    plt.plot(injected_rates, detected_means, 'o-', color='#1f77b4', linewidth=2, label="PAC Detected Errors")
    plt.plot(injected_rates, injected_rates, '--', color='gray', alpha=0.7, label="Ideal Correlation (y=x)")
    plt.xlabel("Injected Label Noise Rate (%)", fontsize=11)
    plt.ylabel("Persistent Error Rate (%)", fontsize=11)
    plt.title(f"Label Noise Recovery ({dataset_name.upper()})", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Subplot 2: Test Accuracy Degradation
    plt.subplot(1, 2, 2)
    plt.plot(injected_rates, test_acc_means, 's-', color='#d62728', linewidth=2, label="Clean Test Acc")
    plt.xlabel("Injected Label Noise Rate (%)", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.title("Robustness to Corrupted Labels", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    fig_path = f"results/figures/{dataset_name}_noise_injection_analysis.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"[SAVED] Figure saved to {fig_path}\n")

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="PAC Controlled Label Noise Robustness Analysis")
    parser.add_argument("--dataset", type=str, default="tabular", choices=["tabular", "fashion", "mnist"])
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-iters", type=int, default=50)
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]
        args.noise_levels = [0.0, 0.10, 0.20]

    run_noise_experiment(
        dataset_name=args.dataset,
        noise_levels=args.noise_levels,
        seeds=args.seeds,
        quick=args.quick,
        max_iters=args.max_iters,
        device=args.device
    )


if __name__ == "__main__":
    main()
