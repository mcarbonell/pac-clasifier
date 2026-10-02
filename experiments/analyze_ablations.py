"""
Ablation Studies for PAC Classifier:
1. Distance Metric Ablation: Cosine Similarity vs Euclidean Distance (L2).
2. Min Cluster Size Ablation: Impact of min_cluster_size on final archetypes K and accuracy.
3. K(t) Trajectory & Convergence: Archetype growth and training accuracy curves over generations.

Outputs generated in results/figures/ and results/summary/.
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

from pac import PurifyingArchetypeClassifier
from experiments.datasets import get_dataset
from experiments.metrics_logger import get_git_commit_hash, get_hardware_info


def run_ablation_distance_metric(dataset_name: str, x_tr, y_tr, x_te, y_te, seeds: list, max_iters: int, device: str) -> dict:
    """Compares Cosine Similarity vs Euclidean Distance across seeds."""
    print("\n" + "=" * 60)
    print("ABLATION 1: Distance Metric (Cosine vs Euclidean)")
    print("=" * 60)

    results = {"cosine": [], "euclidean": []}

    for metric in ["cosine", "euclidean"]:
        print(f"\n---> Evaluating Metric: {metric.upper()}...")
        for seed in seeds:
            torch.manual_seed(seed)
            clf = PurifyingArchetypeClassifier(
                max_iters=max_iters,
                distance_metric=metric,
                bifurcation_mode="confusion",
                device=device
            )
            t0 = time.perf_counter()
            clf.fit(x_tr, y_tr, verbose=False)
            fit_time = time.perf_counter() - t0

            preds, _ = clf.predict(x_te)
            acc = (preds.cpu() == y_te.cpu()).float().mean().item()
            num_arch = len(clf.arch_tensors) if clf.arch_tensors is not None else 0

            run_entry = {
                "metric": metric,
                "seed": seed,
                "test_acc": float(round(acc, 4)),
                "num_archetypes": num_arch,
                "fit_time_sec": float(round(fit_time, 4))
            }
            results[metric].append(run_entry)
            print(f"   Seed {seed:4d} | Test Acc: {acc*100:.2f}% | Archetypes: {num_arch} | Time: {fit_time:.2f}s")

    # Plot
    metrics = ["cosine", "euclidean"]
    acc_means = [np.mean([r["test_acc"] for r in results[m]]) * 100 for m in metrics]
    acc_stds = [np.std([r["test_acc"] for r in results[m]]) * 100 for m in metrics]
    arch_means = [np.mean([r["num_archetypes"] for r in results[m]]) for m in metrics]

    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=150)
    color = '#1f77b4'
    ax1.set_xlabel('Distance Metric', fontsize=11)
    ax1.set_ylabel('Test Accuracy (%)', color=color, fontsize=11)
    bars = ax1.bar(metrics, acc_means, yerr=acc_stds, color=color, alpha=0.7, width=0.4, capsize=5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(min(acc_means) - 5, 100)

    # Secondary axis for archetypes
    ax2 = ax1.twinx()
    color = '#ff7f0e'
    ax2.set_ylabel('Number of Archetypes (K)', color=color, fontsize=11)
    ax2.plot(metrics, arch_means, color=color, marker='o', linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Ablation: Distance Metric ({dataset_name.upper()})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig_path = f"results/figures/{dataset_name}_ablation_distance_metric.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"[SAVED] Figure saved to {fig_path}")

    return results


def run_ablation_min_cluster_size(dataset_name: str, x_tr, y_tr, x_te, y_te, cluster_sizes: list, seed: int, max_iters: int, device: str) -> dict:
    """Evaluates the effect of min_cluster_size on K and Test Accuracy."""
    print("\n" + "=" * 60)
    print("ABLATION 2: Min Cluster Size Sensitivity")
    print("=" * 60)

    results = []

    for size in cluster_sizes:
        torch.manual_seed(seed)
        clf = PurifyingArchetypeClassifier(
            max_iters=max_iters,
            min_cluster_size=size,
            bifurcation_mode="confusion",
            device=device
        )
        t0 = time.perf_counter()
        clf.fit(x_tr, y_tr, verbose=False)
        fit_time = time.perf_counter() - t0

        preds, _ = clf.predict(x_te)
        acc = (preds.cpu() == y_te.cpu()).float().mean().item()
        num_arch = len(clf.arch_tensors) if clf.arch_tensors is not None else 0

        entry = {
            "min_cluster_size": size,
            "test_acc": float(round(acc, 4)),
            "num_archetypes": num_arch,
            "fit_time_sec": float(round(fit_time, 4)),
            "trajectory": clf.generation_stats
        }
        results.append(entry)
        print(f"   min_cluster_size={size:2d} | Test Acc: {acc*100:.2f}% | Archetypes: {num_arch:4d} | Time: {fit_time:.2f}s")

    # Plot
    sizes = [r["min_cluster_size"] for r in results]
    accs = [r["test_acc"] * 100 for r in results]
    archs = [r["num_archetypes"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4), dpi=150)
    color = '#2ca02c'
    ax1.set_xlabel('Minimum Cluster Size (Threshold to Spawn Archetype)', fontsize=11)
    ax1.set_ylabel('Test Accuracy (%)', color=color, fontsize=11)
    ax1.plot(sizes, accs, 's-', color=color, linewidth=2, label="Test Acc")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Discovered Archetypes (K)', color=color, fontsize=11)
    ax2.plot(sizes, archs, 'o--', color=color, linewidth=2, label="# Archetypes")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Ablation: min_cluster_size vs Model Capacity ({dataset_name.upper()})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig_path = f"results/figures/{dataset_name}_ablation_min_cluster_size.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"[SAVED] Figure saved to {fig_path}")

    # Plot K(t) Trajectory curves
    plt.figure(figsize=(8, 4), dpi=150)
    for r in results:
        traj = r["trajectory"]
        gens = [step["generation"] for step in traj]
        k_vals = [step["num_archetypes"] for step in traj]
        plt.plot(gens, k_vals, label=f"min_size={r['min_cluster_size']}")

    plt.xlabel("Generation (Iteration)", fontsize=11)
    plt.ylabel("Active Archetypes K(t)", fontsize=11)
    plt.title(f"Archetype Growth Trajectory K(t) across Generations ({dataset_name.upper()})", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    kt_fig_path = f"results/figures/{dataset_name}_kt_growth_curves.png"
    plt.savefig(kt_fig_path)
    plt.close()
    print(f"[SAVED] K(t) trajectory curve saved to {kt_fig_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="PAC Ablation Studies Suite")
    parser.add_argument("--dataset", type=str, default="tabular", choices=["tabular", "fashion", "mnist"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--cluster-sizes", nargs="+", type=int, default=[1, 2, 5, 10, 20])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-iters", type=int, default=40)
    args = parser.parse_args()

    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/summary", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    if args.quick:
        args.seeds = [42]
        args.cluster_sizes = [1, 5, 10]
        args.max_iters = 10

    x_tr, y_tr, x_te, y_te = get_dataset(args.dataset, quick=args.quick)

    dist_results = run_ablation_distance_metric(
        dataset_name=args.dataset,
        x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te,
        seeds=args.seeds,
        max_iters=args.max_iters,
        device=args.device
    )

    size_results = run_ablation_min_cluster_size(
        dataset_name=args.dataset,
        x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te,
        cluster_sizes=args.cluster_sizes,
        seed=args.seeds[0],
        max_iters=args.max_iters,
        device=args.device
    )

    # Save summary
    summary_path = f"results/summary/{args.dataset}_ablations_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "commit_hash": get_git_commit_hash(),
            "hardware_info": get_hardware_info(),
            "distance_metric_ablation": dist_results,
            "min_cluster_size_ablation": size_results
        }, f, indent=2)
    print(f"\n[DONE] All ablation experiments completed and saved to {summary_path}\n")


if __name__ == "__main__":
    main()
