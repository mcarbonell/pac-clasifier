"""
Comparative Benchmark Suite for PAC Classifier.
Evaluates PAC, GLVQ, KNN, SVM, and MLP across multiple datasets and seeds.
Strictly logs metrics to results/raw/ and results/summary/ adhering to GEMINI.md.

Usage:
    # Rapid dry-run (under 10 seconds):
    python experiments/run_benchmarks.py --quick

    # Full benchmark on Tabular dataset (5 seeds):
    python experiments/run_benchmarks.py --dataset tabular --models all --seeds 42 123 456 789 999

    # Full benchmark on Fashion-MNIST (5 seeds):
    python experiments/run_benchmarks.py --dataset fashion --models pac lvq knn --seeds 42 123 456 789 999
"""

import os
import sys
import time
import argparse
import torch
import numpy as np

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pac import PurifyingArchetypeClassifier
from experiments.baselines import GLVQClassifier, KNNBaseline, SVMBaseline, MLPBaseline
from experiments.datasets import get_dataset
from experiments.metrics_logger import BenchmarkTracker, aggregate_and_save_summary


def instantiate_model(model_name: str, config: dict, seed: int, device: str):
    """Instantiates a classifier given model name and config."""
    name = model_name.lower()
    if name == 'pac':
        return PurifyingArchetypeClassifier(
            max_iters=config.get('max_iters', 50),
            target_acc=config.get('target_acc', 0.999),
            bifurcation_mode=config.get('bifurcation_mode', 'confusion'),
            distance_metric=config.get('distance_metric', 'cosine'),
            min_cluster_size=config.get('min_cluster_size', 1),
            device=device
        )
    elif name == 'lvq':
        return GLVQClassifier(
            prototypes_per_class=config.get('lvq_prototypes_per_class', 2),
            lr=config.get('lvq_lr', 0.02),
            epochs=config.get('lvq_epochs', 15),
            batch_size=config.get('lvq_batch_size', 256),
            device=device
        )
    elif name == 'knn':
        return KNNBaseline(k=config.get('knn_k', 3), metric=config.get('knn_metric', 'cosine'))
    elif name == 'svm':
        return SVMBaseline(c=config.get('svm_c', 10.0), kernel='rbf')
    elif name == 'mlp':
        return MLPBaseline(hidden_dim=config.get('mlp_hidden', 128), max_iter=config.get('mlp_iters', 20), seed=seed)
    else:
        raise ValueError(f"Unknown model name '{model_name}'")


def run_single_experiment(
    model_name: str,
    dataset_name: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    seed: int,
    config: dict,
    device: str,
    verbose: bool = True
) -> dict:
    """Executes a single (model, dataset, seed) training & evaluation run."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    exp_name = f"{dataset_name}_{model_name}"
    tracker = BenchmarkTracker(experiment_name=exp_name, config=config, seed=seed)
    
    model = instantiate_model(model_name, config, seed, device)
    
    tracker.start()
    t_train_start = time.perf_counter()
    
    # Train
    model.fit(x_train, y_train, verbose=verbose)
    train_time = time.perf_counter() - t_train_start
    
    # Test evaluation
    t_eval_start = time.perf_counter()
    preds, scores = model.predict(x_test)
    test_eval_time = time.perf_counter() - t_eval_start
    
    preds_cpu = preds.cpu() if isinstance(preds, torch.Tensor) else torch.tensor(preds)
    y_test_cpu = y_test.cpu() if isinstance(y_test, torch.Tensor) else torch.tensor(y_test)
    test_acc = (preds_cpu == y_test_cpu).float().mean().item()
    
    # Determine prototype count if applicable
    num_prototypes = None
    if hasattr(model, 'arch_tensors') and model.arch_tensors is not None:
        num_prototypes = len(model.arch_tensors)
    elif hasattr(model, 'prototypes') and model.prototypes is not None:
        num_prototypes = len(model.prototypes)
    elif model_name == 'knn':
        num_prototypes = len(x_train)  # KNN retains all training samples
        
    # Record forward pass time
    tracker.record_step(
        iteration=1,
        eval_time=test_eval_time,
        evaluations=len(x_test),
        current_acc=test_acc
    )
    
    extra = {
        "dataset": dataset_name,
        "model": model_name,
        "test_acc": test_acc,
        "train_time_sec": round(train_time, 4),
        "test_eval_time_sec": round(test_eval_time, 4),
        "num_prototypes": num_prototypes
    }
    
    result = tracker.finish(final_objective=test_acc, extra_metrics=extra)
    tracker.save_raw(output_dir="results/raw", result_data=result)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="PAC Comparative Benchmark Suite")
    parser.add_argument("--dataset", type=str, default="tabular", choices=["mnist", "fashion", "tabular", "all"],
                        help="Dataset to evaluate on")
    parser.add_argument("--models", nargs="+", default=["pac", "lvq", "knn", "svm", "mlp"],
                        help="Models to run (pac, lvq, knn, svm, mlp, or all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 999],
                        help="List of random seeds (minimum 5 for scientific rigor)")
    parser.add_argument("--quick", action="store_true",
                        help="Fast verification mode with subsampled data and 1 seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use ('cpu', 'cuda', 'directml')")
    parser.add_argument("--max-iters", type=int, default=50,
                        help="Max iterations for PAC")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]
        datasets = [args.dataset] if args.dataset != "all" else ["tabular", "fashion"]
        print("\n[QUICK MODE] Running with subsampled data, 1 seed, fast verification\n")
    else:
        datasets = ["tabular", "fashion", "mnist"] if args.dataset == "all" else [args.dataset]

    models = ["pac", "lvq", "knn", "svm", "mlp"] if "all" in args.models else args.models

    config = {
        "max_iters": args.max_iters if not args.quick else 5,
        "bifurcation_mode": "confusion",
        "distance_metric": "cosine",
        "min_cluster_size": 1,
        "lvq_epochs": 15 if not args.quick else 3,
        "lvq_prototypes_per_class": 2,
        "mlp_iters": 25 if not args.quick else 5,
        "mlp_hidden": 128,
        "knn_k": 3,
        "quick_mode": args.quick
    }

    print("=" * 75)
    print(f"BENCHMARK SUITE: Datasets={datasets} | Models={models} | Seeds={len(args.seeds)}")
    print("=" * 75)

    all_summaries = {}

    for dname in datasets:
        print(f"\n[DATASET] Loading dataset: {dname.upper()}...")
        x_tr, y_tr, x_te, y_te = get_dataset(dname, quick=args.quick)
        print(f"   Train samples: {x_tr.shape[0]} | Test samples: {x_te.shape[0]} | Dim: {x_tr.shape[1]}")

        for mname in models:
            # Skip SVM on full MNIST/Fashion if on CPU to prevent long run unless explicitly requested
            if mname == "svm" and len(x_tr) > 10000 and not args.quick:
                print(f"[SKIP] Skipping SVM on {dname} (N={len(x_tr)} > 10,000) to avoid slow O(N^2) CPU fit.")
                continue

            exp_name = f"{dname}_{mname}"
            raw_runs = []
            print(f"\n[RUN] Running {mname.upper()} on {dname.upper()} across {len(args.seeds)} seed(s)...")

            for seed in args.seeds:
                res = run_single_experiment(
                    model_name=mname,
                    dataset_name=dname,
                    x_train=x_tr,
                    y_train=y_tr,
                    x_test=x_te,
                    y_test=y_te,
                    seed=seed,
                    config=config,
                    device=args.device,
                    verbose=(args.quick or len(args.seeds) == 1)
                )
                raw_runs.append(res)
                proto_str = f" | Archetypes={res.get('num_prototypes')}" if res.get('num_prototypes') else ""
                print(f"   Seed {seed:4d} -> Test Acc: {res['test_acc']*100:.2f}% | Wall Time: {res['wall_clock_time']:.2f}s{proto_str}")

            summary = aggregate_and_save_summary(exp_name, raw_runs, output_dir="results/summary")
            all_summaries[exp_name] = summary

    # Print final summary table
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY RESULTS TABLE (Results saved to results/summary/ & results/raw/)")
    print("=" * 90)
    header = f"{'Dataset':<12} | {'Model':<8} | {'Test Acc (%)':<18} | {'Wall Time (s)':<16} | {'Overhead (s)':<14} | {'Evaluations':<12}"
    print(header)
    print("-" * 90)

    for exp_key, s in all_summaries.items():
        parts = exp_key.split("_")
        d_label = parts[0]
        m_label = parts[1]
        acc_mean = s["metrics"]["final_objective"]["mean"] * 100
        acc_std = s["metrics"]["final_objective"]["std"] * 100
        time_mean = s["metrics"]["wall_clock_time"]["mean"]
        time_std = s["metrics"]["wall_clock_time"]["std"]
        overhead = s["metrics"]["internal_overhead_time"]["mean"]
        evals = s["metrics"]["total_evaluations"]["mean"]

        acc_str = f"{acc_mean:.2f} +/- {acc_std:.2f}"
        time_str = f"{time_mean:.2f} +/- {time_std:.2f}"
        print(f"{d_label:<12} | {m_label:<8} | {acc_str:<18} | {time_str:<16} | {overhead:<14.2f} | {int(evals):<12}")

    print("=" * 90)
    print("[DONE] Benchmark suite run complete. All results strictly recorded in results/\n")


if __name__ == "__main__":
    main()
