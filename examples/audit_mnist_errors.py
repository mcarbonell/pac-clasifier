import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from pac_v2 import PurifyingArchetypeClassifierV2


def run_mnist_audit():
    print("=" * 70)
    print("PAC-V2: MNIST Label Error Auditor")
    print("Identifica candidatos a mislabels por persistencia de errores")
    print("=" * 70)

    # Load MNIST
    print("\nLoading MNIST dataset...")
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60000, shuffle=False)
    x_train, y_train = next(iter(train_loader))
    x_train = x_train.view(60000, 784)
    y_train = y_train

    print(f"Training samples: {len(x_train)}")

    # Train PAC-V2
    print("\nTraining PAC-V2 until stagnation...")
    pac = PurifyingArchetypeClassifierV2(max_iters=100, target_acc=0.999, min_cluster_size=1)
    pac.fit(x_train, y_train, verbose=True)

    # Extract persistent errors
    persistent_indices = pac.persistent_error_indices
    num_persistent = len(persistent_indices)

    if num_persistent == 0:
        print("\nNo persistent errors found! All samples were classified correctly at some point.")
        return

    print(f"\n{'=' * 70}")
    print("AUDIT RESULTS")
    print(f"{'=' * 70}")
    print(f"Total training samples: {len(x_train)}")
    print(f"Persistent errors (never correct): {num_persistent}")
    print(f"Persistent error rate: {num_persistent / len(x_train) * 100:.2f}%")

    # Gather metadata for each persistent error
    errors_data = []
    by_confusion = {}

    for idx in persistent_indices:
        idx = idx.item()
        true_l = y_train[idx].item()
        pred_l = pac.last_train_preds[idx].item()
        conf = pac.last_train_confidences[idx].item()

        confusion_key = f"{true_l}→{pred_l}"
        by_confusion[confusion_key] = by_confusion.get(confusion_key, []) + [idx]

        errors_data.append({
            "index": idx,
            "true_label": true_l,
            "pred_label": pred_l,
            "confidence": round(conf, 6)
        })

    # Sort by confidence ascending (most uncertain first)
    errors_data.sort(key=lambda x: x["confidence"])

    # Print confusion summary
    print(f"\nConfusion breakdown (top 10):")
    sorted_confusions = sorted(by_confusion.items(), key=lambda x: -len(x[1]))
    for key, indices in sorted_confusions[:10]:
        print(f"  {key}: {len(indices)} samples")

    # Save JSON
    os.makedirs("results/mnist_audit", exist_ok=True)
    json_path = "results/mnist_audit/persistent_errors.json"
    with open(json_path, "w") as f:
        json.dump({
            "total_train": len(x_train),
            "persistent_error_count": num_persistent,
            "persistent_error_rate": round(num_persistent / len(x_train), 6),
            "errors": errors_data,
            "by_confusion": {k: v for k, v in sorted_confusions}
        }, f, indent=2)
    print(f"\nJSON saved to: {json_path}")

    # Visualize top N persistent errors (lowest confidence = most suspicious)
    TOP_N = min(100, num_persistent)
    top_errors = errors_data[:TOP_N]

    cols = 10
    rows = (TOP_N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    fig.suptitle(
        f"MNIST Persistent Errors (Never Correct) | Total: {num_persistent} | Showing {TOP_N} lowest confidence",
        fontsize=14
    )

    for i, ax in enumerate(axes.flat):
        if i < len(top_errors):
            err = top_errors[i]
            idx = err["index"]
            img = x_train[idx].cpu().view(28, 28).numpy()
            ax.imshow(img, cmap='gray')

            color = 'red' if err["true_label"] != err["pred_label"] else 'orange'
            ax.set_title(
                f"#{idx}\nT:{err['true_label']} P:{err['pred_label']}\nC:{err['confidence']:.3f}",
                fontsize=6, color=color
            )
        ax.axis('off')

    plt.tight_layout()
    img_path = "results/mnist_audit/persistent_errors.png"
    plt.savefig(img_path, dpi=200)
    print(f"Visualization saved to: {img_path}")

    # Also create a confusion-focused grid: one subplot per confusion type
    # Show up to 8 examples per confusion type, top 12 confusion types
    TOP_CONFUSIONS = 12
    top_conf_keys = [k for k, _ in sorted_confusions[:TOP_CONFUSIONS]]
    max_per_conf = 8

    fig2, axes2 = plt.subplots(
        len(top_conf_keys), max_per_conf,
        figsize=(max_per_conf * 1.3, len(top_conf_keys) * 1.5)
    )
    fig2.suptitle("Persistent Errors by Confusion Type", fontsize=14)

    for row, conf_key in enumerate(top_conf_keys):
        indices = by_confusion[conf_key][:max_per_conf]
        for col in range(max_per_conf):
            ax = axes2[row, col] if len(top_conf_keys) > 1 else axes2[col]
            if col < len(indices):
                idx = indices[col]
                img = x_train[idx].cpu().view(28, 28).numpy()
                ax.imshow(img, cmap='gray')
                ax.set_title(f"#{idx}", fontsize=7)
            if col == 0:
                ax.set_ylabel(conf_key, fontsize=9, rotation=0, ha='right', va='center')
            ax.axis('off')

    plt.tight_layout()
    img_path2 = "results/mnist_audit/persistent_errors_by_confusion.png"
    plt.savefig(img_path2, dpi=200)
    print(f"Confusion grid saved to: {img_path2}")

    # Print sample indices for manual inspection
    print(f"\n{'=' * 70}")
    print("SAMPLE INDICES FOR MANUAL INSPECTION")
    print(f"{'=' * 70}")
    print("Top 20 most suspicious (lowest confidence):")
    for err in errors_data[:20]:
        status = "MISLABEL?" if err['true_label'] != err['pred_label'] else "AMBIGUOUS"
        print(f"  Index {err['index']:5d}: True={err['true_label']}, Pred={err['pred_label']}, "
              f"Conf={err['confidence']:.4f} [{status}]")


if __name__ == "__main__":
    run_mnist_audit()
