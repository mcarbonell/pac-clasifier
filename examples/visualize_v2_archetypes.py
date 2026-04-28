import sys
import os

# Allow importing pac_v2 from project root when running from examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pac_v2 import PurifyingArchetypeClassifierV2


def run():
    print("Loading MNIST dataset...")
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60000, shuffle=False)
    x_train, y_train = next(iter(train_loader))
    x_train = x_train.view(60000, 784)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)
    x_test, y_test = next(iter(test_loader))
    x_test = x_test.view(10000, 784)

    print("Training PAC-V2...")
    pac = PurifyingArchetypeClassifierV2(max_iters=100, target_acc=0.999)
    pac.fit(x_train, y_train, verbose=True)

    preds, _ = pac.predict(x_test)
    test_acc = (preds.cpu() == y_test).float().mean().item()
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
    print(f"Total Archetypes: {len(pac.arch_labels)}")

    # Build enriched archetype list: (tensor, true_label, cid, gen, confused_with)
    archetypes_enriched = []
    for idx in range(len(pac.arch_tensors)):
        tensor = pac.arch_tensors[idx]
        true_label = pac.arch_labels[idx].item()
        cid = pac.arch_cluster_ids[idx].item()
        info = pac.cluster_history.get(cid, {})
        gen = info.get('generation', 0)
        confused_with = info.get('confused_with')
        archetypes_enriched.append({
            'tensor': tensor,
            'true_label': true_label,
            'cid': cid,
            'gen': gen,
            'confused_with': confused_with
        })

    # Group by true label; base (gen=0) first, then by generation
    archs_by_digit = {i: [] for i in range(10)}
    for a in archetypes_enriched:
        archs_by_digit[a['true_label']].append(a)

    for digit in range(10):
        archs_by_digit[digit].sort(key=lambda x: (x['gen'], x['cid']))

    max_archs = max(len(lst) for lst in archs_by_digit.values())
    plot_cols = min(max_archs, 20)  # Cap for readability

    fig, axes = plt.subplots(10, plot_cols, figsize=(plot_cols * 1.3, 16))
    fig.suptitle(
        f"PAC-V2: Confusion-Aware Archetypes (Test Acc: {test_acc * 100:.2f}%) | Total: {len(pac.arch_labels)} archetypes",
        fontsize=16
    )

    # Color map for confused_with labels (0-9)
    cmap_conf = plt.get_cmap('tab10')

    for digit in range(10):
        lst = archs_by_digit[digit]
        for col in range(plot_cols):
            ax = axes[digit, col]
            if col < len(lst):
                a = lst[col]
                tensor = a['tensor']
                ax.imshow(tensor.cpu().view(28, 28).numpy(), cmap='magma')

                if a['gen'] == 0:
                    ax.set_title(f"Base '{digit}'", fontsize=8, color='red')
                else:
                    pred = a['confused_with']
                    color = cmap_conf(pred) if pred is not None else 'white'
                    # Convert rgba to hex for title color
                    title_color = mcolors.to_hex(color)
                    ax.set_title(
                        f"{digit}→{pred} (g{a['gen']})",
                        fontsize=7,
                        color=title_color
                    )
            ax.axis('off')

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    save_path = "results/figures/v2_confusion_archetypes.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to: {save_path}")


if __name__ == "__main__":
    run()
