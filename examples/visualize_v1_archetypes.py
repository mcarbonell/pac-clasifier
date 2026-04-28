import sys
import os

# Allow importing pac from project root when running from examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from pac import PurifyingArchetypeClassifier



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

    print("Training PAC-V1...")
    pac = PurifyingArchetypeClassifier(max_iters=100, target_acc=0.999)
    pac.fit(x_train, y_train, verbose=True)

    preds, _ = pac.predict(x_test)
    test_acc = (preds.cpu() == y_test).float().mean().item()
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
    print(f"Total Archetypes: {len(pac.arch_labels)}")

    # Group archetypes by their true label
    archs_by_digit = {i: [] for i in range(10)}
    for tensor, label in zip(pac.arch_tensors, pac.arch_labels):
        archs_by_digit[label.item()].append(tensor)

    max_archs = max(len(lst) for lst in archs_by_digit.values())
    plot_cols = min(max_archs, 20)  # Cap columns for readability

    fig, axes = plt.subplots(10, plot_cols, figsize=(plot_cols * 1.2, 14))
    fig.suptitle(
        f"PAC-V1: Purified Archetypes (Test Acc: {test_acc * 100:.2f}%) | Total: {len(pac.arch_labels)} archetypes",
        fontsize=16
    )

    for digit in range(10):
        lst = archs_by_digit[digit]
        for col in range(plot_cols):
            ax = axes[digit, col]
            if col < len(lst):
                tensor = lst[col]
                ax.imshow(tensor.cpu().view(28, 28).numpy(), cmap='magma')
                if col == 0:
                    ax.set_title(f"Base '{digit}'", fontsize=9, color='red')
            ax.axis('off')

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    save_path = "results/figures/v1_purified_archetypes.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to: {save_path}")


if __name__ == "__main__":
    run()
