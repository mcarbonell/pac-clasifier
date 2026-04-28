import torch
from torchvision import datasets, transforms
import sys
import os

# Add parent directory to path to import pac_v2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pac_v2 import PurifyingArchetypeClassifierV2
import time

def run_mnist_pac_v2():
    print("="*60)
    print("PAC-V2: Purifying Archetype Classifier")
    print("Bifurcation by Misclassification Label")
    print("="*60)
    
    print("\nLoading MNIST dataset...")
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    
    # Load into memory
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60000, shuffle=False)
    x_train, y_train = next(iter(train_loader))
    x_train = x_train.view(60000, 784)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)
    x_test, y_test = next(iter(test_loader))
    x_test = x_test.view(10000, 784)
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Features per sample: {x_train.shape[1]}")
    
    print("\nInitializing PAC-V2...")
    pac = PurifyingArchetypeClassifierV2(
        max_iters=100, 
        target_acc=0.999,
        min_cluster_size=1
    )
    
    print("Training PAC-V2 (this may take a few seconds)...")
    start_time = time.time()
    pac.fit(x_train, y_train, verbose=True)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # Print confusion archetype summary
    pac.print_confusion_archetypes()
    
    # Get detailed archetype info
    print("\nDetailed Archetype Statistics:")
    info = pac.get_archetype_info()
    
    # Count by generation
    gen_counts = {}
    for cid, data in info.items():
        gen = data['generation']
        gen_counts[gen] = gen_counts.get(gen, 0) + 1
    print(f"  Archetypes by generation: {dict(sorted(gen_counts.items()))}")
    
    # Count confusion archetypes by confused_with label
    confusion_targets = {}
    for cid, (true_l, pred_l) in pac.cluster_confusion_map.items():
        confusion_targets[pred_l] = confusion_targets.get(pred_l, 0) + 1
    print(f"  Most confusing classes (as predicted): {dict(sorted(confusion_targets.items(), key=lambda x: -x[1])[:5])}")
    
    print("\nEvaluating on Test Set...")
    preds, confidences = pac.predict(x_test)
    
    correct = (preds.cpu() == y_test).float().sum().item()
    total = len(y_test)
    test_acc = correct / total
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Total Discovered Archetypes: {len(pac.arch_labels)}")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Compression Ratio: {len(x_train) / len(pac.arch_labels):.1f}x")
    print(f"{'='*60}")
    
    # Show some example predictions
    print("\nSample Predictions (first 10 test images):")
    for i in range(10):
        true_label = y_test[i].item()
        pred_label = preds[i].item()
        conf = confidences[i].item()
        status = "✓" if true_label == pred_label else "✗"
        print(f"  {status} True: {true_label}, Pred: {pred_label}, Confidence: {conf:.4f}")

if __name__ == "__main__":
    run_mnist_pac_v2()
