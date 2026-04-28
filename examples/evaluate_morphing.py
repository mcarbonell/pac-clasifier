import torch
from torchvision import datasets, transforms
from pac_v2.classifier import PurifyingArchetypeClassifierV2
import time

def run_evaluation():
    print("Loading MNIST dataset...")
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
    
    print("Initializing PAC-V2...")
    pac = PurifyingArchetypeClassifierV2(max_iters=100, target_acc=0.999)
    
    print("Training PAC-V2...")
    start_time = time.time()
    pac.fit(x_train, y_train, verbose=True)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    # 1. Standard Prediction
    print("\nEvaluating Standard Prediction...")
    start_std = time.time()
    preds_std, sims_std = pac.predict(x_test)
    end_std = time.time()
    
    correct_std = (preds_std.cpu() == y_test).float().sum().item()
    acc_std = correct_std / len(y_test)
    
    # 2. Morphing Prediction (on a subset for speed, or full if possible)
    # We'll use a subset of 1000 images to keep it reasonable, or full if you prefer.
    # Let's try 1000 first to see the effect.
    subset_size = 1000
    print(f"\nEvaluating Morphing Prediction (Subset of {subset_size} images)...")
    x_test_sub = x_test[:subset_size]
    y_test_sub = y_test[:subset_size]
    
    start_morph = time.time()
    preds_morph, sims_morph = pac.predict_with_morphing(x_test_sub, top_k=10, morph_iters=10)
    end_morph = time.time()
    
    correct_morph = (preds_morph.cpu() == y_test_sub).float().sum().item()
    acc_morph = correct_morph / subset_size
    
    print(f"\n=======================================")
    print(f"Standard Accuracy: {acc_std * 100:.2f}% (Time: {end_std - start_std:.4f}s)")
    print(f"Morphing Accuracy: {acc_morph * 100:.2f}% (Time: {end_morph - start_morph:.4f}s for {subset_size} imgs)")
    print(f"Improvement: {(acc_morph - acc_std) * 100:.2f}%")
    print(f"Total Archetypes: {len(pac.arch_labels)}")
    print(f"=======================================")

if __name__ == "__main__":
    run_evaluation()
