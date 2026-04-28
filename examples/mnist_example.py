import torch
from torchvision import datasets, transforms
from pac import PurifyingArchetypeClassifier
import time

def run_mnist_pac():
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
    
    print("Initializing Purifying Archetype Classifier (PAC)...")
    pac = PurifyingArchetypeClassifier(max_iters=100, target_acc=0.999)
    
    print("Training PAC (this may take a few seconds)...")
    start_time = time.time()
    pac.fit(x_train, y_train, verbose=True)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    print("\nEvaluating on Test Set...")
    preds, confidences = pac.predict(x_test)
    
    correct = (preds.cpu() == y_test).float().sum().item()
    total = len(y_test)
    
    print(f"=======================================")
    print(f"FINAL TEST ACCURACY: {correct/total * 100:.2f}%")
    print(f"Total Discovered Archetypes: {len(pac.arch_labels)}")
    print(f"=======================================")

if __name__ == "__main__":
    run_mnist_pac()
