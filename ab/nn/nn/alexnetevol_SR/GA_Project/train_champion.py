# train_champion.py

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

# --- Path Setup ---
# Add the directory containing the champion model to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
ARCHITECT_DIR = os.path.join(current_dir, 'Alex_architect')
if ARCHITECT_DIR not in sys.path:
    sys.path.append(ARCHITECT_DIR)

# --- Import The Champion ---
# Import the Net class from your GA-generated model file
try:
    from alexnet_ga_0 import Net
except ImportError:
    print(f"FATAL ERROR: Could not find 'alexnet_ga_0.py' in the directory '{ARCHITECT_DIR}'")
    print("Please ensure the champion model file exists.")
    sys.exit(1)


# --- Configuration ---
NUM_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DROPOUT = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- Starting Final Training for Champion Model 'alexnet_ga_0' ---")
    print(f"Using device: {DEVICE}")

    # --- 1. Load and Prepare Data (with Data Augmentation) ---
    print("Loading CIFAR-10 dataset...")
    # Transformations for the training set for better generalization
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    # Transformations for the test set (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- 2. Initialize Model and Optimizer ---
    # We need to manually define the shapes and parameters for the model's __init__
    in_shape = (3, 32, 32)
    out_shape = (10,)
    prm = {'lr': LEARNING_RATE, 'momentum': MOMENTUM, 'dropout': DROPOUT}

    model = Net(in_shape=in_shape, out_shape=out_shape, prm=prm, device=DEVICE)
    model.train_setup(prm) # This sets up the optimizer inside the model

    # --- 3. The Training and Evaluation Loop ---
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.learn(train_loader) # Use the model's own .learn() method for one epoch

        # --- Evaluation Phase ---
        model.eval() # Switch to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Accuracy: {accuracy:.2f}%")

    print("\n--- Final Training Complete ---")
    print(f"Final Validation Accuracy for alexnet_ga_0: {accuracy:.2f}%")


if __name__ == "__main__":
    main()