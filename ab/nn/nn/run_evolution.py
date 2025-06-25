# run_evolution_standalone.py

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# --- Step 1: Import our custom modules ---
from AlexNet_evolvable import Net, SEARCH_SPACE  # Our evolvable Net and its search space
from genetic_algorithm import GeneticAlgorithm  # Our GA engine

# --- Step 2: Define Experiment Parameters ---
# GA Parameters
POPULATION_SIZE = 10
NUM_GENERATIONS = 5
MUTATION_RATE = 0.15
ELITISM_COUNT = 2

# Fitness Evaluation Parameters
BATCH_SIZE = 128
NUM_EPOCHS_PER_EVAL = 3

# --- Step 3: The Main Execution Block ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load a standard Torchvision Dataset (e.g., CIFAR-10) ---
    print("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the full training set
    full_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Split the training set into a smaller training set and a validation set
    train_size = int(0.8 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_subset, val_subset = random_split(full_train_set, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Manually define input and output shapes for CIFAR-10
    in_shape = (3, 32, 32)
    out_shape = (10,)
    print(f"Input shape: {in_shape}, Output shape: {out_shape}")


    # --- Step 4: Define the Fitness Function ---
    def fitness_function(chromosome: dict) -> float:
        try:
            model = Net(in_shape, out_shape, chromosome, device)
            model.train_setup(prm=chromosome)

            for epoch in range(NUM_EPOCHS_PER_EVAL):
                model.learn(train_loader)

            # --- Manual Evaluation Loop ---
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"  - Chromosome evaluated. Fitness (Accuracy): {accuracy:.2f}%")
            return accuracy

        except Exception as e:
            print(f"  - Error evaluating chromosome: {e}. Assigning fitness 0.")
            return 0.0


    # --- Step 5: Initialize and Run the Genetic Algorithm ---
    print("\n--- Starting Genetic Algorithm ---")
    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        search_space=SEARCH_SPACE,
        elitism_count=ELITISM_COUNT,
        mutation_rate=MUTATION_RATE
    )
    best_individual = ga.run(
        num_generations=NUM_GENERATIONS,
        fitness_function=fitness_function
    )

    # --- Step 6: Display the Final Results ---
    print("\n--- Evolution Finished! ---")
    print("Best performing network architecture found:")
    print(f"  - Fitness (Validation Accuracy): {best_individual['fitness']:.2f}%")
    print("  - Chromosome (Parameters):")
    for gene, value in best_individual['chromosome'].items():
        print(f"    - {gene}: {value}")