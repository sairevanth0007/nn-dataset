# run_ga.py

import os
import sys
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import shutil # Added for copying files

# --- Path Setup ---
# Forcefully add the current working directory to the Python path.
# This ensures that when we run from the nn-dataset root, it can find the 'ab' package (for final run).
# This script MUST be run from the root of the 'nn-dataset' directory.
sys.path.insert(0, os.getcwd())
# --------------------

# --- Local Imports ---
from ga_engine import GeneticAlgorithm
from model_template import get_model_code

# Required for the worker's direct PyTorch training loop
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# --- Configuration ---

# 1. Genetic Algorithm Parameters
POPULATION_SIZE = 40
NUM_GENERATIONS = 20
MUTATION_RATE = 0.20
ELITISM_COUNT = 4
PARALLEL_WORKERS = 9 # Use 60% of 16 cores, rounded down

# 2. Fitness Evaluation Parameters (for the GA's proxy evaluation)
FITNESS_EPOCHS = 5           # Train each candidate for 5 epochs
FITNESS_DATASET_NAME = 'CIFAR10'
FITNESS_PROXY_SUBSET_RATIO = 0.2   # Use 20% of the data for faster evaluation
FITNESS_BATCH_SIZE = 64      # Fixed batch size for proxy eval (2^6)

# Fixed Hyperparameters for the architectures in the GA search
FIXED_LR = 0.01
FIXED_MOMENTUM = 0.9
FIXED_DROPOUT = 0.5

# 3. Directory and File Paths
# Get the directory where this script lives, to find other GA files
GA_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHITECT_DIR = os.path.join(GA_PROJECT_DIR, 'Alex_architect')
CHECKPOINT_PATH = os.path.join(GA_PROJECT_DIR, 'ga_checkpoint.pkl')

# Get the absolute path to the nn-dataset root for subprocess calls (for final run)
NN_DATASET_ROOT_FOR_SUBPROCESS = os.getcwd() # Should be /home/akashdeepsingh/nn-dataset

# 4. Global Shapes (needed by worker to instantiate model, and are simple tuples so can be global)
GLOBAL_IN_SHAPE = (3, 32, 32)
GLOBAL_OUT_SHAPE = (10,)

# --- Search Space Definition --- <<<--- MOVED HERE
SEARCH_SPACE = {
    'conv1_filters': [32, 64],
    'conv1_kernel': [3, 5],
    'conv1_stride': [1, 2],
    'conv2_filters': [64, 128, 192],
    'conv2_kernel': [3],
    'conv3_filters': [192, 256, 384],
    'conv4_filters': [256, 384],
    'conv5_filters': [256],
    'fc1_neurons': [1024, 2048],
    'fc2_neurons': [1024, 2048],
}
# --- End of Search Space Definition ---


# --- Data Loading for Workers (This will be done inside each worker's process) ---
# Removed global data loaders from here as they cause multiprocessing issues.
# Data loading will be handled independently by each worker.


# This is the function that will be executed by each parallel worker.
def fitness_function_worker(individual: dict) -> float:
    """
    Takes a single individual, generates its model file, and evaluates its fitness
    by running a direct PyTorch training loop within the worker process.
    Each worker initializes its own data loaders for safety.
    """
    chromosome = individual['chromosome']
    model_id = individual['id']
    model_name = f'alexnet_ga_{model_id}'

    # 1. Generate the model's Python code
    model_code = get_model_code(chromosome)

    # 2. Save the code to a .py file in the Alex_architect directory
    model_path_in_architect = os.path.join(ARCHITECT_DIR, f"{model_name}.py")
    with open(model_path_in_architect, 'w') as f:
        f.write(model_code)

    # Add the Alex_architect directory to sys.path so we can import the generated model
    # This is done here because each worker process has its own sys.path.
    original_sys_path = sys.path[:]
    sys.path.insert(0, ARCHITECT_DIR)

    # Determine device for this worker (each worker can use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # --- Worker's own Data Loading for Proxy Evaluation ---
        # Define transformations (consistent with main process)
        train_transform_worker = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        test_transform_worker = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

        # Load datasets (download=True is safe here as main process should have downloaded)
        full_train_set_worker = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform_worker)
        full_test_set_worker = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform_worker)

        # Create proxy subset for training
        train_subset_size = int(FITNESS_PROXY_SUBSET_RATIO * len(full_train_set_worker))
        remaining_size = len(full_train_set_worker) - train_subset_size
        train_proxy_subset_worker, _ = random_split(full_train_set_worker, [train_subset_size, remaining_size])

        # Create DataLoaders for this worker (num_workers=0 is crucial in child processes)
        train_loader_proxy_worker = DataLoader(train_proxy_subset_worker, batch_size=FITNESS_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        test_loader_proxy_worker = DataLoader(full_test_set_worker, batch_size=FITNESS_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        # Import the generated model's Net class
        # The '__import__' and 'getattr' method ensures dynamic import works across processes
        model_module = __import__(model_name)
        Net = getattr(model_module, 'Net')

        # Instantiate the model
        prm_for_model = {'lr': FIXED_LR, 'momentum': FIXED_MOMENTUM, 'dropout': FIXED_DROPOUT}
        model = Net(in_shape=GLOBAL_IN_SHAPE, out_shape=GLOBAL_OUT_SHAPE, prm=prm_for_model, device=device)
        model.train_setup(prm_for_model)

        # --- Manual Training Loop for Proxy Evaluation ---
        for epoch in range(FITNESS_EPOCHS):
            model.learn(train_loader_proxy_worker) # Train on proxy data

        # --- Manual Evaluation Loop ---
        model.eval() # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader_proxy_worker: # Evaluate on full test set
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        
        # print(f"Worker {model_id} evaluated. Accuracy: {accuracy:.2f}%", file=sys.stderr) # Uncomment for verbose debugging
        return accuracy

    except Exception as e:
        print(f"Worker Error for {model_id}: {e}", file=sys.stderr)
        return 0.0
    finally:
        # Restore the original sys.path to prevent pollution
        sys.path = original_sys_path
        # Clean up the generated model file after evaluation
        # if os.path.exists(model_path_in_architect): # <<< THIS LINE IS NOW COMMENTED OUT
        #     os.remove(model_path_in_architect)      # <<< THIS LINE IS NOW COMMENTED OUT


def parallel_evaluator(individuals_to_evaluate: list) -> list:
    """
    Uses a multiprocessing Pool to evaluate a list of individuals in parallel.
    """
    # Using 'fork' context for better compatibility with PyTorch/CUDA and dynamic imports.
    ctx = multiprocessing.get_context('fork') 
    with ctx.Pool(processes=PARALLEL_WORKERS) as pool:
        # Use tqdm to create a progress bar for the parallel evaluations
        results = list(tqdm(pool.imap(fitness_function_worker, individuals_to_evaluate), total=len(individuals_to_evaluate)))
    return results


if __name__ == "__main__":
    # Ensure the architecture directory exists
    os.makedirs(ARCHITECT_DIR, exist_ok=True)
    
    print("--- Starting Genetic Algorithm for Neural Architecture Search ---")

    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        search_space=SEARCH_SPACE, # SEARCH_SPACE is now defined above
        elitism_count=ELITISM_COUNT,
        mutation_rate=MUTATION_RATE,
        checkpoint_path=CHECKPOINT_PATH
    )

    best_individual = ga.run(
        num_generations=NUM_GENERATIONS,
        fitness_function=None, # We provide a parallel evaluator instead
        parallel_evaluator=parallel_evaluator
    )

    print("\n--- Evolution Finished! ---")
    if best_individual:
        print("Best performing network architecture found:")
        print(f"  - Fitness (Validation Accuracy): {best_individual['fitness']:.4f}%")
        print(f"  - Model ID: alexnet_ga_{best_individual['id']}")
        print("  - Chromosome (Parameters):")
        for gene, value in best_individual['chromosome'].items():
            print(f"    - {gene}: {value}")

        champion_model_name_base = f"alexnet_ga_{best_individual['id']}" 
        champion_model_filename = f"{champion_model_name_base}.py"
        source_path = os.path.join(ARCHITECT_DIR, champion_model_filename)
        
        print(f"\nThe full code for this model is saved at: {source_path}")
        
        # --- NEW: PHASE 2 - AUTOMATIC FINAL VALIDATION (Using framework's run.py) ---
        print("\n--- Starting Final Validation Phase ---")
        print("This will train the champion model for 100 epochs using the framework's run.py")
        
        try:
            # Re-generate the champion's model code from its stored chromosome
            # This ensures we have the content regardless of whether the temp file was deleted by other workers.
            champion_model_code = get_model_code(best_individual['chromosome'])

            # 1. Save the champion model to the framework's model directory directly
            # os.getcwd() must be the nn-dataset root for this to work correctly.
            destination_dir = os.path.join(os.getcwd(), 'ab', 'nn') 
            destination_path = os.path.join(destination_dir, champion_model_filename)
            
            with open(destination_path, 'w') as f:
                f.write(champion_model_code)
            
            print(f"Successfully created champion model at: {destination_path}")

            # 2. Construct the command to run the framework for full validation
            final_train_command = [
                'python',
                'run.py',
                '-c', f'img-classification_{FITNESS_DATASET_NAME.lower()}_acc_{champion_model_name_base}',
                '-e', '100',  # Full training epochs
                '-t', '1',     # Force one new trial
                '--min_batch_binary_power', '6', '-b', '6', # Fixed batch size for full run
                '--min_learning_rate', str(FIXED_LR), '-l', str(FIXED_LR), 
                '--min_momentum', str(FIXED_MOMENTUM), '-m', str(FIXED_MOMENTUM),
                '--min_dropout', str(FIXED_DROPOUT), '-d', str(FIXED_DROPOUT),
                '-f', 'norm_299' # Important: use norm_299 for the full AlexNet on larger images!
            ]
            
            print("\nExecuting final training command:")
            print(' '.join(final_train_command))
            
            # 3. Execute the command
            import subprocess
            subprocess.run(final_train_command, cwd=os.getcwd(), check=True)
            
            print("\n--- Final Validation Complete ---")

        except FileNotFoundError:
            print(f"Error: Could not find the champion model file at {source_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error: The final training run failed with exit code {e.returncode}")
            # Printing stdout/stderr from the subprocess might be helpful for debugging
            print("Subprocess stdout:", e.stdout.decode('utf-8') if e.stdout else "", file=sys.stderr) 
            print("Subprocess stderr:", e.stderr.decode('utf-8') if e.stderr else "", file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred during final validation: {e}")

    else:
        print("Could not determine a best individual. The evolution may have been stopped early or no valid models were found.")