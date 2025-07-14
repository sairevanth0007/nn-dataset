# run_ga.py

import os
import sys
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import shutil 
import warnings 

# --- Suppress PyTorch UserWarnings for cleaner output ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- Path Setup ---
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Also add the nn-dataset root to sys.path so subprocesses can find 'ab' package directly.
# This script MUST be run from nn-dataset/ab/nn/ea/ for correct relative paths.
NN_DATASET_ROOT_FOR_SUBPROCESS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if NN_DATASET_ROOT_FOR_SUBPROCESS not in sys.path:
    sys.path.insert(0, NN_DATASET_ROOT_FOR_SUBPROCESS)
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

# --- Search Space Definition (DEFINITIVELY MOVED HERE: VERY EARLY IN CONFIGURATION) ---
SEARCH_SPACE = {
    'conv1_filters': [32, 64, 96],    
    'conv1_kernel': [3, 5],           
    'conv1_stride': [1, 2],           
    'conv2_filters': [64, 128, 192, 256], 
    'conv2_kernel': [3],              
    'conv3_filters': [192, 256, 384, 512], 
    'conv4_filters': [256, 384, 512], 
    'conv5_filters': [256, 512],      
    'fc1_neurons': [1024, 2048, 4096], 
    'fc2_neurons': [1024, 2048, 4096], 
}
# --- End of Search Space Definition ---

# 1. Genetic Algorithm Parameters (UPDATED)
POPULATION_SIZE = 100      
NUM_GENERATIONS = 50       
MUTATION_RATE = 0.20
ELITISM_COUNT = 4          
PARALLEL_WORKERS = 9       

# 2. Fitness Evaluation Parameters (for the GA's proxy evaluation)
FITNESS_EPOCHS = 5           
FITNESS_DATASET_NAME = 'CIFAR10'
FITNESS_PROXY_SUBSET_RATIO = 0.2   
FITNESS_BATCH_SIZE = 64      

# Fixed Hyperparameters for the architectures in the GA search
FIXED_LR = 0.01
FIXED_MOMENTUM = 0.9
FIXED_DROPOUT = 0.5

# 3. Directory and File Paths (UPDATED for new 'ea' structure)
GA_PROJECT_DIR = os.path.abspath(os.path.dirname(__file__)) 
GENERATED_MODELS_DIR = os.path.join(GA_PROJECT_DIR, '..', 'nn') 
CHECKPOINT_PATH = os.path.join(GA_PROJECT_DIR, 'ga_checkpoint.pkl')

# Absolute path to the data root (e.g., 'nn-dataset/data')
# This is crucial for multiprocessing 'spawn' context.
ABSOLUTE_DATA_ROOT = os.path.join(NN_DATASET_ROOT_FOR_SUBPROCESS, 'data')
os.makedirs(ABSOLUTE_DATA_ROOT, exist_ok=True) # Ensure data dir exists

# 4. Global Shapes (needed by worker to instantiate model, and are simple tuples so can be global)
GLOBAL_IN_SHAPE = (3, 32, 32)
GLOBAL_OUT_SHAPE = (10,)


# This is the function that will be executed by each parallel worker.
def fitness_function_worker(individual: dict) -> tuple: # Returns tuple (id, fitness)
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

    # 2. Save the code to a .py file in the GENERATED_MODELS_DIR
    model_path_in_generated_dir = os.path.join(GENERATED_MODELS_DIR, f"{model_name}.py")
    with open(model_path_in_generated_dir, 'w') as f:
        f.write(model_code)

    # Add the GENERATED_MODELS_DIR to sys.path so we can import the generated model
    original_sys_path = sys.path[:]
    sys.path.insert(0, GENERATED_MODELS_DIR)

    # Determine device for this worker (each worker can use GPU if available)
    # Check for CUDA availability *within the worker process*
    worker_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # This also determines if pin_memory should be used
    use_pin_memory = worker_device.type == 'cuda'

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

        # Load datasets with ABSOLUTE_DATA_ROOT
        full_train_set_worker = torchvision.datasets.CIFAR10(root=ABSOLUTE_DATA_ROOT, train=True, download=True, transform=train_transform_worker)
        full_test_set_worker = torchvision.datasets.CIFAR10(root=ABSOLUTE_DATA_ROOT, train=False, download=True, transform=test_transform_worker)

        # Create proxy subset for training
        train_subset_size = int(FITNESS_PROXY_SUBSET_RATIO * len(full_train_set_worker))
        remaining_size = len(full_train_set_worker) - train_subset_size
        train_proxy_subset_worker, _ = random_split(full_train_set_worker, [train_subset_size, remaining_size])

        # Create DataLoaders for this worker (num_workers=0 is crucial in child processes)
        train_loader_proxy_worker = DataLoader(train_proxy_subset_worker, batch_size=FITNESS_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=use_pin_memory)
        test_loader_proxy_worker = DataLoader(full_test_set_worker, batch_size=FITNESS_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=use_pin_memory)

        # Import the generated model's Net class
        model_module = __import__(model_name)
        Net = getattr(model_module, 'Net')

        # Instantiate the model
        prm_for_model = {'lr': FIXED_LR, 'momentum': FIXED_MOMENTUM, 'dropout': FIXED_DROPOUT}
        model = Net(in_shape=GLOBAL_IN_SHAPE, out_shape=GLOBAL_OUT_SHAPE, prm=prm_for_model, device=worker_device) # Use worker_device
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
                inputs, labels = inputs.to(worker_device), labels.to(worker_device) # Use worker_device
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        
        # Uncomment for verbose debugging of individual workers (will make a lot of output!)
        # print(f"[Worker {model_id}] Proxy Eval complete. Accuracy: {accuracy:.2f}% (Device: {worker_device})", file=sys.stderr)
        return (model_id, accuracy) # Return ID along with fitness
    
    except Exception as e:
        print(f"[Worker Error {model_id}] Failed during evaluation. Device: {worker_device}. Error: {e}", file=sys.stderr)
        return (model_id, 0.0) # Return ID with 0.0 fitness for failure
    finally:
        # Restore the original sys.path to prevent pollution
        sys.path = original_sys_path
        # Files are kept as per new requirement, so no deletion here.


def parallel_evaluator(individuals_to_evaluate: list) -> list:
    """
    Uses a multiprocessing Pool to evaluate a list of individuals in parallel.
    """
    # Using 'spawn' context for better compatibility with PyTorch/CUDA and dynamic imports.
    ctx = multiprocessing.get_context('spawn') 
    with ctx.Pool(processes=PARALLEL_WORKERS) as pool:
        # Use tqdm to create a progress bar for the parallel evaluations
        # map() returns results in order, which is good.
        results = list(tqdm(pool.map(fitness_function_worker, individuals_to_evaluate), 
                            total=len(individuals_to_evaluate),
                            desc="Evaluating Models")) # Added description for tqdm
    return results


if __name__ == "__main__":
    # Ensure the GENERATED_MODELS_DIR exists
    os.makedirs(GENERATED_MODELS_DIR, exist_ok=True)
    
    print("--- Starting Genetic Algorithm for Neural Architecture Search ---")
    # Check and print device for the MAIN process
    main_process_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main process running on device: {main_process_device}")
    print(f"GA Parameters: Population={POPULATION_SIZE}, Generations={NUM_GENERATIONS}, Parallel Workers={PARALLEL_WORKERS}")
    print(f"Proxy Eval: Epochs={FITNESS_EPOCHS}, Data Subset={FITNESS_PROXY_SUBSET_RATIO*100}%, Batch Size={FITNESS_BATCH_SIZE}")
    print(f"Generated models will be saved in: {GENERATED_MODELS_DIR}")

    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        search_space=SEARCH_SPACE, # SEARCH_SPACE is now defined above
        elitism_count=ELITISM_COUNT,
        mutation_rate=MUTATION_RATE,
        checkpoint_path=CHECKPOINT_PATH
    )

    best_individual = ga.run(
        num_generations=NUM_GENERATIONS,
        parallel_evaluator=parallel_evaluator
    )

    print("\n--- Evolution Finished! ---")
    if best_individual:
        print("\n==============================================")
        print("           BEST ARCHITECTURE FOUND            ")
        print("==============================================")
        print(f"  - Fitness (Proxy Validation Accuracy): {best_individual['fitness']:.4f}%")
        print(f"  - Model ID: alexnet_ga_{best_individual['id']}")
        print("  - Chromosome (Parameters):")
        for gene, value in best_individual['chromosome'].items():
            print(f"    - {gene}: {value}")
        print("==============================================")


        champion_model_name_base = f"alexnet_ga_{best_individual['id']}" 
        champion_model_filename = f"{champion_model_name_base}.py"
        source_path = os.path.join(GENERATED_MODELS_DIR, champion_model_filename)
        
        print(f"\nThe full code for this champion model is saved at: {source_path}")
        
        # --- NEW: PHASE 2 - AUTOMATIC FINAL VALIDATION (Using framework's run.py) ---
        print("\n--- Starting Final Validation Phase (Full 100 Epochs) ---")
        print(f"Training champion model '{champion_model_name_base}' using nn-dataset framework.")
        
        try:
            # Re-generate the champion's model code from its stored chromosome
            champion_model_code = get_model_code(best_individual['chromosome'])

            # Save the champion model to the GENERATED_MODELS_DIR.
            with open(source_path, 'w') as f:
                f.write(champion_model_code)
            
            print(f"Champion model ensures to be in place at: {source_path}")

            # Construct the command to run the framework for full validation
            final_train_command = [
                'python',
                'run.py',
                '-c', f'img-classification_{FITNESS_DATASET_NAME.lower()}_acc_{champion_model_name_base}',
                '-e', '100',  # Full training epochs
                '-t', '-1',     # Force one *additional* trial to bypass existing trial checks
                '--min_batch_binary_power', '6', '-b', '6', # Fixed batch size for full run (2^6=64)
                '--min_learning_rate', str(FIXED_LR), '-l', str(FIXED_LR), 
                '--min_momentum', str(FIXED_MOMENTUM), '-m', str(FIXED_MOMENTUM),
                '--min_dropout', str(FIXED_DROPOUT), '-d', str(FIXED_DROPOUT),
                '-f', 'norm_299' # Important: use norm_299 for the full AlexNet on larger images!
            ]
            
            print("\nExecuting final training command:")
            print(' '.join(final_train_command))
            
            # 3. Execute the command
            import subprocess
            subprocess.run(final_train_command, cwd=NN_DATASET_ROOT_FOR_SUBPROCESS, check=True)
            
            print("\n--- Final Validation Complete ---")

        except FileNotFoundError:
            print(f"Error: Could not find the champion model file at {source_path}", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error: The final training run failed with exit code {e.returncode}", file=sys.stderr)
            print(f"Subprocess Stdout:\n{e.stdout.decode('utf-8') if e.stdout else ''}", file=sys.stderr) 
            print(f"Subprocess Stderr:\n{e.stderr.decode('utf-8') if e.stderr else ''}", file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred during final validation: {e}", file=sys.stderr)

    else:
        print("Could not determine a best individual. The evolution may have been stopped early or no valid models were found.")