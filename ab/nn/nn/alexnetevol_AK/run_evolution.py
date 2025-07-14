import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import Evaluators
from tqdm import tqdm

# Adjust sys.path to ensure all modules are discoverable.
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir))
sys.path.insert(0, project_root)


# --- Step 1: Import our custom modules ---
# We import AlexNet_evolvable as a module because SimpleAccuracyEvaluator needs to access its Net class.
import AlexNet_evolvable
from genetic_algorithm import GeneticAlgorithm

# Import the evaluator classes from your consolidated file
from Evaluators.unified_evaluators import NNEvalEvaluator, SimpleAccuracyEvaluator


# --- Step 2: Define Experiment Parameters ---
# GA Parameters
POPULATION_SIZE = 10
NUM_GENERATIONS = 5
MUTATION_RATE = 0.15
ELITISM_COUNT = 2
CHECKPOINT_FILE = 'ga_evolution_checkpoint.dill' # Ensure this matches genetic_algorithm.py


# Fitness Evaluation Parameters (will be used by the chosen evaluator)
EVAL_BATCH_SIZE = 128
EVAL_NUM_EPOCHS = 3
DATASET_NAME = 'CIFAR10'

# Defaults for NNEvalEvaluator (if you choose to use it)
TASK_NAME = 'img-classification'
METRIC_NAME = 'acc'
DEFAULT_LR_EVAL = 0.01
DEFAULT_DROPOUT_EVAL = 0.2
DEFAULT_MOMENTUM_EVAL = 0.9
DEFAULT_TRANSFORM_EVAL = 'norm_256_flip'


# --- Step 3: The Main Execution Block ---
if __name__ == "__main__":
    print("\n--- Starting Genetic Algorithm based Neural Architecture Search ---")

    # *** CHOOSE AND INSTANTIATE YOUR DESIRED EVALUATOR HERE ***
    # This evaluator object will now *provide* the fitness function that GA's run method expects.

    # Option 1: Use NNEvalEvaluator (requires 'ab' library installed, or uses the mock)
    # print(f"\nInitializing NNEvalEvaluator for {DATASET_NAME}...")
    # alexnet_evolvable_path = os.path.abspath(os.path.join(project_root, 'AlexNet_evolvable.py'))
    # current_evaluator = NNEvalEvaluator(
    #     alexnet_evolvable_module_path=alexnet_evolvable_path,
    #     task=TASK_NAME,
    #     dataset=DATASET_NAME,
    #     metric=METRIC_NAME,
    #     nn_train_epochs=EVAL_NUM_EPOCHS,
    #     lr=DEFAULT_LR_EVAL,
    #     batch_size=EVAL_BATCH_SIZE,
    #     dropout=DEFAULT_DROPOUT_EVAL,
    #     momentum=DEFAULT_MOMENTUM_EVAL,
    #     transform=DEFAULT_TRANSFORM_EVAL,
    #     save_to_db=False,
    #     root_temp_dir="nneval_temp_evals"
    # )

    # Option 2: Use SimpleAccuracyEvaluator (pure PyTorch, no 'ab' library dependency)
    print(f"\nInitializing SimpleAccuracyEvaluator for {DATASET_NAME}...")
    current_evaluator = SimpleAccuracyEvaluator(
        alexnet_evolvable_module=AlexNet_evolvable, # Pass the imported module directly
        dataset_name=DATASET_NAME,
        epochs=EVAL_NUM_EPOCHS,
        batch_size=EVAL_BATCH_SIZE
    )


    # --- Step 4: Initialize and Run the Genetic Algorithm ---
    print("\n--- Initializing Genetic Algorithm Engine ---")

    # Create an instance of our GA engine.
    # The 'evaluator' is NOT passed to GeneticAlgorithm's __init__ anymore, as per your request.
    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        search_space=AlexNet_evolvable.SEARCH_SPACE,
        elitism_count=ELITISM_COUNT,
        mutation_rate=MUTATION_RATE,
        checkpoint_path=CHECKPOINT_FILE
    )

    best_chromosome = None
    best_fitness = None
    try:
        # Run the evolution!
        # *** HERE WE PASS THE EVALUATOR'S METHOD AS THE FITNESS FUNCTION TO ga.run() ***
        best_chromosome, best_fitness = ga.run(
            num_generations=NUM_GENERATIONS,
            fitness_function=current_evaluator.evaluate # <--- This is the line you want!
        )

    except Exception as e:
        print(f"\nAn unhandled error occurred during the GA evolution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Crucially, ensure cleanup is still called on the evaluator object,
        # as it was instantiated in this script.
        if 'current_evaluator' in locals() and hasattr(current_evaluator, 'cleanup'):
            print("\nRunning evaluator cleanup...")
            current_evaluator.cleanup()
        print("\nProgram execution finished.")

    # --- Step 5: Display the Final Results ---
    if best_chromosome:
        print("\n--- Evolution Finished! ---")
        print("Best performing network architecture found:")
        print(f"  - Fitness (Validation Accuracy): {best_fitness:.2f}%")
        print("  - Chromosome (Parameters):")
        for gene, value in best_chromosome.items():
            print(f"    - {gene}: {value}")
    else:
        print("\n--- Evolution Finished with no successful individuals found (all had errors or no evaluations). ---")

    print("\nTo fully train this best model, you would now create a new Net instance with this")
    print("chromosome and train it for many more epochs (e.g., 50-100) using a full dataset.")