# ga_engine.py

import random
import pickle
import os
from tqdm import tqdm

class GeneticAlgorithm:
    """
    This class implements the core logic for the genetic algorithm.
    It includes robust checkpointing and a comprehensive duplicate architecture checker.
    """
    def __init__(self, population_size, search_space, elitism_count, mutation_rate, checkpoint_path='ga_checkpoint.pkl'):
        """
        Initializes the GA engine.

        Args:
            population_size (int): The number of individuals (chromosomes) in each generation.
            search_space (dict): The dictionary defining the search space.
            elitism_count (int): The number of best individuals to carry over to the next generation.
            mutation_rate (float): The probability (0.0 to 1.0) of a gene mutating.
            checkpoint_path (str): The file path for saving and loading the GA state.
        """
        self.population_size = population_size
        self.search_space = search_space
        self.elitism_count = elitism_count
        self.mutation_rate = mutation_rate
        self.population = []  # List of current individuals: {'chromosome': ..., 'fitness': ..., 'id': ...}
        # Stores {'chromosome_tuple': (fitness_score, id_assigned)} for all unique evaluated architectures
        self.evaluated_architectures = {} 
        self.checkpoint_path = checkpoint_path
        self.next_model_id = 0 # Counter for unique IDs, updated dynamically

    def _create_random_chromosome(self):
        """Creates a single random chromosome."""
        return {key: random.choice(values) for key, values in self.search_space.items()}

    def _initialize_population(self):
        """
        Creates the initial population of random individuals.
        Assigns unique IDs, reusing IDs for historical duplicates.
        """
        # Determine the starting ID for new models
        if self.evaluated_architectures:
            # Find the maximum ID already assigned in history
            self.next_model_id = max(id_val for _, id_val in self.evaluated_architectures.values()) + 1
        else:
            self.next_model_id = 0 # Start from 0 if no history

        new_population_list = []
        for _ in range(self.population_size):
            chromosome = self._create_random_chromosome()
            chromosome_hashable = tuple(sorted(chromosome.items()))
            
            if chromosome_hashable in self.evaluated_architectures:
                # Duplicate in history, reuse its fitness and original ID
                fitness, model_id = self.evaluated_architectures[chromosome_hashable]
            else:
                # New chromosome, assign a truly unique ID
                model_id = self.next_model_id
                self.next_model_id += 1 # Increment for the next new unique model
                fitness = None # Will be evaluated
            
            new_population_list.append({'chromosome': chromosome, 'fitness': fitness, 'id': model_id})
        
        self.population = new_population_list
        print(f"Initialized population with {self.population_size} individuals. Next unique ID available: {self.next_model_id}")

    def _save_checkpoint(self, generation_num):
        """Saves the current state to a file."""
        state = {
            'generation': generation_num,
            'population': self.population,
            'evaluated_architectures': self.evaluated_architectures,
            'next_model_id': self.next_model_id
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"--- Checkpoint saved for end of Generation {generation_num} ---")

    def _load_checkpoint(self):
        """Loads state from a checkpoint file if it exists."""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            
            start_gen = state.get('generation', 0)
            self.population = state.get('population', [])
            self.evaluated_architectures = state.get('evaluated_architectures', {})
            self.next_model_id = state.get('next_model_id', 0)

            # Ensure evaluated_architectures entries are in the (fitness, id) tuple format
            # This handles loading older checkpoints where only fitness might have been stored
            temp_evaluated_architectures = {}
            max_id_in_loaded_history = -1
            if self.evaluated_architectures:
                for chromo_tuple, value in self.evaluated_architectures.items():
                    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], float) and isinstance(value[1], int):
                        temp_evaluated_architectures[chromo_tuple] = value
                        max_id_in_loaded_history = max(max_id_in_loaded_history, value[1])
                    else: # Old format, just fitness. Assign a dummy new ID for historical continuity
                        # This scenario should ideally not be hit with updated saving.
                        print(f"Warning: Converting old evaluated_architectures format for {chromo_tuple}")
                        max_id_in_loaded_history += 1 # Increment for this old entry
                        temp_evaluated_architectures[chromo_tuple] = (value, max_id_in_loaded_history)
                        
            self.evaluated_architectures = temp_evaluated_architectures
            
            # Ensure self.next_model_id is correctly set after loading history
            if self.evaluated_architectures:
                self.next_model_id = max(self.next_model_id, max_id_in_loaded_history + 1)
            else:
                self.next_model_id = 0 # No history, start ID from 0

            print(f"--- Resuming from checkpoint at start of Generation {start_gen + 1} ---")
            print(f"Total unique architectures evaluated so far: {len(self.evaluated_architectures)}")
            print(f"Next available model ID: {self.next_model_id}")
            return start_gen, self.population
        return 0, None  # Start from scratch if no checkpoint

    def _crossover(self, parent1_chromo, parent2_chromo):
        """Performs single-point crossover between two parent chromosomes."""
        child_chromo = {}
        genes = list(self.search_space.keys())
        crossover_point = random.randint(1, len(genes) - 1)
        for i, gene in enumerate(genes):
            if i < crossover_point:
                child_chromo[gene] = parent1_chromo[gene]
            else:
                child_chromo[gene] = parent2_chromo[gene]
        return child_chromo

    def _mutate(self, chromosome):
        """Mutates a chromosome by randomly changing some of its genes."""
        mutated_chromo = chromosome.copy()
        for gene, values in self.search_space.items():
            if random.random() < self.mutation_rate:
                current_value = mutated_chromo[gene]
                possible_values = [v for v in values if v != current_value]
                if possible_values:
                    mutated_chromo[gene] = random.choice(possible_values)
        return mutated_chromo

    def _selection(self):
        """Selects a single individual using tournament selection."""
        tournament_size = 3
        # Ensure we don't try to sample more individuals than exist in the population
        sample_size = min(tournament_size, len(self.population))
        competitors = random.sample(self.population, sample_size)
        # Filter out any individuals with None fitness before sorting
        valid_competitors = [ind for ind in competitors if ind['fitness'] is not None]
        if not valid_competitors:
             # Fallback: if all sampled competitors have None fitness, return a random one
             return random.choice(self.population)
        winner = sorted(valid_competitors, key=lambda x: x['fitness'], reverse=True)[0]
        return winner

    def run(self, num_generations, parallel_evaluator=None):
        """
        The main loop that runs the entire evolutionary process.
        Args:
            num_generations (int): The number of generations to evolve.
            parallel_evaluator (function, optional): A function to evaluate multiple individuals in parallel.
        """
        start_gen, loaded_population = self._load_checkpoint()
        if loaded_population:
            self.population = loaded_population
        else:
            self._initialize_population()

        best_ever_individual = None
        # Find best ever from loaded evaluated_architectures history
        if self.evaluated_architectures:
            # max() on dictionary items (key=value) with key=value.get (for nested tuple access)
            best_chromo_tuple = max(self.evaluated_architectures, key=lambda k: self.evaluated_architectures[k][0])
            best_fitness = self.evaluated_architectures[best_chromo_tuple][0] # Access the fitness
            best_id_from_history = self.evaluated_architectures[best_chromo_tuple][1] # Access the ID
            
            # Create a simple individual dict for the best historical one for display
            best_ever_individual = {'chromosome': dict(best_chromo_tuple), 'fitness': best_fitness, 'id': best_id_from_history}
            print(f"Loaded best historical fitness: {best_fitness:.4f}% (ID: {best_id_from_history})")

        for gen in range(start_gen, num_generations):
            print(f"\n===== Generation {gen + 1}/{num_generations} =====")

            # 1. Fitness Evaluation
            individuals_to_evaluate = []
            for individual in self.population:
                chromosome_hashable = tuple(sorted(individual['chromosome'].items()))
                
                if chromosome_hashable in self.evaluated_architectures:
                    # Duplicate found, use cached fitness and ID
                    individual['fitness'] = self.evaluated_architectures[chromosome_hashable][0] # Get fitness
                    # The ID is already assigned in _initialize_population or child creation
                    # print(f"  - Model ID {individual['id']} (DUPLICATE) skipped evaluation, fitness: {individual['fitness']:.4f}")
                else:
                    individuals_to_evaluate.append(individual)

            print(f"Evaluating fitness for {len(individuals_to_evaluate)} new (non-duplicate) individuals...")

            if individuals_to_evaluate:
                # Evaluate new individuals. results_map is a list of (id, fitness)
                results_map = parallel_evaluator(individuals_to_evaluate) 
                
                for result_id, result_fitness in results_map:
                    for individual in individuals_to_evaluate:
                        if individual['id'] == result_id:
                            individual['fitness'] = result_fitness
                            # Store in history as (fitness, id) tuple
                            self.evaluated_architectures[tuple(sorted(individual['chromosome'].items()))] = (result_fitness, individual['id'])
                            break
            else:
                print("No new individuals to evaluate this generation (all were duplicates/cached).")

            # 2. Sort the current population by fitness (highest first)
            # Filter out any with None fitness (shouldn't happen with proper evaluation or caching)
            self.population.sort(key=lambda x: x['fitness'] if x['fitness'] is not None else -1, reverse=True)
            
            # 3. Update the best ever individual (from current population or history)
            if self.population and self.population[0]['fitness'] is not None:
                current_gen_best = self.population[0]
                if best_ever_individual is None or current_gen_best['fitness'] > best_ever_individual['fitness']:
                    best_ever_individual = current_gen_best.copy() # Use .copy() to be safe
                    print(f"*** New best overall fitness found: {best_ever_individual['fitness']:.4f}% (ID: {best_ever_individual['id']}) ***")

            # Print best of current generation (guaranteed to be evaluated or from cache)
            if self.population and self.population[0]['fitness'] is not None:
                print(f"Best fitness in Generation {gen + 1}: {self.population[0]['fitness']:.4f}% (ID: {self.population[0]['id']})")
            else:
                print(f"No valid fitness found in Generation {gen + 1} (all might have failed or were not evaluated).")


            # 4. Create the next generation
            next_generation = []
            if self.elitism_count > 0:
                elites = self.population[:self.elitism_count]
                next_generation.extend(elites)

            num_children_to_create = self.population_size - len(next_generation)
            for _ in range(num_children_to_create):
                parent1 = self._selection()
                parent2 = self._selection()
                child_chromosome = self._crossover(parent1['chromosome'], parent2['chromosome'])
                mutated_child_chromosome = self._mutate(child_chromosome)
                
                # Check if this newly created/mutated chromosome is a duplicate in our history
                child_chromosome_hashable = tuple(sorted(mutated_child_chromosome.items()))
                
                if child_chromosome_hashable in self.evaluated_architectures:
                    # Duplicate found in history, use its cached fitness and original ID
                    fitness, existing_id = self.evaluated_architectures[child_chromosome_hashable]
                    child_to_add = {'chromosome': mutated_child_chromosome, 'fitness': fitness, 'id': existing_id}
                else:
                    # New chromosome, assign a truly unique ID
                    model_id_for_child = self.next_model_id
                    self.next_model_id += 1 # Increment for the next new unique model
                    child_to_add = {'chromosome': mutated_child_chromosome, 'fitness': None, 'id': model_id_for_child}

                next_generation.append(child_to_add)

            self.population = next_generation

            # 5. Save a checkpoint at the end of the generation
            self._save_checkpoint(generation_num=gen + 1)
        
        print("\n===== Evolution Complete =====")
        print(f"Total unique architectures evaluated during this run: {len(self.evaluated_architectures)}")
        return best_ever_individual