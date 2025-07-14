# ga_engine.py

import random
import pickle
import os
from tqdm import tqdm

class GeneticAlgorithm:
    """
    This class implements the core logic for the genetic algorithm.
    It is designed to be general, resilient with checkpointing, and handles
    the main evolutionary loop.
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
        self.population = []  # This will hold our list of individuals
        self.checkpoint_path = checkpoint_path

    def _create_random_chromosome(self):
        """Creates a single random chromosome."""
        return {key: random.choice(values) for key, values in self.search_space.items()}

    def _initialize_population(self):
        """Creates the initial population of random individuals."""
        self.population = [
            {'chromosome': self._create_random_chromosome(), 'fitness': None, 'id': i}
            for i in range(self.population_size)
        ]
        print(f"Initialized population with {self.population_size} individuals.")

    def _save_checkpoint(self, generation_num):
        """Saves the current state (generation number and population) to a file."""
        state = {
            'generation': generation_num,
            'population': self.population
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
            population = state.get('population', [])
            print(f"--- Resuming from checkpoint at start of Generation {start_gen + 1} ---")
            return start_gen, population
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

    def run(self, num_generations, fitness_function, parallel_evaluator=None):
        """
        The main loop that runs the entire evolutionary process.
        Args:
            num_generations (int): The number of generations to evolve.
            fitness_function (function): A function that takes a chromosome and returns a fitness score.
            parallel_evaluator (function, optional): A function to evaluate multiple individuals in parallel.
        """
        start_gen, loaded_population = self._load_checkpoint()
        if loaded_population:
            self.population = loaded_population
        else:
            self._initialize_population()

        best_ever_individual = None
        # Find best ever from loaded population, if any
        if loaded_population:
            valid_pop = [ind for ind in loaded_population if ind['fitness'] is not None]
            if valid_pop:
                best_ever_individual = sorted(valid_pop, key=lambda x: x['fitness'], reverse=True)[0]


        for gen in range(start_gen, num_generations):
            print(f"\n===== Generation {gen + 1}/{num_generations} =====")

            # 1. Fitness Evaluation
            individuals_to_evaluate = [ind for ind in self.population if ind['fitness'] is None]
            print(f"Evaluating fitness for {len(individuals_to_evaluate)} new individuals...")

            if individuals_to_evaluate:
                if parallel_evaluator:
                    # Use the parallel evaluator if provided
                    results = parallel_evaluator(individuals_to_evaluate)
                    for i, ind in enumerate(individuals_to_evaluate):
                        ind['fitness'] = results[i]
                else:
                    # Fallback to serial evaluation
                    for ind in tqdm(individuals_to_evaluate):
                        ind['fitness'] = fitness_function(ind)

            # 2. Sort the population by fitness
            valid_population = [ind for ind in self.population if ind['fitness'] is not None]
            invalid_population = [ind for ind in self.population if ind['fitness'] is None]
            valid_population.sort(key=lambda x: x['fitness'], reverse=True)
            self.population = valid_population + invalid_population

            # 3. Update the best ever individual
            if valid_population:
                current_best = self.population[0]
                if best_ever_individual is None or current_best['fitness'] > best_ever_individual['fitness']:
                    best_ever_individual = current_best.copy()
                    print(f"*** New best overall fitness found: {best_ever_individual['fitness']:.4f} ***")

            print(f"Best fitness in Generation {gen + 1}: {self.population[0]['fitness']:.4f}")

            # 4. Create the next generation
            next_generation = []
            if self.elitism_count > 0:
                elites = self.population[:self.elitism_count]
                next_generation.extend(elites)

            num_children = self.population_size - len(next_generation)
            for i in range(num_children):
                parent1 = self._selection()
                parent2 = self._selection()
                child_chromosome = self._crossover(parent1['chromosome'], parent2['chromosome'])
                mutated_child_chromosome = self._mutate(child_chromosome)
                # Assign a unique ID for the new child
                new_id = max([ind['id'] for ind in self.population] + [-1]) + 1
                next_generation.append({'chromosome': mutated_child_chromosome, 'fitness': None, 'id': new_id})

            self.population = next_generation

            # 5. Save a checkpoint at the end of the generation
            self._save_checkpoint(generation_num=gen + 1)

        print("\n===== Evolution Complete =====")
        return best_ever_individual