"""
Genetic Algorithm Hyperparameter Tuner for NeuroGraph
Optimizes discrete neural network hyperparameters through evolutionary search
"""

import os
import sys
import yaml
import json
import random
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Union, Any
import logging

# Add current directory to path for NeuroGraph imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# NeuroGraph imports
from train.modular_train_context import create_modular_train_context


class GeneticHyperparameterTuner:
    """
    Genetic Algorithm for optimizing NeuroGraph hyperparameters.
    
    Optimizes 10 key hyperparameters through evolutionary search with
    actual training runs for fitness evaluation.
    """
    
    def __init__(self, crossover_rate=0.3, mutation_rate=0.2, elite_percentage=0.5):
        """
        Initialize the genetic algorithm tuner.
        
        Args:
            crossover_rate: Probability of crossover between parents (0.0 to 1.0)
            mutation_rate: Probability of mutation per gene (0.0 to 1.0)
            elite_percentage: Percentage of population to preserve as elite (0.0 to 1.0)
        """
        # Validate parameters
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError(f"crossover_rate must be between 0.0 and 1.0, got {crossover_rate}")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(f"mutation_rate must be between 0.0 and 1.0, got {mutation_rate}")
        if not (0.0 <= elite_percentage <= 1.0):
            raise ValueError(f"elite_percentage must be between 0.0 and 1.0, got {elite_percentage}")
        
        # Hyperparameter search spaces
        self.search_space = {
            'vector_dim': [5, 8, 10],
            'phase_bins': [16, 32, 64, 128],
            'mag_bins': [64, 128, 256, 512, 1024],
            'cardinality': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'decay_factor': [0.9, 0.925, 0.95, 0.975],
            'orthogonality_threshold': [0.05, 0.1, 0.15, 0.2],
            'warmup_epochs': [3, 5, 8, 10],
            'min_activation_strength': [0.01, 0.05, 0.1, 0.2, 0.5],
            'batch_size': [3, 5, 8, 10]
        }
        
        # Fixed parameters
        self.fixed_params = {
            'accumulation_steps': 8,
            'num_epochs': 50,
            'validation_samples': 500
        }
        
        # GA parameters (user-configurable)
        self.tournament_size = 3
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_percentage = elite_percentage
        
        # Setup logging
        self.setup_logging()
        
        # Log GA configuration
        self.logger.info(f"GA Parameters: crossover_rate={crossover_rate}, "
                        f"mutation_rate={mutation_rate}, elite_percentage={elite_percentage}")
        
    def setup_logging(self):
        """Setup logging for GA progress tracking."""
        log_dir = "logs/genetic_algorithm"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/ga_tuning_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Genetic Algorithm Hyperparameter Tuner initialized")
        
    def generate_individual(self) -> Dict[str, Any]:
        """
        Generate a random individual (hyperparameter configuration).
        
        Returns:
            Dictionary containing random hyperparameter values
        """
        individual = {}
        for param_name, param_space in self.search_space.items():
            individual[param_name] = random.choice(param_space)
        
        return individual
    
    def create_neurograph_config(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a complete NeuroGraph configuration from GA individual.
        
        Args:
            individual: Dictionary of hyperparameter values
            
        Returns:
            Complete NeuroGraph configuration dictionary
        """
        # Calculate derived parameters
        total_nodes = 200 + 10 + 790  # input + output + intermediate
        input_nodes = 200
        output_nodes = 10
        
        # Build complete configuration
        config = {
            'mode': 'modular',
            'device': 'cuda',
            
            # Architecture parameters
            'architecture': {
                'total_nodes': total_nodes,
                'input_nodes': input_nodes,
                'output_nodes': output_nodes,
                'vector_dim': individual['vector_dim'],
                'seed': 42
            },
            
            # Resolution parameters
            'resolution': {
                'phase_bins': individual['phase_bins'],
                'mag_bins': individual['mag_bins'],
                'resolution_increase': individual['phase_bins'] * individual['mag_bins'] // 64  # Relative to base
            },
            
            # Graph structure
            'graph_structure': {
                'cardinality': individual['cardinality']
            },
            
            # Training parameters
            'training': {
                'gradient_accumulation': {
                    'enabled': True,
                    'accumulation_steps': self.fixed_params['accumulation_steps'],
                    'lr_scaling': 'sqrt',
                    'buffer_size': 1500
                },
                'optimizer': {
                    'base_learning_rate': individual['learning_rate'],
                    'effective_learning_rate': individual['learning_rate'] * (self.fixed_params['accumulation_steps'] ** 0.5),
                    'num_epochs': self.fixed_params['num_epochs'],
                    'warmup_epochs': individual['warmup_epochs'],
                    'batch_size': individual['batch_size']
                }
            },
            
            # Forward pass parameters
            'forward_pass': {
                'max_timesteps': 50,
                'decay_factor': individual['decay_factor'],
                'min_activation_strength': individual['min_activation_strength'],
                'use_radiation': True,
                'top_k_neighbors': 4
            },
            
            # Input processing
            'input_processing': {
                'adapter_type': 'linear_projection',
                'input_dim': 784,
                'normalization': 'layer_norm',
                'dropout': 0.1,
                'learnable': True
            },
            
            # Class encoding
            'class_encoding': {
                'type': 'orthogonal',
                'num_classes': 10,
                'encoding_dim': individual['vector_dim'],
                'orthogonality_threshold': individual['orthogonality_threshold']
            },
            
            # Loss function
            'loss_function': {
                'type': 'categorical_crossentropy',
                'temperature': 1.0,
                'label_smoothing': 0.0
            },
            
            # Batch evaluation
            'batch_evaluation': {
                'enabled': True,
                'batch_size': 16,
                'streaming': True
            },
            
            # Debugging
            'debugging': {
                'evaluation_samples': self.fixed_params['validation_samples'],
                'final_evaluation_samples': self.fixed_params['validation_samples'],
                'log_level': 'INFO'
            },
            
            # Paths (will be set dynamically)
            'paths': {
                'graph_path': None,  # Will be set in evaluate_fitness
                'training_curves_path': None,
                'checkpoint_path': None
            }
        }
        
        return config
    
    def evaluate_fitness(self, individual: Dict[str, Any]) -> float:
        """
        Evaluate fitness of an individual through actual NeuroGraph training.
        
        Args:
            individual: Dictionary of hyperparameter values
            
        Returns:
            Validation accuracy as fitness score (0.0 to 1.0)
        """
        temp_dir = None
        try:
            # Create temporary directory for this evaluation
            temp_dir = tempfile.mkdtemp(prefix="ga_eval_")
            
            # Create NeuroGraph configuration
            config = self.create_neurograph_config(individual)
            
            # Set temporary paths
            config['paths']['graph_path'] = os.path.join(temp_dir, "temp_graph.pkl")
            config['paths']['training_curves_path'] = os.path.join(temp_dir, "curves.png")
            config['paths']['checkpoint_path'] = os.path.join(temp_dir, "checkpoints/")
            
            # Save temporary config file
            config_path = os.path.join(temp_dir, "temp_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Log evaluation start
            param_str = ", ".join([f"{k}={v}" for k, v in individual.items()])
            self.logger.info(f"Evaluating individual: {param_str}")
            
            # Create and train NeuroGraph model
            trainer = create_modular_train_context(config_path)
            
            # Run training
            losses = trainer.train()
            
            # Evaluate validation accuracy
            validation_accuracy = trainer.evaluate_accuracy(
                num_samples=self.fixed_params['validation_samples'],
                use_batch_evaluation=True
            )
            
            self.logger.info(f"Individual fitness: {validation_accuracy:.4f}")
            
            return validation_accuracy
            
        except Exception as e:
            self.logger.error(f"Error evaluating individual: {e}")
            # Return low fitness for failed evaluations
            return 0.0
            
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")
    
    def tournament_selection(self, population: List[Dict[str, Any]], 
                           fitness_scores: List[float]) -> Dict[str, Any]:
        """
        Select an individual using tournament selection.
        
        Args:
            population: List of individuals
            fitness_scores: Corresponding fitness scores
            
        Returns:
            Selected individual
        """
        # Select random individuals for tournament
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        # Find winner (highest fitness)
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        
        return population[winner_idx].copy()
    
    def uniform_crossover(self, parent1: Dict[str, Any], 
                         parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform uniform crossover between two parents.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Tuple of two offspring individuals
        """
        offspring1 = {}
        offspring2 = {}
        
        for param_name in self.search_space.keys():
            if random.random() < 0.5:
                # Take from parent1
                offspring1[param_name] = parent1[param_name]
                offspring2[param_name] = parent2[param_name]
            else:
                # Take from parent2
                offspring1[param_name] = parent2[param_name]
                offspring2[param_name] = parent1[param_name]
        
        return offspring1, offspring2
    
    def mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate an individual by randomly changing some parameters.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        for param_name, param_space in self.search_space.items():
            if random.random() < self.mutation_rate:
                # Mutate this parameter
                mutated[param_name] = random.choice(param_space)
        
        return mutated
    
    def select_top_k(self, population: List[Dict[str, Any]], 
                     fitness_scores: List[float], k: int) -> List[Dict[str, Any]]:
        """
        Select top-k individuals based on fitness scores.
        
        Args:
            population: List of individuals
            fitness_scores: Corresponding fitness scores
            k: Number of top individuals to select
            
        Returns:
            List of top-k individuals
        """
        # Create pairs of (individual, fitness) and sort by fitness
        paired = list(zip(population, fitness_scores))
        paired.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness (descending)
        
        # Return top-k individuals
        return [individual for individual, _ in paired[:k]]
    
    def genetic_hyperparam_search(self, config_input: Union[str, Dict], 
                                 generations: int = 10, 
                                 population_size: int = 50, 
                                 top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Main genetic algorithm hyperparameter search function.
        
        Args:
            config_input: Base configuration (dict or YAML file path) - currently unused
            generations: Number of generations to evolve
            population_size: Size of population per generation
            top_k: Number of best configurations to return
            
        Returns:
            List of top-k best hyperparameter configurations with fitness scores
        """
        self.logger.info(f"Starting GA hyperparameter search:")
        self.logger.info(f"  Generations: {generations}")
        self.logger.info(f"  Population size: {population_size}")
        self.logger.info(f"  Top-k: {top_k}")
        self.logger.info(f"  Search space: {len(self.search_space)} parameters")
        
        # Initialize population
        self.logger.info("Initializing random population...")
        population = [self.generate_individual() for _ in range(population_size)]
        
        # Track best individuals across generations
        all_time_best = []
        generation_stats = []
        
        # Evolution loop
        for generation in range(generations):
            self.logger.info(f"\n=== Generation {generation + 1}/{generations} ===")
            
            # Evaluate fitness for all individuals
            self.logger.info("Evaluating population fitness...")
            fitness_scores = []
            
            for i, individual in enumerate(population):
                self.logger.info(f"Evaluating individual {i + 1}/{population_size}")
                fitness = self.evaluate_fitness(individual)
                fitness_scores.append(fitness)
            
            # Track statistics
            best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            worst_fitness = min(fitness_scores)
            
            # Find best individual of this generation
            best_idx = fitness_scores.index(best_fitness)
            best_individual = population[best_idx].copy()
            best_individual['fitness'] = best_fitness
            best_individual['generation'] = generation + 1
            
            # Update all-time best
            all_time_best.append(best_individual)
            
            # Log generation statistics
            self.logger.info(f"Generation {generation + 1} Results:")
            self.logger.info(f"  Best fitness: {best_fitness:.4f}")
            self.logger.info(f"  Average fitness: {avg_fitness:.4f}")
            self.logger.info(f"  Worst fitness: {worst_fitness:.4f}")
            self.logger.info(f"  Best config: {best_individual}")
            
            # Store generation statistics
            generation_stats.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'worst_fitness': worst_fitness,
                'best_individual': best_individual
            })
            
            # Create next generation (except for last generation)
            if generation < generations - 1:
                self.logger.info("Creating next generation...")
                new_population = []
                
                # Elitism: Keep best individuals based on user-specified percentage
                elite_count = max(1, int(population_size * self.elite_percentage))
                elite_individuals = self.select_top_k(population, fitness_scores, elite_count)
                new_population.extend(elite_individuals)
                
                self.logger.info(f"Preserving {elite_count} elite individuals ({self.elite_percentage:.1%})")
                
                # Generate offspring through crossover and mutation
                while len(new_population) < population_size:
                    # Selection
                    parent1 = self.tournament_selection(population, fitness_scores)
                    parent2 = self.tournament_selection(population, fitness_scores)
                    
                    # Crossover
                    if random.random() < self.crossover_rate:
                        offspring1, offspring2 = self.uniform_crossover(parent1, parent2)
                    else:
                        offspring1, offspring2 = parent1.copy(), parent2.copy()
                    
                    # Mutation
                    offspring1 = self.mutate(offspring1)
                    offspring2 = self.mutate(offspring2)
                    
                    # Add to new population
                    new_population.extend([offspring1, offspring2])
                
                # Trim to exact population size
                population = new_population[:population_size]
        
        # Select top-k best individuals from all generations
        self.logger.info(f"\nSelecting top-{top_k} individuals from all generations...")
        all_fitness = [ind['fitness'] for ind in all_time_best]
        top_individuals = self.select_top_k(all_time_best, all_fitness, top_k)
        
        # Save results
        self.save_results(top_individuals, generation_stats)
        
        # Log final results
        self.logger.info(f"\n=== Final Results ===")
        for i, individual in enumerate(top_individuals):
            self.logger.info(f"Rank {i + 1}: Fitness = {individual['fitness']:.4f}, "
                           f"Generation = {individual['generation']}")
        
        return top_individuals
    
    def save_results(self, top_individuals: List[Dict[str, Any]], 
                     generation_stats: List[Dict[str, Any]]):
        """
        Save GA results to JSON files.
        
        Args:
            top_individuals: Top-k best individuals
            generation_stats: Statistics for each generation
        """
        results_dir = "results/genetic_algorithm"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save top individuals
        top_results_file = f"{results_dir}/top_individuals_{timestamp}.json"
        with open(top_results_file, 'w') as f:
            json.dump(top_individuals, f, indent=2, default=str)
        
        # Save generation statistics
        stats_file = f"{results_dir}/generation_stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(generation_stats, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to:")
        self.logger.info(f"  Top individuals: {top_results_file}")
        self.logger.info(f"  Generation stats: {stats_file}")


# Convenience function for direct usage
def genetic_hyperparam_search(config_input: Union[str, Dict], 
                             generations: int = 10, 
                             population_size: int = 50, 
                             top_k: int = 5,
                             crossover_rate: float = 0.3,
                             mutation_rate: float = 0.2,
                             elite_percentage: float = 0.5) -> List[Dict[str, Any]]:
    """
    Genetic Algorithm hyperparameter search for NeuroGraph.
    
    Args:
        config_input: Base configuration (dict or YAML file path) - currently unused
        generations: Number of generations to evolve
        population_size: Size of population per generation  
        top_k: Number of best configurations to return
        crossover_rate: Probability of crossover between parents (0.0 to 1.0)
        mutation_rate: Probability of mutation per gene (0.0 to 1.0)
        elite_percentage: Percentage of population to preserve as elite (0.0 to 1.0)
        
    Returns:
        List of top-k best hyperparameter configurations with fitness scores
    """
    tuner = GeneticHyperparameterTuner(
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elite_percentage=elite_percentage
    )
    return tuner.genetic_hyperparam_search(config_input, generations, population_size, top_k)


if __name__ == "__main__":
    # Example usage
    print("NeuroGraph Genetic Algorithm Hyperparameter Tuner")
    print("=" * 50)
    
    # Run GA with small population for testing
    results = genetic_hyperparam_search(
        config_input={},  # Not used currently
        generations=3,
        population_size=10,
        top_k=3
    )
    
    print(f"\nTop {len(results)} configurations found:")
    for i, config in enumerate(results):
        print(f"\nRank {i + 1}:")
        print(f"  Fitness: {config['fitness']:.4f}")
        print(f"  Generation: {config['generation']}")
        print(f"  Parameters: {config}")
