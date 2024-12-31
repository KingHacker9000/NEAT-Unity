from abc import ABC, abstractmethod
from NEAT_Unity import Genome, NEAT, NEATManager, SamplingDist
import math
import random

class Simulation(ABC):
    """
    Abstract base class for running simulations and setting fitness values for NEAT genomes.
    """

    def __init__(self):
        """
        Initialize the simulation with the given environment.
        """
        pass

    @abstractmethod
    def run_simulation(self, neat_manager: NEATManager):
        pass

    @abstractmethod
    def run_frame(self):
        pass


class FunctionApproximationSimulation(Simulation):
    """
    Attempts to approximate sin(2πx) for x in [0,1].
    Uses several sampled points and tries to minimize MSE.
    """
    
    def __init__(self, input_size=2, output_size=1, population_size=150, c1=1, c2=1, c3=0.4, dt=5, activation='relu'):
        # We assume we have 1 input node + 1 bias node = total 2 inputs
        # 1 output node
        # You can add the bias node explicitly by always feeding '1' as second input.
        self.neat = NEAT(input_size, output_size, population_size, c1, c2, c3, dt, SamplingDist(), activation)
        self.neat_manager = NEATManager(self.neat)
        
        # Generate a set of training points
        self.training_points = [random.random() for _ in range(500)]  # 20 random points in [0,1]

    def run_simulation(self):
        best_mse = float('inf')
        best_genome = self.neat.population[0]
        generation = 0

        while best_mse > 0.01 and generation < 500:
            # Reset fitness
            for genome in self.neat.population:
                genome.set_fitness(0)

            # Run a frame which assigns fitness
            self.run_frame()

            # Identify best genome and its mse
            curr_best_mse, curr_best_genome = self.get_best_mse_genome()
            if curr_best_mse < best_mse:
                best_mse = curr_best_mse
                best_genome = curr_best_genome

            print("-" * 40)
            print(f"Generation: {generation}")
            print(f"Best MSE: {best_mse:.5f}")
            print(f"Best Fitness: {best_genome.fitness}")
            print(f"Best Genome: {best_genome}")
            print(f"Species Count: {len(self.neat.species)}")
            print("-" * 40)

            self.neat_manager.next_generation()
            generation += 1

        return best_genome, best_mse

    def run_frame(self):
        """
        Evaluate each genome on the set of training points.
        Fitness is calculated as: fitness = 1/(1+MSE)
        MSE = average((predicted - actual)^2) over training points
        """
        for genome in self.neat.population:
            errors = []
            for x in self.training_points:
                # Input: [x, 1.0] (1.0 acts as bias)
                y_pred = genome.forward_pass([x, 1.0])[0]
                y_true = math.sin(2 * math.pi * x)
                errors.append(abs(y_true - y_pred))
            mse = sum(errors) / len(errors)
            fitness = 1/(1+mse)
            genome.set_fitness(fitness)

    def get_best_mse_genome(self) -> tuple[float, Genome]:
        best_mse = float('inf')
        best_genome = None
        for genome in self.neat.population:
            # Reverse the fitness calculation to get MSE
            # fitness = 1/(1+MSE) => MSE = (1/fitness)-1
            if genome.fitness > 0:
                mse = (1/genome.fitness) - 1
            else:
                mse = float('inf')

            if mse < best_mse:
                best_mse = mse
                best_genome = genome
        return best_mse, best_genome

    
class XORSimulation(Simulation):

    def __init__(self):
        self.neat = NEAT(3, 2, 150, 1, 1, 0.4, 40, SamplingDist(), 'relu')
        self.neat_manager = NEATManager(self.neat)

    def run_simulation(self):
        best_score = 0
        best_genome = self.neat.population[0]
        i = 0

        while best_score <  4:
            for inputs in [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]:
                self.run_frame(inputs)
            
            best_score, best_genome = self.get_best(best_score, best_genome)
            print("-"* 40)
            print("Generation:", i)
            print("Best Score:", best_score)
            print("Best Genome:", best_genome)
            print("Species Count:", len(self.neat.species))
            print("-"* 40)
            self.neat_manager.next_generation()
            i += 1

        return best_genome, best_score

    def get_best(self, curr_best_score: float, curr_best_genome: Genome) -> tuple[float, Genome]:
        best_score = curr_best_score
        best_genome = curr_best_genome

        for genome in self.neat.population:
            if best_score < genome.fitness:
                best_score = genome.fitness
                best_genome = genome

        return (best_score, best_genome)

    def run_frame(self, inputs):
        correct_output = inputs[0] ^ inputs[1]
        for genome in self.neat.population:
            outputs = genome.forward_pass(inputs)
            if outputs[0] > outputs[1] and correct_output == 0:
                genome.add_fitness(1)
            elif outputs[1] > outputs[0] and correct_output == 1:
                genome.add_fitness(1)
            

# Example usage
if __name__ == "__main__":
    import pickle

    sim = FunctionApproximationSimulation()
    best_genome, best_mse = sim.run_simulation()
    # with open('Genomes/brain_funcApprox.pkl', 'wb') as f:
    #     pickle.dump(best_genome, f)
    print("Final Best MSE:", best_mse)
    print("Final Best Genome:", best_genome)
