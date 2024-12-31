from __future__ import annotations
import random, math
from activation_functions import ActivationFunction, get_activation_function

INPUT_LAYER = 0
OUTPUT_LAYER = -1
DEBUG = False


def random_non_uniform_choice(l: list[Genome]) -> Genome:
    assert len(l) > 0, "List is Empty"

    total = sum(genome.adjusted_fitness for genome in l)

    if total == 0:
        return random.choice(l)

    prob = [genome.adjusted_fitness/total for genome in l]
    cutoff = [sum(prob[:i+1]) for i in range(len(prob))] ### Optimize Cumilative Sum

    r  = random.random()

    for i in range(len(l)):
        if r <= cutoff[i]:
            return l[i]
    
    if cutoff[-1] < 0.99:
        print("cutoff:", cutoff)
        raise ValueError
    
    return l[-1]

class NodeGene: 
    """
    Contains:
        - value: float
        - innovation: int
    """

    def __init__(self, innovation: int) -> None:
        self.value = 0
        self.innovation: int = innovation


    def get_node(self, l:list[NodeGene], create: bool = True) -> NodeGene:
        for node in l:
            if node.innovation == self.innovation:
                return node
        
        if create:
            node = NodeGene(self.innovation)
            l.append(node)
            return node
        
        # Node is not Found
        raise ValueError
    
    def evaluate_activation(self, activation_func: ActivationFunction) -> None:
        self.value = activation_func.forward(self.value)

    def reset_value(self) -> None:
        self.value = 0

    def set_value(self, value: float) -> None:
        self.value = value
    
    def add_value(self, value: float) -> None:
        self.value += value

class ConnectionGene:
    """
    Contains:
        - in_node: NodeGene
        - out_node: NodeGene
        - weight: float
        - innovation: int
        - enabled: bool
    """

    def __init__(self, in_node: NodeGene, out_node: NodeGene, innovation: int, weight: float = None, enabled: bool = True) -> None:
        
        self.in_node: NodeGene = in_node
        self.out_node: NodeGene = out_node
        self.weight: float = random.uniform(-1, 1) if weight is None else weight
        self.innovation: int = innovation
        self.enabled: bool = enabled

    def evaluate(self) -> None:
        """Evaluate the Value and Add to the Output Node"""
        self.out_node.add_value(self.in_node.value * self.weight)

class Genome:
    """
    Contains List of Connection Genes
    """

    def __init__(self, activation_func: ActivationFunction) -> None:
        
        self.connection_genes: list[ConnectionGene] = []
        self.nodes: list[list[NodeGene]] = [[], []]
        self.fitness: float = 0
        self.adjusted_fitness: float = None
        self.activation_func: ActivationFunction = activation_func

    def __repr__(self) -> str:
        # For convenience, determine how many layers we have
        # Typically, NEAT can have multiple hidden layers if evolved over time.
        # We assume self.nodes is a list of layers, with indices from 0 to len(self.nodes)-1
        # The OUTPUT_LAYER is defined as -1, but we handle that indexing logic here.
        layer_count = len(self.nodes)
        if layer_count < 2:
            # At least two layers are expected: input and output
            return "Genome with insufficient layer structure"

        # Prepare a layer-wise representation
        # Each node is represented as N(innovation)
        layer_strs = []
        for i, layer in enumerate(self.nodes):
            nodes_str = " ".join(f"N({node.innovation})" for node in layer)
            layer_strs.append(nodes_str if nodes_str else "--Empty--")

        # Now represent connections
        # We'll display them as transitions from left layer to right layer
        # For each pair of adjacent layers, show connections that cross them
        # Note: In a fully connected feedforward layout, connections flow from lower-indexed layers to higher-indexed layers,
        # but NEAT can sometimes create recurrent or cross-layer connections. We'll just show them where they lead.
        
        # First, map node innovation to (layer_index, node_index_in_layer)
        node_positions = {}
        for li, layer in enumerate(self.nodes):
            for ni, node in enumerate(layer):
                node_positions[node.innovation] = (li, ni)

        # Sort connections by their out_node layer for more structured visualization
        # This will help group connections by where they go.
        # But be aware NEAT doesn't strictly enforce layered feedforward structure; some connections may jump multiple layers.
        connections_by_out_layer = {}
        for conn in self.connection_genes:
            out_layer, _ = node_positions[conn.out_node.innovation]
            connections_by_out_layer.setdefault(out_layer, []).append(conn)

        # Construct a multiline string
        # 1) Print the layers
        # 2) For each layer (except the last), print connections to subsequent layers if any
        lines = []
        for li in range(layer_count):
            # Print this layer
            lines.append(f"L{li}: " + layer_strs[li])
            
            # Print connections from this layer to all others where connections lead
            # Collect connections whose input layer is li
            conn_lines = []
            for conn in self.connection_genes:
                in_layer, in_pos = node_positions[conn.in_node.innovation]
                out_layer, out_pos = node_positions[conn.out_node.innovation]

                if in_layer == li:
                    # Show a connection: N(in_innovation) --w--> N(out_innovation)
                    conn_str = f"  N({conn.in_node.innovation}) --{conn.weight:.2f}{'X' if not conn.enabled else '>'} N({conn.out_node.innovation}) [L{in_layer}->{out_layer}]"
                    conn_lines.append(conn_str)

            if conn_lines:
                lines.append("Connections from L{}:".format(li))
                lines.extend(conn_lines)

        # Combine all info
        header = (
            f"Genome:\n"
            f"Fitness: {self.fitness}\n"
            f"Adjusted Fitness: {self.adjusted_fitness}\n"
            f"Activation Function: {self.activation_func.__class__.__name__}\n"
            "Network Structure:\n"
        )

        return header + "\n".join(lines)


    def add_fitness(self, score: float) -> None:
        self.fitness += score

    def set_fitness(self, score: float) -> None:
        self.fitness = score

    def forward_pass(self, input: list[float]) -> list[float]:
        """Evaluate the Output of the Genome based on the given Input"""

        assert len(input) == len(self.nodes[INPUT_LAYER]), f"Input Size: {len(input)} != Length of Input Nodes: {len(self.nodes[INPUT_LAYER])}"

        self.reset_nodes()

        self.set_input(input)

        for layer in self.nodes[:OUTPUT_LAYER:]:
            for node in layer:
                if node not in self.nodes[INPUT_LAYER]:
                    node.evaluate_activation(self.activation_func)
                for conn in self.get_node_connections(node):
                    conn.evaluate()
        
        out = self.get_output_values()
        if DEBUG:
            print("Forward Pass Value:", self.get_output_values())
        return out
    
    def get_output_values(self) -> list[float]:
        """Return a List of the Output Values"""
        return [x.value for x in self.nodes[OUTPUT_LAYER]]

    def get_node_connections(self, node: NodeGene) -> list[ConnectionGene]:
        """Return a list of all Connections that have the given node as the input node"""
        connections = []

        for conn in self.connection_genes:
            if conn.in_node == node and conn.enabled:
                connections.append(conn)

        return connections
                
    def set_input(self, input: list[float]) -> None:
        """Set the Input Node Values"""
        assert len(input) == len(self.nodes[INPUT_LAYER]), f"Input Size: {len(input)} != Length of Input Nodes: {len(self.nodes[INPUT_LAYER])}"

        for i in range(len(self.nodes[INPUT_LAYER])):
            input_node = self.nodes[INPUT_LAYER][i]
            input_node.set_value(input[i])

    def reset_nodes(self) -> None:
        """
        Reset Values of all Nodes to 0
        """

        for layer in self.nodes:
            for node in layer:
                node.reset_value()


    def initialize_gen_zero(self, input_size, output_size, innovation_num: int = 0):
        """
        Initialize The Zeroth Generation with randomised weights
        """
        innovation = innovation_num

        for _ in range(input_size):
            node = NodeGene(innovation)
            self.nodes[INPUT_LAYER].append(node)
            innovation += 1

        for _ in range(output_size):
            node = NodeGene(innovation)
            self.nodes[OUTPUT_LAYER].append(node)
            innovation += 1

        for input_node in self.nodes[INPUT_LAYER]:
            for output_node in self.nodes[OUTPUT_LAYER]:
                conn = ConnectionGene(input_node, output_node, innovation)
                innovation += 1

                self.connection_genes.append(conn)

        if DEBUG:
            print("Input Nodes:", len(self.nodes[INPUT_LAYER]))
            print("Output Nodes:", len(self.nodes[OUTPUT_LAYER]))
            print("Connections:", len(self.connection_genes))
            
            for conn in self.connection_genes:
                print(conn.innovation, ":", conn.weight)

            print("-"*40)

    def connection_exists(self, in_node: NodeGene, out_node: NodeGene) -> bool:
        for conn in self.connection_genes:
            if conn.in_node == in_node and conn.out_node == out_node:
                return True
        return False


    def add_connection(self, in_node, out_node, innovation) -> None:
        conn = ConnectionGene(in_node, out_node, innovation)
        self.connection_genes.append(conn)

    def get_layer(self, node: NodeGene) -> int:
        for layer in range(len(self.nodes)):
            if node in self.nodes[layer]:
                return layer
            
        print("Node Not Found")
        raise ValueError

    def add_nodes(self, nodes: list[NodeGene], parent: Genome) -> None:

        self.nodes = [[] for _ in range(len(parent.nodes))]
        for i in range(len(parent.nodes)):
            for node in parent.nodes[i]:
                n = node.get_node(nodes, False)
                self.nodes[i].append(n)


    def copy(self) -> Genome:
        # Create a new genome with the same activation function
        new_genome = Genome(self.activation_func)
        new_genome.fitness = self.fitness
        new_genome.adjusted_fitness = self.adjusted_fitness

        # Copy the nodes. We copy layer by layer. 
        # Since NodeGene is identified by innovation, we can just recreate them.
        new_genome.nodes = []
        node_map = {}  # Map old node innovation to new node object

        for layer in self.nodes:
            new_layer = []
            for node in layer:
                new_node = NodeGene(node.innovation)
                new_node.set_value(node.value)  # Although typically 0, we keep it consistent
                new_layer.append(new_node)
                node_map[node.innovation] = new_node
            new_genome.nodes.append(new_layer)

        # Copy connections
        new_connections = []
        for conn in self.connection_genes:
            # Find corresponding in_node and out_node from node_map
            in_node = node_map[conn.in_node.innovation]
            out_node = node_map[conn.out_node.innovation]
            new_conn = ConnectionGene(in_node, out_node, conn.innovation, conn.weight, conn.enabled)
            new_connections.append(new_conn)

        new_genome.connection_genes = new_connections
        return new_genome


    def get_connection(self, innovation) -> ConnectionGene:
        for conn in self.connection_genes:
            if conn.innovation == innovation:
                return conn

    def cross(self, other: Genome) -> Genome:
        new_genome = Genome(self.activation_func)

        self.connection_genes.sort(key=lambda x: x.innovation)
        other.connection_genes.sort(key=lambda x: x.innovation)

        nodes: list[NodeGene] = []
        new_genes: list[ConnectionGene] = []

        for conn in self.connection_genes:
            i = conn.innovation
            other_gene = other.get_connection(i)
            in_node = conn.in_node.get_node(nodes)
            out_node = conn.out_node.get_node(nodes)

            if other_gene:
                r_gene = random.choice([other_gene, conn])
                cross_gene = ConnectionGene(in_node, out_node, r_gene.innovation, r_gene.weight, r_gene.enabled)
            else:
                cross_gene = ConnectionGene(in_node, out_node, conn.innovation, conn.weight, conn.enabled)

            new_genes.append(cross_gene)
        
        new_genome.add_nodes(nodes, self)
        new_genome.connection_genes = new_genes

        return new_genome

    def compatability_distance(self, rep: Genome, c1: float, c2: float, c3: float) -> float:
        """
        Compatability Distance: δ = (c1 * E)/N + (c2*D)/N + c3*W
        E: No. of Excess Genes
        D: No. of Disjoint Genes
        N: Total No. of Genes in larger Genome
        W: Average Weight difference in matching genes incl. Disabled Genes.
        """
        # Sort connection genes by innovation
        self_genes = sorted(self.connection_genes, key=lambda g: g.innovation)
        rep_genes = sorted(rep.connection_genes, key=lambda g: g.innovation)

        i, j = 0, 0
        E = 0  # Excess
        D = 0  # Disjoint
        W = 0  # Weight differences sum
        M = 0  # Matching genes count

        # Get max innovation in both
        max_self_inno = self_genes[-1].innovation if self_genes else 0
        max_rep_inno = rep_genes[-1].innovation if rep_genes else 0
        max_inno = max(max_self_inno, max_rep_inno)

        while i < len(self_genes) and j < len(rep_genes):
            g1 = self_genes[i]
            g2 = rep_genes[j]

            if g1.innovation == g2.innovation:
                # Matching gene
                M += 1
                W += abs(g1.weight - g2.weight)
                i += 1
                j += 1
            elif g1.innovation < g2.innovation:
                # Gene in self not in rep
                D += 1
                i += 1
            else:
                # Gene in rep not in self
                D += 1
                j += 1

        # Remaining genes in self are either disjoint or excess
        while i < len(self_genes):
            # If innovation number is beyond the other genome's max innovation => Excess
            if self_genes[i].innovation > max_rep_inno:
                E += 1
            else:
                D += 1
            i += 1

        # Remaining genes in rep are either disjoint or excess
        while j < len(rep_genes):
            if rep_genes[j].innovation > max_self_inno:
                E += 1
            else:
                D += 1
            j += 1

        # Compute average weight difference W
        W = (W / M) if M > 0 else 0

        # N is the length of the larger genome or 1 to avoid division by zero
        N = max(len(self_genes), len(rep_genes))
        if N < 20:  # NEAT paper suggests using N=1 if the number of genes is small
            N = 1

        delta = (c1 * E) / N + (c2 * D) / N + (c3 * W)
        return delta

class SamplingDist:
    """
    Sampling Distribution for Mutation
    """
    pass


class NEAT:
    """
    Keeps track of global innovation number

    Hyperparameters:
        - c1, c2, c3
        - population
        - Mutation Sampling Distribution
    """

    def __init__(self, input_size: int, output_size: int, population_size: int, c1: float, c2: float, c3: float, compatability_threshold: float, sampling_dist: SamplingDist, activation_function_name: str = 'ReLU') -> None:

        self.input_size: int = input_size
        self.output_size: int = output_size
        self.population_size: int = population_size
        self.c1: float = c1
        self.c2: float = c2
        self.c3: float = c3
        self.dt: float = compatability_threshold
        self.sampling_dist: SamplingDist = sampling_dist

        self.generation_number: int = 0
        self.global_innovation = 0

        self.population: list[Genome] = []
        self.species_representatives: list[Genome] = []
        self.species: list[list[Genome]] = [[]]

        self.activation_func: ActivationFunction = get_activation_function(activation_function_name)

        # Initialize Gen 0 Population
        self.initialize_population()
        self.select_species_rep()

    def initialize_population(self) -> None:
        
        for _ in range(self.population_size):
            g = Genome(self.activation_func)
            g.initialize_gen_zero(self.input_size, self.output_size)
            self.population.append(g)
            self.species[0].append(g)

        # All initialised Full connected networks
        self.global_innovation = (self.input_size * self.output_size) + self.input_size + self.output_size
        # Select Species Representative
        self.species_representatives.append(random.choice(self.species[0]))


    def mutate_population(self) -> None:
        """
        Mutation Types:
            Non-structural Innovation
                - Perturbation of Connection Weight
            Structural Innovation:
                - Add Connection Mutation: randomised weight
                - Add Node Mutation: Connection-in has weight of 1, Connection-out has weight of Connection-old.
                    Connection-old is disabled.
        """

        for genome in self.population:
            # Non-structural Innovation
            for conn in genome.connection_genes:
                r = random.random()
                if r <= 0.72:
                    # Small Perturbation
                    conn.weight += random.uniform(-0.5, 0.5)
                elif r <= 0.8:
                    # Random Value
                    conn.weight = random.uniform(-1, 1)

            
            # Structural Innovation
            if random.random() <= 0.05:
                # Add new Connection
                layer_one = random.randint(0, len(genome.nodes)-2)
                layer_two = random.randint(layer_one+1, len(genome.nodes)-1)

                in_node = random.choice(genome.nodes[layer_one])
                out_node = random.choice(genome.nodes[layer_two])

                if not genome.connection_exists(in_node, out_node):
                    genome.add_connection(in_node, out_node, self.global_innovation)
                    self.global_innovation += 1

            if random.random() <= 0.03:
                # Add new Node
                node = NodeGene(self.global_innovation)
                self.global_innovation += 1

                conn_old = random.choice(genome.connection_genes)
                if not conn_old.enabled:
                    continue

                conn_old.enabled = False

                node_in = conn_old.in_node
                node_out = conn_old.out_node

                layer_one = genome.get_layer(node_in)
                layer_two = genome.get_layer(node_out)

                if layer_two - layer_one == 1:
                    genome.nodes.insert(layer_two, [])
                else:
                    layer_two -= 1

                genome.nodes[layer_two].append(node)

                conn_in = ConnectionGene(node_in, node, self.global_innovation, 1)
                self.global_innovation += 1
                genome.connection_genes.append(conn_in)
                conn_out = ConnectionGene(node, node_out, self.global_innovation, conn_old.weight)
                self.global_innovation += 1 
                genome.connection_genes.append(conn_out)

    def speciation(self, new_population: list[Genome]) -> None:
        
        self.species = [[] for _ in range(len(self.species_representatives))]
        for genome in new_population:
            added = False
            for i in range(len(self.species_representatives)):
                rep = self.species_representatives[i]

                if genome.compatability_distance(rep, self.c1, self.c2, self.c3) <= self.dt and not added:
                    self.species[i].append(genome)
                    added = True
                    break
            
            if not added:
                if DEBUG:
                    print("NEW SPECIES ALERT!!")
                self.species.append([genome])
                self.species_representatives.append(genome)
        
        # Remove Empty Species
        self.species = [specie for specie in self.species if len(specie) > 0]
        ### POTENTIAL BUG: May need to remove corresponding representative

    def get_species(self, genome: Genome) -> int:
        
        for i in range(len(self.species)):
            if genome in self.species[i]:
                return i
        
        print("Species NOT found")
        raise ValueError

    def evaluate_adjusted_fitness(self):
        """
        Adjusted Fitness Function: fi' = fi / (Σj:1->n sh(δ(i, j))) = fi/Ns
        sh(δ(i, j)) = 0, δ(i, j)  > δt
                    = 1, δ(i, j) <= δt
        Ns: No of Organisms in the same species. 
        """
        for genome in self.population:

            species_index = self.get_species(genome)
            genome.adjusted_fitness = genome.fitness / len(self.species[species_index])

    def remove_bottom_performers(self) -> None:
        """
        Keep only top 25% of the Species
        """
        for i in range(len(self.species)):
            self.species[i].sort(key=lambda x: x.adjusted_fitness, reverse=True)
            self.species[i] = self.species[i][:math.ceil(0.25 * len(self.species[i])):]

            if len(self.species[i]) == 0: print("EMPTY SPECIES")

    def crossover(self) -> list[Genome]:
        species_fitness = [sum([genome.adjusted_fitness for genome in species]) for species in self.species]
        total_fitness = sum(species_fitness)
        species_fitness_proportion = [int(round((sf / total_fitness) * self.population_size)) if total_fitness > 0 else 0 for sf in species_fitness]

        new_population: list[Genome] = []

        for i in range(len(self.species)):
            j = 0
            while j < species_fitness_proportion[i] and self.species[i] != []:
                genome_one = random_non_uniform_choice(self.species[i])
                genome_two = random_non_uniform_choice(self.species[i])

                if genome_one.adjusted_fitness >= genome_two.adjusted_fitness:
                    new_genome = genome_one.cross(genome_two)
                else:
                    new_genome = genome_two.cross(genome_one)

                new_population.append(new_genome)
                j += 1
                if DEBUG:
                    print("New Dude Alert")
        
        if DEBUG:
            print("Crossover Complete")
            print("new population:", len(new_population))

        return new_population

    def generate_next_generation(self) -> None:

        new_population = self.crossover()
        self.select_species_rep()
        self.speciation(new_population)
        self.population = new_population

    def select_species_rep(self) -> None:
        
        for i in range(len(self.species)):
            assert len(self.species[i]) > 0, "Empty Species"
            self.species_representatives[i] = random.choice(self.species[i])


class NEATManager:
    """
    Streamlines NEAT Development Process, with set parameters for running Generations
    """

    def __init__(self, neat: NEAT) -> None:
        self.neat: NEAT = neat

    def next_generation(self) -> None:
        """
        Evaluate Adjusted Fitness -> Survival of the fittest -> 
        Crossover -> Extinction of Prev Generation -> Mutate New Generation
        """
        print("Evaluating Adjusted Fitness")
        self.neat.evaluate_adjusted_fitness()
        print("Removing Bottom Performers")
        self.neat.remove_bottom_performers()
        print("Generating Next Generation")
        self.neat.generate_next_generation()
        print("Mutating Next Generation")
        self.neat.mutate_population()


if __name__ == "__main__":
    pass