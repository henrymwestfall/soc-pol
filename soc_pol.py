import random
import math
import pickle
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy
from scipy.stats import linregress, ttest_1samp
from scipy.cluster.vq import kmeans2

random.seed(1)


# utility functions
def shuffled(iterable):
    copy_to_shuffle = iterable.copy()
    random.shuffle(copy_to_shuffle)
    return copy_to_shuffle


class Individual:
    def __init__(self, id_number: int):
        self.id_number = id_number
        
        self.outbound_connections = []
        self.inbound_connections = []
        
        self.outbox = []
        self.next_outbox = []

        self.belief_state = random.randint(10, 50) * random.choice([-1, 1]) / 100
        self.next_belief_state = 0

        # disagreement thresholds
        self.disconnect_threshold = 0.3
        self.follow_threshold = 0.1
        self.reshare_threshold = 0.2

    def update(self):
        '''
        Set current belief state and outbox to their next states
        '''

        self.belief_state = self.next_belief_state
        self.next_belief_state = 0
        
        self.outbox = self.next_outbox
        self.next_outbox = []

    def read_messages(self):
        '''
        Read and react to messages
        '''
        
        # generate inbox of messages
        inbox = []
        for other in self.outbound_connections:
            inbox += other.outbox

        state = self.belief_state
        for post in inbox[:50]:
            
            disagreement = abs(post.message - state)
            
            memory = abs(state) * 2 # faster than abs(state) / 0.5
            assert memory <= 1, f"memory exceeds 1 {memory}"
            assert memory >= 0, f"memory is below 0 {memory}"

            # update state
            state = memory * state + (1 - memory) * post.message

            # handle thresholds
            if disagreement > self.disconnect_threshold:
                self.disconnect_from(post.author)
                continue
            
            if disagreement < self.follow_threshold:
                self.connect_to(post.author)
            if disagreement < self.reshare_threshold:
                self.next_outbox.append(post)
                post.reshares += 1

        assert abs(state) <= 0.5, f"belief state too extreme {state}"

        self.next_belief_state = state

        # make a post if confident enough
        confidence = abs(self.next_belief_state) * 2 # similar to memory
        if random.random() <= confidence:
            self.next_outbox.append(Post(self, self.next_belief_state))

    def connect_to(self, other):
        '''
        Create a one-way connection from this individual to other
        '''
        
        if not (other in self.outbound_connections):
            self.outbound_connections.append(other)
            other.inbound_connections.append(self)

    def disconnect_from(self, other):
        '''
        Delete the one-way connection from this individual to other if it exists
        '''

        if other in self.outbound_connections:
            self.outbound_connections.remove(other)
            other.inbound_connections.remove(self)

    def get_path_count(self, depth: int, current_depth=0):
        '''
        Recursively count the inbound paths to this individual.
        '''
        
        num_paths_to_me = len(self.inbound_connections)
        if current_depth < depth:
            for other in self.inbound_connections:
                num_paths_to_me += other.get_path_count(depth, current_depth + 1)
        return num_paths_to_me


class Post:
    def __init__(self, author: Individual, message: float):
        self.author = author
        self.message = message

        # info that could be used by sorting algorithm
        self.reshares = 0
        self.reads = 0
        self.likes = 0


class Network:
    def __init__(self, population, initial_connections, preferential_attachment_passes):
        self.population = population
        self.initial_connections = initial_connections
        self.preferential_attachment_passes = preferential_attachment_passes

        self.individuals = [Individual(i) for i in range(self.population)]

    def plot_network(self, polarization_color_threshold=0.3, save_path=None):
        '''
        Display a directed spring graph of the connections between individuals. Color the individuals
        based on belief state.
        '''

        assert polarization_color_threshold > 0, "polarization color threshold must be positive"

        G = nx.DiGraph()

        # add nodes for each individual
        G.add_nodes_from([indv.id_number for indv in self.individuals])

        # build the color map
        color_map = []
        for indv in self.individuals:
            if indv.belief_state >= polarization_color_threshold:
                color_map.append("red")
            elif indv.belief_state <= -polarization_color_threshold:
                color_map.append("blue")
            else:
                color_map.append("gray")

        # add edges
        for indv in self.individuals:
            for other in indv.outbound_connections:
                G.add_edge(indv.id_number, other.id_number)

        # draw
        plt.title("Network Connections")
        pos = nx.spring_layout(G)
        nx.draw(G, node_color=color_map, pos=pos, node_size=5, with_labels=False, edge_color="b", arrow_size=10, arrowstyle="->")

        # save if requested
        if save_path != None:
            plt.savefig(save_path)

        # in case nx.draw didn't work, use plt.show
        plt.show()

    def plot_belief_state_histogram(self, bins=20):
        """
        Plot a histogram of the belief states
        """
        
        all_belief_states = [indv.belief_state for indv in self.individuals]
        plt.hist(all_belief_states, bins, facecolor="blue", alpha=0.5)
        plt.xlabel("Belief States")
        plt.ylabel("Frequency")
        plt.title("Belief State Frequencies")
        plt.show()

    def plot_degrees(self):
        """
        Plot log(degrees) vs log(ranks) of the individuals in the network
        """
        
        degrees = sorted([math.log10(len(indv.inbound_connections) + 1) for indv in self.individuals], reverse=True)
        ranks = [math.log10(r + 1) for r, _ in enumerate(self.individuals)]

        z = numpy.polyfit(ranks, degrees, 1)
        p = numpy.poly1d(z)
        plt.plot(ranks, p(ranks), "r--")

        r_value = linregress(ranks, degrees)[2]
        r2 = r_value ** 2

        print("R^2: ", r2)

        plt.plot(ranks, degrees, "b.")
        plt.xlabel("Rank")
        plt.ylabel("log(degrees)")
        plt.title("Degree Distribution")
        plt.show()
                
    def get_connection_probabilities(self, depth=2) -> dict:
        '''
        Get a dictionary of the probabilities of connecting to each individual
        '''
        
        all_path_counts = [indv.get_path_count(depth) for indv in self.individuals]
        sum_path_counts = sum(all_path_counts)
        connection_probabilities = {indv : count / sum_path_counts for indv, count in zip(self.individuals, all_path_counts)}
        return connection_probabilities

    def generate(self):
        '''
        Generate network connections. First, create a random graph where each
        individual connects to `self.initial_connections` others. Then,
        do `self.preferential_attachment_passes` where individuals preferentially
        attach to others.
        '''

        # connect randomly
        for indv in self.individuals:
            for other in shuffled(self.individuals)[:self.initial_connections]:
                indv.connect_to(other)
                other.connect_to(indv)


        # preferential attachment passes
        for i in range(self.preferential_attachment_passes):
            connection_probabilities = self.get_connection_probabilities()
            for indv in self.individuals:
                has_connected = False
                while not has_connected:
                    for other, prob in connection_probabilities.items():
                        if (random.random() <= prob) and (indv != other):
                            indv.connect_to(other)
                            has_connected = True
                            break
                

class Simulation(Network):
    def __init__(self, length, population, initial_connections, preferential_attachment_passes):
        Network.__init__(self, population, initial_connections, preferential_attachment_passes)

        self.length = length
        self.progress = 0

        self.generate()
        self.plot_degrees()

    def save(self, path):
        # TODO: fix recursion error
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def plot_all(self):
        self.plot_degrees()
        self.plot_belief_state_histogram()
        self.plot_network()

    def run(self):
        '''
        Run the simulation for the predetermined length
        '''
        
        while self.progress < self.length:
            for indv in self.individuals:
                indv.read_messages()
            for indv in self.individuals:
                indv.update()

            # display progress        
            if (self.progress + 1) % (0.1 * self.length) == 0:
                print(f"{int(100 * (self.progress + 1) / self.length)}%", end="")
            elif (self.progress + 1) % int(0.01 * self.length) == 0:
                print(".", end="")

            self.progress += 1
        print()

        self.plot_all()

    def __enter__(self):
        return self

    def __exit__(self, _, __, ___):
        self.save("unfinished.sim")

def main():
    length = 100
    population = 1000
    initial_connections = 1
    preferential_attachment_passes = 4
    with Simulation(length, population, initial_connections, preferential_attachment_passes) as sim:
        sim.run()

    
if __name__ == "__main__":
    main()
