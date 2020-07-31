import math
import random
import sys

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn
import cProfile
from progress.bar import Bar

import utils


GRAVITY = 10

def get_bin_centers(bins):
    centers = []
    last = None
    for post in bins:
        if last is None:
            last = post
            continue

        center = (post + last) * 0.5
        centers.append(center)
        last = post
    return centers


class Individual:
    def __init__(self, parent_network, tag):
        self.network = parent_network
        self.tag = tag

        self.belief_state = 0.0
        self.mass = 100
        self.vel = 0.0

        self.friction_coefficient = 0.001
        self.drag_coefficient = 0.01

        self.belief_state_log = [self.belief_state]
        self.friction_log = [self.friction]
        self.vel_log = [self.vel]

    @property
    def inbound_connections(self):
        return self.network.get_inbound_connections_to(self)

    @property
    def outbound_connections(self):
        return self.network.get_outbound_connections_from(self)

    @property
    def internal_messages(self):
        while True:
            message = np.random.normal(0, 0.5, 1)[0]

            if (message < 1) and (message > -1):
                continue

            yield message

    @property
    def friction(self):
        fc = abs(self.belief_state) * self.friction_coefficient
        f = fc * self.mass * GRAVITY
        f = min([f, abs(self.vel)])
        assert f >= 0
        sign = -math.copysign(1, self.vel)
        f *= sign
        return f

    @property
    def wind_resistance(self):
        f = self.drag_coefficient * (self.vel ** 2)
        f = min([f, abs(self.vel)])
        f *= -math.copysign(1, self.vel)
        return f

    def read_message(self, message, time=0.01):
        # apply force
        force = message + self.friction + self.wind_resistance
        self.vel += (force / self.mass) * time

        # move
        self.belief_state += self.vel

        # clamp belief state
        prev_belief_state = self.belief_state
        self.belief_state = min(1.0, self.belief_state)
        self.belief_state = max(-1.0, self.belief_state)
        if prev_belief_state != self.belief_state:
            self.vel = 0

        # record friction
        self.friction_log.append(self.friction)

        self.vel_log.append(self.vel)

    def listen_to_other(self, everyone):
        selection = everyone.copy() # shallow
        if self in selection:
            selection.remove(self)

        if len(selection) > 0:
            other = random.choice(selection)
            self.read_message(other.belief_state)

    def read_internal_message(self):
        message = next(self.internal_messages)
        self.read_message(message)

    def log_belief_state(self):
        # record belief state
        self.belief_state_log.append(self.belief_state)

    def plot_belief_states(self):
        plt.title("Belief States Over Time")
        plt.ylabel("Belief State (in range [-1, 1])")
        plt.xlabel("Message #")
        plt.plot(self.belief_state_log, "b")

    def plot_friction(self):
        plt.plot(self.friction_log, "r")

    def plot_vel(self):
        plt.plot(self.vel_log, "g")


class Simulation:
    def __init__(self, length, population, seed=1):
        self.seed = seed
        self.set_seed(self.seed)

        self.length = length
        self.population = population

        self.network = Network(self.population)

        # logs
        self.full_belief_state_log = []
        self.summed_log = [0 for i in range(self.length)]
        self.mean_log = [0 for i in range(self.length)]

    def run(self):
        with Bar('Running Simulation', fill='#', suffix='%(percent).1f%% - %(eta)ds') as bar:
            for step in range(self.length):
                self.resolve_timestep(step)
                bar.next()

    def resolve_timestep(self, step):
        self.update_everyone()
        self.log_belief_states(step)

    def update_everyone(self):
        # TODO: use network connections
        for indv in self.network.individuals:
            indv.read_internal_message()
        for indv in self.network.individuals:
            indv.listen_to_other(indv.outbound_connections)

    def log_belief_states(self, step):
        all_belief_states = []
        for indv in self.network.individuals:
            indv.log_belief_state()
            self.summed_log[step] += indv.belief_state
            self.mean_log[step] = self.summed_log[step] / self.population
            all_belief_states.append(indv.belief_state)
        
        self.full_belief_state_log.append(all_belief_states)
        
    def plot_results(self):
        self.network.draw_graph()

        for indv in self.network.individuals:
            indv.plot_belief_states()
        plt.show()

        plt.plot(self.mean_log)
        plt.show()

        # create and save belief state plots over time
        with Bar("Saving Plots", fill="#", suffix='%(percent).1f%% - %(eta)ds') as bar:
            for timestep, _ in enumerate(self.full_belief_state_log):
                self.display_belief_state_histogram(timestep, mode="save")
                bar.next()

    def display_belief_state_histogram(self, timestep, bins=np.arange(-1.1,1.3,0.2), mode="save"):
        belief_states = np.array(self.full_belief_state_log[timestep])
        
        fig = plt.figure()
        axes = plt.gca()
        axes.set_ylim([0, 1])
        axes.set_xlim([-1,1])

        hist, _ = np.histogram(belief_states, bins)
        normalized_hist = [v / self.population for v in hist]
        #print("Normalized Hist: ", [round(v, 4) for v in normalized_hist])
        xs = get_bin_centers(bins)
        #print("Bin Centers: ", [round(v, 4) for v in xs])
        plt.plot(xs, normalized_hist, "b")

        plt.xlabel("Belief State")
        plt.ylabel("Percent of Population")
        plt.title("Belief State Histogram")

        if mode == "save":
            plt.savefig(f"./belief_histograms/belief_states_{timestep}.png")
            plt.clf()
        elif mode == "show":
            plt.show()
        else:
            raise ValueError("mode must be 'save' or 'show'")
        plt.close(fig)

    def set_seed(self, new_seed):
        self.seed = new_seed
        random.seed(self.seed)
        np.random.seed(self.seed)


class Network:
    def __init__(self, size):
        self.size = size
        self.graph = nx.scale_free_graph(self.size)
        self.individuals = [Individual(self, tag) for tag in self.graph.nodes]

    def draw_graph(self, max_node_size=300):
        # determine node sizes
        nodes = list(self.graph.nodes)
        degrees = [self.graph.degree[n] for n in nodes]
        highest = max(degrees)
        node_sizes = [(d / highest) * max_node_size for d in degrees]

        nx.draw(self.graph, node_sizes=node_sizes)
        plt.title("Network Connections")
        plt.show()
    
    def get_frequency_vs_rank_points(self):
        all_degrees = list(dict(self.graph.degree).values())
        unique_degrees = sorted(set(all_degrees))

        counts = np.array([all_degrees.count(d) for d in unique_degrees])
        ranks = np.array(range(len(counts)))

        log_counts = np.log(counts)
        log_ranks = np.log(ranks)

        return log_counts, log_ranks

    def graph_frequency_vs_rank_points(self):
        log_counts, log_ranks = self.get_frequency_vs_rank_points()
        trend_xs, trend_ys = utils.trendline(log_counts, log_ranks)
        plt.xlabel("log(rank)")
        plt.ylabel("log(frequency)")
        plt.plot(log_ranks, log_counts, "b.")
        plt.plot(trend_xs, trend_ys, "r--")
        plt.show()

    def get_inbound_connections_to(self, individual):
        nbunch = self.individuals.index(individual)
        view = self.graph.in_edges(nbunch)
        return [self.individuals[e[1]] for e in view]

    def get_outbound_connections_from(self, individual):
        nbunch = self.individuals.index(individual)
        view = self.graph.out_edges(nbunch)
        return [self.individuals[e[1]] for e in view]

def main():
    seaborn.set()

    try:
        steps = int(sys.argv[1])
        size = int(sys.argv[2])
    except:
        steps = 100
        size = 800
    seed = random.randint(1, 1000)
    print(f"Seed: {seed}")
    sim = Simulation(length=steps, population=size, seed=seed)
    sim.run()

    sim.plot_results()

    return sim.mean_log[-1]


if __name__ == "__main__":
    # TODO: create Simulation class
    # globally define figure
    
    main()
    print("Exiting.")