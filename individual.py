import math
import random
import sys
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn
import cProfile
from multiprocessing import Process
from progress.bar import Bar
from pygame import Color # for coloring nodes

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


class Message:
    def __init__(self, author, value):
        self.author = author
        self.value = value


class Individual:
    def __init__(self, parent_network, tag):
        self.network = parent_network
        self.tag = tag

        self.belief_state = 0.0
        self.next_belief_state = 0.0
        self.mass = 100
        self.vel = 0.0
        self.comfort_levels = {}

        self.friction_coefficient = 0.001
        self.drag_coefficient = 0.01

        self.reshare_threshold = 0.15

        self.belief_state_log = [self.belief_state]
        self.friction_log = [self.friction]
        self.vel_log = [self.vel]

        self.outbox = [Message(self, self.belief_state)]
        self.next_outbox = self.outbox.copy()

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

            yield Message(self, message)

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
        # TODO: split this function into multiple functions

        # apply force
        force = message.value + self.friction + self.wind_resistance
        dv = (force / self.mass) * time
        self.vel += dv

        if not (message.author is self):
            # adjust comfort levels and make connection if possible
            self.comfort_levels[message.author] = self.comfort_levels.get(message.author, 0) + dv

            # disconnect or connect based on comfort level
            if self.check_agreement(message.author) and abs(message.value) > 0.5:
                self.network.connect(self, message.author)
            elif self.check_disagreement(message.author) and abs(message.value) > 0.5:
                self.network.disconnect(self, message.author)

            # reshare message if dv exceeds threshold
            if abs(dv) >= self.reshare_threshold:
                self.next_outbox.append(message)

        # move
        self.next_belief_state += self.vel

        # clamp belief state
        prev_belief_state = self.next_belief_state
        self.next_belief_state = min(1.0, self.next_belief_state)
        self.next_belief_state = max(-1.0, self.next_belief_state)
        if prev_belief_state != self.next_belief_state: # if we got clamped, cut velocity
            self.vel = 0

        # record friction and velocity
        self.friction_log.append(self.friction)
        self.vel_log.append(self.vel)

    def update(self):
        self.belief_state = self.next_belief_state
        self.outbox = self.next_outbox.copy() # shallow
        self.next_outbox = [Message(self, self.belief_state)]

    def connected_to(self, other):
        return other in self.outbound_connections

    def listen_to_other(self, everyone):
        selection = everyone.copy() # shallow
        if self in selection:
            selection.remove(self)

        if len(selection) > 0:
            other = random.choice(selection)
            message = random.choice(other.outbox)

            self.read_message(message)

    def check_disagreement(self, other):
        return not self.check_agreement(other)

    def check_agreement(self, other):
        return math.copysign(1, other.belief_state) == math.copysign(1, self.belief_state)

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
        plt.plot(self.belief_state_log, "b", alpha=0.07, linewidth=3)

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
        self.individuals = self.network.individuals.copy() # shallow

        # logs
        self.full_belief_state_log = []
        self.summed_log = [0 for i in range(self.length)]
        self.mean_log = [0 for i in range(self.length)]

        # timing
        self.start_time = 0
        self.end_time = 0

    def run(self):
        self.start_time = time.time()
        with Bar('Running Simulation', fill='#', suffix='%(percent).1f%% - %(eta)ds', max=self.length) as bar:
            for step in range(self.length):
                self.resolve_timestep(step)
                bar.next()
        self.end_time = time.time()

    def resolve_timestep(self, step):
        self.update_everyone()
        self.log_belief_states(step)

    def process_group(self, group, callback):
        for indv in group:
            callback(indv)

    def call_indv_read_internal_message(self, indv):
        indv.read_internal_message()

    def call_indv_listen_to_other(self, indv):
        indv.listen_to_other(indv.outbound_connections)

    def call_indv_update(self, indv):
        indv.update()

    # NOTE: this does not yet work
    def update_everyone_parallel(self):
        # TODO: use network connections
        random.shuffle(self.individuals) # change order of things

        # processes
        end1 = int(len(self.individuals) * 0.25)
        end2 = int(len(self.individuals) * 0.5)
        end3 = int(len(self.individuals) * 0.75)

        callbacks = [self.call_indv_read_internal_message, self.call_indv_listen_to_other, self.call_indv_update]

        for callback in callbacks:
            p1 = Process(target=self.process_group, args=(self.individuals[:end1], callback))
            p1.start()
            #p1.join()

            p2 = Process(target=self.process_group, args=(self.individuals[end1:end2], callback))
            p2.start()
            #p2.join()

            p3 = Process(target=self.process_group, args=(self.individuals[end2:end3], callback))
            p3.start()
            #p3.join()

            p4 = Process(target=self.process_group, args=(self.individuals[end3:], callback))
            p4.start()
            #p4.join()

            while p1.is_alive() or p2.is_alive() or p3.is_alive() or p4.is_alive():
                time.sleep(0.05)

    def update_everyone(self):
        # TODO: use network connections
        random.shuffle(self.individuals) # change order of things
        for indv in self.individuals:
            indv.read_internal_message()
        for indv in self.individuals:
            indv.listen_to_other(indv.outbound_connections)
        for indv in self.individuals:
            indv.update()

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
        plt.title("Belief States Over Time")
        plt.ylabel("Belief States")
        plt.xlabel("Steps")
        plt.show()

        plt.plot(self.mean_log)
        plt.show()

        # create and save belief state plots over time
        with Bar("Saving Plots", fill="#", suffix='%(percent).1f%% - %(eta)ds', max=len(self.full_belief_state_log)) as bar:
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

    def display_2d_belief_state_histogram(self):
        xs = []
        ys = []
        with Bar("Creating 2d Histogram", fill="#", suffix="%(percent).1f%% - %(eta)ds", max=len(self.full_belief_state_log)) as bar:
            for step, states in enumerate(self.full_belief_state_log):
                for state in states:
                    xs.append(step)
                    ys.append(state)
                bar.next()

        plt.hist2d(xs, ys, bins=100)
        plt.title("Histogram of Belief States Over Time")
        plt.xlabel("Steps")
        plt.ylabel("Belief States")
        plt.show()

    def plot_outbound_connection_change(self, bins=100):
        all_changes = []
        for indv in self.individuals:
            original_connections = self.network.get_outbound_connections_from(indv, "original")
            current_connections = self.network.get_outbound_connections_from(indv)
            changes = 0
            for connection in original_connections + current_connections:
                if (connection in original_connections) ^ (connection in original_connections):
                    changes += 1
            all_changes.append(changes)
        
        plt.hist(all_changes, bins)
        plt.show()

    def save_belief_state_log(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.full_belief_state_log, f)

        

    def set_seed(self, new_seed):
        self.seed = new_seed
        random.seed(self.seed)
        np.random.seed(self.seed)


class Network:
    def __init__(self, size):
        self.size = size
        self.original_graph = nx.scale_free_graph(self.size)
        self.graph = self.original_graph.copy()
        self.individuals = [Individual(self, tag) for tag in self.graph.nodes]

    def disconnect(self, src, target):
        src_node = self.individuals.index(src)
        target_node = self.individuals.index(target)
        self.graph.remove_edge(src_node, target_node)

    def connect(self, src, target):
        src_node = self.individuals.index(src)
        target_node = self.individuals.index(target)
        self.graph.add_edge(src_node, target_node)

    def get_rgb_color_from_state(self, belief_state):
        if belief_state > 0:
            h = 0 # red hue
        else:
            h = 120 # green hue (blue often does not appear with seaborn styling)
        s = abs(belief_state) * 100 # saturation depends on polarization
        if s < 30: # clamp at 30 to avoid white dots
            s = 30
        color = Color(0, 0, 0)
        color.hsva = (h, s, 100, 100) # value and alpha are always 100
        rgb = (color.r / 255, color.g / 255, color.b / 255)
        return rgb

    def draw_graph(self, max_node_size=300):
        # determine node sizes
        nodes = list(self.graph.nodes)
        degrees = [len(self.graph.in_edges(n)) for n in nodes]
        highest = max(degrees)
        node_sizes = [(d / highest) * max_node_size for d in degrees]
        node_colors = []
        for node in nodes:
            indv = self.individuals[node]
            c = self.get_rgb_color_from_state(indv.belief_state)
            node_colors.append(c)

        nx.draw_networkx(self.graph, node_list=nodes, node_size=node_sizes, with_labels=False, node_color=node_colors)
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

    def get_inbound_connections_to(self, individual, graph_state="current"):
        if graph_state == "current":
            graph = self.graph
        else:
            graph = self.original_graph

        nbunch = self.individuals.index(individual)
        view = graph.in_edges(nbunch)
        return [self.individuals[e[1]] for e in view]

    def get_outbound_connections_from(self, individual, graph_state="current"):
        if graph_state == "current":
            graph = self.graph
        else:
            graph = self.original_graph
        nbunch = self.individuals.index(individual)
        view = graph.out_edges(nbunch)
        return [self.individuals[e[1]] for e in view]

def main():
    seaborn.set()

    try:
        steps = int(sys.argv[1])
        size = int(sys.argv[2])
    except:
        steps = 100
        size = 800
    seed = 811#random.randint(1, 1000)
    print(f"Seed: {seed}")
    sim = Simulation(length=steps, population=size, seed=seed)
    sim.run()
    print("Total Runtime:", sim.end_time - sim.start_time)

    sim.save_belief_state_log("Belief-State-Log.pkl")

    sim.display_2d_belief_state_histogram()
    sim.network.graph_frequency_vs_rank_points()
    sim.plot_outbound_connection_change()
    sim.plot_results()

    return sim.mean_log[-1]


if __name__ == "__main__":
    # TODO: create Simulation class
    # globally define figure
    
    main()
    print("Exiting.")