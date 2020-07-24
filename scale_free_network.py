import random
import math

import networkx as nx
import matplotlib.pyplot as plt

network = {} # maps inbound connections
total_degree = 0 # number of inbound connections in network
k = 1 # constant added to degrees during generation

def name_sequence():
    n = 0
    while True:
        yield n
        n += 1
names = name_sequence()

def add_individual():
    global total_degree

    name = next(names)
    network[name] = []

    # override normal behavior if there are less than 2 left
    if len(network) == 1:
        existing_indv = list(network.keys())[0]
        network[name].append(existing_indv)
        network[existing_indv].append(name)
        total_degree += 2
        return
    elif len(network) == 0:
        network[name] = []
        return

    for other, connections in network.items():
        other_degree = len(connections)
        if random.random() < (other_degree + k) / total_degree:
            total_degree += 1
            network[other].append(name)

def get_edges():
    edges = []
    for individual, connections in network.items():
        for connection in connections:
            edge = (connection, individual)
            edges.append(edge)
    return edges

population = 1000
everyone = list(range(population))
for i in everyone:
    add_individual()

G = nx.DiGraph()
edges = get_edges()
G.add_nodes_from(everyone)
G.add_edges_from(edges)
layout = nx.layout.spring_layout(G)

nx.draw(G, pos=layout, node_size=8)
plt.show()

# degree distribution
# log(rank) vs log(degree)
def log(x):
    try:
        return math.log(x)
    except ValueError:
        return math.log(x + 1)

individuals_ranked = sorted(network.keys(), key=lambda indv: len(network[indv]))
ys = [log(rank) for rank in range(population)]
xs = [log(len(network[indv])) for indv in individuals_ranked]

plt.xlabel("log(Degree)")
plt.ylabel("log(Rank)")
plt.plot(xs, ys, "b.")
plt.show()