import math

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


population = 10000
G = nx.scale_free_graph(population)

max_size = 300

nodes = list(G.nodes)
degrees = [G.degree[n] for n in nodes]
highest = max(degrees)
node_sizes = [(d / highest) * max_size for d in degrees]


# nx.draw(G, node_size=node_sizes)
# plt.show()

# degree distribution
def log(x):
    try:
        return math.log(x)
    except ValueError:
        return math.log(x + 1)


# plot the frequency of degrees vs the rank of degrees

all_degrees = list(dict(G.degree).values())
unique_degrees = sorted(list(set(all_degrees)))
print(all_degrees)

counts = [all_degrees.count(d) for d in unique_degrees]
ranks = range(len(counts))

log_counts = [math.log(c) for c in counts]
log_ranks = [math.log(r + 1) for r in ranks]

plt.xlabel("log(rank)")
plt.ylabel("log(degrees)")
plt.plot(log_ranks, log_counts, "b.")
plt.show()