import math
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import cProfile


GRAVITY = 10
NUM_MESSAGES = 100
NUM_INDIVIDUALS = 3

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
    def __init__(self):
        self.belief_state = 0.0
        self.mass = 1
        self.vel = 0.0

        self.friction_coefficient = 0.001
        self.drag_coefficient = 0.01

        self.belief_state_log = [self.belief_state]
        self.friction_log = [self.friction]
        self.vel_log = [self.vel]

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
        other = random.choice(everyone)

        # listen to someone else
        while not (other is self):
            other = random.choice(everyone)
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

        self.individuals = [Individual() for i in range(population)]

        # logs
        self.full_belief_state_log = []
        self.summed_log = [0 for i in range(self.length)]
        self.mean_log = [0 for i in range(self.length)]

    def run(self):
        for step in range(self.length):
            self.resolve_timestep(step)

    def resolve_timestep(self, step):
        self.update_everyone()
        self.log_belief_states(step)

    def update_everyone(self):
        for indv in self.individuals:
            indv.read_internal_message()
        for indv in self.individuals:
            indv.listen_to_other(self.individuals)

    def log_belief_states(self, step):
        all_belief_states = []
        for indv in self.individuals:
            indv.log_belief_state()
            self.summed_log[step] += indv.belief_state
            self.mean_log[step] = self.summed_log[step] / self.population
            all_belief_states.append(indv.belief_state)
        
        self.full_belief_state_log.append(all_belief_states)
        
    def plot_results(self):
        for indv in self.individuals:
            indv.plot_belief_states()
        plt.show()

        plt.plot(self.mean_log)
        plt.show()

        # create and save belief state plots over time
        for timestep, _ in enumerate(self.full_belief_state_log):
            self.display_belief_state_histogram(timestep, mode="save")

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
    def __init__(self):
        pass


def display_belief_state_histogram(timestep, full_belief_states, bins=np.arange(-1.1,1.3,0.2), mode="save"):
    belief_states = np.array(full_belief_states[timestep])
    
    fig = plt.figure()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    axes.set_xlim([-1,1])

    hist, _ = np.histogram(belief_states, bins)
    normalized_hist = [v / NUM_INDIVIDUALS for v in hist]
    print("Normalized Hist: ", [round(v, 4) for v in normalized_hist])
    xs = get_bin_centers(bins)
    #print("Bin Centers: ", [round(v, 4) for v in xs])
    plt.plot(xs, normalized_hist, "b")

    if mode == "save":
        plt.savefig(f"./belief_histograms/belief_states_{timestep}.png")
        plt.clf()
    elif mode == "show":
        plt.show()
    else:
        raise ValueError("mode must be 'save' or 'show'")
    plt.close(fig)

def create_internal_messages(count):
    messages = [[] for _ in range(NUM_INDIVIDUALS)]
    for message_list in messages:
        while len(message_list) < NUM_MESSAGES:
            message = np.random.normal(0, 0.5, 1)[0]

            if (message < 1) and (message > -1):
                message_list.append(message)

    assert len(messages) == NUM_INDIVIDUALS
    assert len(messages[0]) == NUM_MESSAGES

    return messages

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def run_simulation():
    messages = create_internal_messages(NUM_MESSAGES)

    # show distribution
    plt.hist(messages, int(NUM_MESSAGES * 0.01))
    plt.show()

    # run simulation
    summed_log = [0 for m in range(NUM_MESSAGES)]
    full_belief_state_log = []
    individuals = [Individual() for _ in range(NUM_INDIVIDUALS)]
    for message_index in range(NUM_MESSAGES):
        for index, indv in enumerate(individuals):
            indv.read_message(messages[index][message_index])
        
        # make everyone listen to a random other person
        for indv in individuals:
            indv.listen_to_other(individuals)

        # log belief states
        all_belief_states = []
        for indv in individuals:
            indv.log_belief_state()
            summed_log[message_index] += indv.belief_state
            all_belief_states.append(indv.belief_state)
        
        full_belief_state_log.append(all_belief_states)

    return summed_log, full_belief_state_log, individuals

def plot_results(summed_log, full_belief_state_log, individuals):
    # plot belief states
    for indv in individuals:
        indv.plot_belief_states()
    plt.show()
    
    mean_log = [sum_at_timestep / NUM_INDIVIDUALS for sum_at_timestep in summed_log]
    plt.plot(mean_log)
    # indv.plot_friction()
    #plt.show()
    #indv.plot_vel()
    plt.show()

    for timestep, _ in enumerate(full_belief_state_log):
        display_belief_state_histogram(timestep, full_belief_state_log, mode="save")

def main():
    seaborn.set()

    seed = random.randint(1, 1000)
    print(f"Seed: {seed}")
    sim = Simulation(100, 10, seed=seed)
    sim.run()

    return sim.mean_log[-1]


if __name__ == "__main__":
    # TODO: create Simulation class
    # globally define figure
    
    main()

    """
    fig = plt.Figure()

    set_seed(1)
    results = run_simulation()

    print(np.array(results[1]))

    plot_results(*results)
    """