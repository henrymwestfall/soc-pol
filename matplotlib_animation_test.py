import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# fig, ax = plt.subplots()
# xdata, ydata = [], []
# ln, = plt.plot([], [], 'ro')

# def init():
#     ax.set_xlim(0, 2)
#     ax.set_ylim(-1, 1)
#     return ln,

# def update(frame):
#     xdata.append(frame)
#     ydata.append(np.sin(frame))
#     ln.set_data(xdata, ydata)
#     return ln,

# ani = FuncAnimation(fig, update, frames=100,
#                     init_func=init, blit=True)
# plt.show()

#data = np.random.random((10, 100))
data = [
    [1, 2, 3, 2, 1],
    [3, 1, -1, -3, -5]
]

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 3)
    ax.set_ylim(-3, 3)
    return ln,

def update(frame):
    step_data = []
    for dlist in data:
        step_data.append(dlist[frame])
    ln = plt.hist(step_data, 3)
    return ln,

ani = FuncAnimation(fig, update, frames=5,
                    init_func=init, blit=True)
plt.show()