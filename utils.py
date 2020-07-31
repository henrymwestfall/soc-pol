import numpy as np

def trendline(xs, ys):
    z = np.polyfit(xs, ys, 1)
    p = np.poly1d(z)
    return xs, [p(x) for x in xs]