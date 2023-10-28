import math

import numpy as np
import pyswarms as ps
from matplotlib import pyplot as plt
from pyswarms.utils.plotters import plot_cost_history


def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)


def formatedEndurance(x):
    return -1 * endurance(x[0], x[1], x[2], x[3], x[4], x[5])


def f(x):
    n_particles = x.shape[0]
    j = [formatedEndurance(x[i]) for i in range(n_particles)]
    return np.array(j)


x_max = np.ones(6)
x_min = np.zeros(6)
my_bounds = (x_min, x_max)

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=12, dimensions=6,
                                    options=options, bounds=my_bounds)

res = optimizer.optimize(f, iters=1000)
cost_history = optimizer.cost_history

# Czy masz podobny? Tak
print(res)
# na etapie punkty d:
# (1.0102910980554718, array([0.16270833, 0.99477711, 0.19453196, 0.28531419, 0.84257653,
#        0.93565404]))

# na etapie punkty e:
# (-2.8221081441719758, array([0.54669107, 0.60453027, 0.99535154, 0.99816857, 0.11382871,
#        0.4992115 ]))


plot_cost_history(cost_history)
plt.show()
