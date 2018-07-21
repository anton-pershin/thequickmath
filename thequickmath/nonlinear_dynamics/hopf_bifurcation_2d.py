import sys
sys.path.append('/home/tony/projects/anton-pershin/thequickmath')
import numpy as np
import matplotlib.pyplot as plt
from thequickmath.differential_equations.timesteppers import rk4

mu = -0.2

def f(t_, x_):
    return np.array([mu*x_[0] - x_[1] + x_[0]*(x_[1])**2, \
                     x_[0] + mu*x_[1] + (x_[1])**3])

def build_trajectory(x_0, t_end, h):
    N = int(t_end // h)
    t = np.linspace(0, t_end, N + 1)
    x = np.zeros((2, len(t)))
    x[:, 0] = x_0
    for i in range(N):
        x[:, i + 1] = rk4(x[:, i], t[i], f, h)
    return t, x

x20s = [0.1, 0.2, 0.6, 0.62]

# Time-evolution of trajectories
fig, axes = plt.subplots(len(x20s), 1)
for ax, x20 in zip(axes, x20s):
    t, x = build_trajectory(x_0=np.array([0., x20]), t_end=50., h=0.01)
    ax.plot(t, x[0, :], linewidth=2)
    ax.set_xlabel('$t$', fontsize=16)
    ax.grid()
plt.show()

## Phase portrait
#fig, ax = plt.subplots(1, 1)
#
#for x_10 in np.linspace(100., 2000., 10):
#    for x_20 in np.linspace(100., 2000., 10):
#        print('Calculating x_1={}, x_2={}'.format(x_10, x_20))
#        t, x = build_trajectory(x_0=np.array([x_10, x_20]), t_end=20., h=0.01)
#        ax.plot(x[0, :], x[1, :], color='blue')
#ax.set_xlabel('$x_1$', fontsize=16)
#ax.set_ylabel('$x_2$', fontsize=16)
#ax.legend(fontsize=16)
#ax.grid()
#plt.show()
