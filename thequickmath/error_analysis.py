import matplotlib.pyplot as plt
import numpy as np

def vector_errors(gen_error_func, param_range):
    errors = np.zeros((len(param_range),))
    for param in param_range:
        error_vector = gen_error_func(param)
    plt.plot(np.array(param_range), errors)
    plt.show()

def func_func_errors(exact_func, gen_approx_func, param_range, space_mesh, log=True):
    plotting_func = plt.semilogy if log else plt.plot
    exact_func_mesh = np.vectorize(exact_func)(space_mesh)
    for param in param_range:
        approx_func = gen_approx_func(param)
        vapprox_func = np.vectorize(approx_func)
        plotting_func(space_mesh, np.abs(exact_func_mesh - vapprox_func(space_mesh)), linewidth=2, label=str(param))
    plt.legend()
    plt.show()

def simple_error(exact_func, approx_vector, space_mesh, log=True):
    plotting_func = plt.semilogy if log else plt.plot
    exact_func_mesh = np.vectorize(exact_func)(space_mesh)
    if log:
        plt.semilogy(space_mesh, np.abs(exact_func_mesh - approx_vector), 'o')
        plt.xlabel('$t$')
        plt.ylabel('$\log|y_{exact} - y|$')
    else:
        plt.plot(space_mesh, exact_func_mesh - approx_vector, 'o')
        plt.xlabel('$t$')
        plt.ylabel('$y_{exact} - y$')
    plt.legend()
    plt.grid()
    plt.show()