import numpy as np

MAX_ITERS = 100
DOUBLE_PRECISION_THRESHOLD = 1e-15

def euler(y_n, t_n, f, h):
    return y_n + h*f(t_n, y_n)

def trapezoidal_rule(y_n, t_n, f, h, y_nn=None):
    if y_nn is None: # we should solve nonlinear system then
        raise Exception('Not implemented')

    t_nn = t_n + h
    return y_n + h/2. * (f(t_n, y_n) + f(t_nn, y_nn))

def heun(y_n, t_n, f, h):
    # euler is predictor, trapezoidal is corrector
    y_pred = euler(y_n, t_n, f, h)
    return trapezoidal_rule(y_n, t_n, f, h, y_pred)

def rk4(y_n, t_n, f, h):
    k_1 = h*f(t_n, y_n)
    k_2 = h*f(t_n + h/2., y_n + k_1/2.)
    k_3 = h*f(t_n + h/2., y_n + k_2/2.)
    k_4 = h*f(t_n + h, y_n + k_3)
    return y_n + 1/6. * (k_1 + 2*k_2 + 2*k_3 + k_4)

def ab4(y_4n, t_n, f, h):
    # Adams-Bashforth 4-step scheme
    # y_n == y_4n[3], y_{n-1} == y_4n[2], ..., y_{n-3} == y_4n[0]
    return y_4n[3] + h/24. * (55*f(t_n, y_4n[3]) - 59*f(t_n - h, y_4n[2]) + 37*f(t_n - 2*h, y_4n[1]) - 9*f(t_n - 3*h, y_4n[0]))

def am4(y_3n, t_n, f, h, y_nn=None):
    # Adams-Moulton 3-step scheme
    # y_n == y_3n[2], y_{n-1} == y_3n[1], y_{n-2} == y_3n[0]
    if y_nn is None:
        raise Exception('Not implemented')

    return y_3n[2] + h/24. * (9*f(t_n + h, y_nn) + 19*f(t_n, y_3n[2]) - 5*f(t_n - h, y_3n[1]) + f(t_n - 2*h, y_3n[0]))

def am_predictor_corrector(y_4n, t_n, f, h):
    # Apply predictor
    y_nn = ab4(y_4n, t_n, f, h)

    # Use inner iteration for implicit method (corrector) to converge
    for k in range(MAX_ITERS):
        y_nn_prev = y_nn
        y_nn = am4(y_4n[1:], t_n, f, h, y_nn=y_nn_prev)
        if np.abs(y_nn_prev - y_nn) < DOUBLE_PRECISION_THRESHOLD:

            print('Iteration converged at k = {} with error = {}'.format(k, np.abs(y_nn_prev - y_nn)))
            return y_nn

    return y_nn