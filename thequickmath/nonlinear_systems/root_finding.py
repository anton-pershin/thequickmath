import numpy as np
import matplotlib.pyplot as plt

MAX_ITERS = 1000
DOUBLE_PRECISION_THRESHOLD = 1e-15
SINGLE_PRECISION_THRESHOLD = 1e-8

def newton_raphson(x_0, f, df):
    x = np.copy(x_0)
#    xs = [x_0]
    for i in range(MAX_ITERS):
        x_last = x
        try:
            x = x_last - np.dot(np.linalg.inv(df(x_last)), f(x_last))
        except np.linalg.LinAlgError:
            return np.zeros_like(x_0)
        x_norm = np.linalg.norm(x)
        abs_error = np.linalg.norm(x - x_last)
#        if x_norm < SINGLE_PRECISION_THRESHOLD:
#            error = abs_error
#        else:
#            error = abs_error / np.linalg.norm(x)
        #print('\t i={}, error={}'.format(i, abs_error))
#        xs.append(x)
        if abs_error < SINGLE_PRECISION_THRESHOLD:
#            xs = np.array(xs)
#            plt.loglog(np.array([np.linalg.norm(x_) for x_ in xs[:-1]]), np.array([np.linalg.norm(x_) for x_ in xs[1:]]))
#            xx = np.array([10**(-16), 10**(-1)])
#            plt.loglog(xx, 10 * xx**2)
#            plt.grid()
#            plt.show()
            return x
    return x

def steepest_descent(x_0, f, df):
    x = np.copy(x_0)
    g = lambda x: np.sum(f(x)**2)
    print(g(x_0))
    for i in range(MAX_ITERS):
        x_last = x
        grad_g = 2*np.dot(df(x_last).T, f(x_last))
        z = grad_g / np.linalg.norm(grad_g)
        h = lambda alpha: g(x_last - alpha*z)
        alpha_1 = 0.
        alpha_3 = 1.
        h_1 = h(alpha_1)
        while h(alpha_3) > h_1:
            alpha_3 /= 2.
        alpha_2 = alpha_3 / 2.
        h_2 = h(alpha_2)
        h_3 = h(alpha_3)
        a = h_1 / ((alpha_1 - alpha_2)*(alpha_1 - alpha_3))
        b = h_2 / ((alpha_2 - alpha_1)*(alpha_2 - alpha_3))
        c = h_3 / ((alpha_3 - alpha_1)*(alpha_3 - alpha_2))
        alpha_opt = (a*(alpha_2 + alpha_3) + b*(alpha_1 + alpha_3) + c*(alpha_1 + alpha_2)) / (2*(a+b+c))
#        h_2 = (h(alpha_2) - h_1) / alpha_2
#        tilde_h_2 = (h(alpha_3) - h(alpha_2)) / (alpha_3 - alpha_2)
#        h_3 = (tilde_h_2 - h_2) / alpha_3
#        alpha_opt = (alpha_2 - h_2 / h_3) / 2.
        x = x_last - alpha_opt * z
        x_norm = np.linalg.norm(x)
        abs_error = np.linalg.norm(x - x_last)
#        if x_norm < SINGLE_PRECISION_THRESHOLD:
#            error = abs_error
#        else:
#            error = abs_error / np.linalg.norm(x)
        print('\t i={}, g_value={}, error={}'.format(i, g(x), abs_error))
        if abs_error < SINGLE_PRECISION_THRESHOLD:
            return x
    print('OUT OF 1000 ITERATIONS')
    return x