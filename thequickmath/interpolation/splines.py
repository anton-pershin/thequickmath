import numpy as np

def qubic_spline_coeff(x_nodes, y_nodes):
    """Here underscored variables are related to the matrix equation,
    whereas normal ones stand for the spline coefficients
    """
    polynomials_num = len(x_nodes) - 1
    coeffs = np.zeros((polynomials_num, 3))
    hs = (x_nodes - np.roll(x_nodes, 1))[1:]
    ys = (y_nodes - np.roll(y_nodes, 1))[1:]
    # Build A
    upper_diag = hs.copy()
    upper_diag[0] = 0
    lower_diag = hs.copy()
    lower_diag[-1] = 0
    diag = np.r_[[1], 2 * (upper_diag[1:] + lower_diag[:-1]), [1]]
    A_ = np.diag(upper_diag, 1) + np.diag(diag) + np.diag(lower_diag, -1)
    # Build b
    b_ = np.r_[[0], 3 / hs[1:] * ys[1:] - 3 / hs[:-1] * ys[:-1], [0]]
    c = np.dot(np.linalg.inv(A_), b_)
    a = y_nodes[:-1]
    b = 1 / hs * ys - hs / 3 * (2 * c[:-1] + c[1:])
    d = 1 / (3 * hs) * (c[1:] - c[:-1])
    return np.c_[a, b, c[:-1], d]

def qubic_spline(x, x_nodes, qs_coeff):
    for i in range(len(x_nodes)):
        if x < x_nodes[i] or i == len(x_nodes) - 1:
            dx = x - x_nodes[i - 1]
            return np.dot(qs_coeff[i - 1], np.array([1., dx, dx**2, dx**3]))
    return 0

def make_qubic_spline(x_nodes, y_nodes):
    coeff = qubic_spline_coeff(x_nodes, y_nodes)
    spline_vectorized = np.vectorize(qubic_spline, excluded=set((1, 2)))
    def _qubic_spline(x):
        return spline_vectorized(x, x_nodes, coeff)
    return _qubic_spline