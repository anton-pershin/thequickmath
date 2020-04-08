import numpy as np
import scipy.linalg
from functools import partial

def fourier_diff_matrix(N, p=1):
    '''
    Returns spectral differentiation matrix of order p built on a grid x_j = jh where j = 1...N.
    The periodic interval is then (0; 2 pi].
    '''
    h = 2.*np.pi / N
    if p == 1:
        return scipy.linalg.circulant([1./2 * (-1)**j / np.tan(j*h/2.) if j != 0 else 0 for j in range(N)])
    elif p == 2:
        return scipy.linalg.circulant([-1./2 * (-1)**j / np.sin(j*h/2.)**2 if j != 0 else -np.pi**2 / (3.*h**2) - 1./6 for j in range(N)])
    Exception('p = {} is not implemented!'.format(p))

def cheb_points(N):
    '''
    Returns N Chebyshev points: x_j = cos(j pi / N) where j = 0,...,N.
    '''
    jj = np.arange(0, N + 1)
    return np.cos(jj*np.pi / N)

def cheb_diff_matrix(N, p=1):
    '''
    Returns spectral differentiation matrix of order p built on a grid x_j = cos(j pi / N) where j = 0,...,N.
    The interval is then [-1; 1].
    '''
    c = lambda idx: 2 if idx == 0 or idx == N else 1
    xx = cheb_points(N)
    D = np.zeros((N + 1, N + 1))
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if i != j:
                D[i, j] = c(i) / c(j) * (-1)**(i+j) / (xx[i] - xx[j])
        D[i, i] = -np.sum(D[i, :])
    return np.linalg.matrix_power(D, p)

def spectral_diff_via_matrix_multiplication(v, D_matrix=None, D_matrix_builder=fourier_diff_matrix):
    if D_matrix is None:
        D_matrix = D_matrix_builder()
    return np.dot(D_matrix, v)

def fourier_diff_via_matrix_multiplication(v, D_matrix=None):
    return spectral_diff_via_matrix_multiplication(v, D_matrix=D_matrix, D_matrix_builder=partial(fourier_diff_matrix, len(v)))

def cheb_diff_via_matrix_multiplication(v, D_matrix=None):
    return spectral_diff_via_matrix_multiplication(v, D_matrix=D_matrix, D_matrix_builder=partial(cheb_diff_matrix, len(v) - 1))

def fourier_diff_via_fft(v, order):
    N = len(v)
    if N % 2 != 0:
        Exception('Only even number of nodes can be considered')
    hat_v = np.fft.fft(v)
    diff_coeffs = np.array([(1j * k)**order for k in list(range(int(N//2))) + list(range(int(-N//2), 0))]) # it replicates the standard order of the Fourier coefficients in hat_v:
    # A[0] contains the zero-frequency term, A[1:n/2] contains the positive-frequency terms, 
    # and A[n/2+1:] contains the negative-frequency terms, in order of decreasingly negative frequency.
    hat_w = diff_coeffs * hat_v
    return np.fft.ifft(hat_w)

def cheb_diff_via_fft(v, order):
    N = len(v) - 1
    x = cheb_points(N)
    V_ = np.r_[v, v[-2:0:-1]] # V = [v_0, ..., v_N, v_{N-1}, ..., v_1]
    hat_V = np.fft.fft(V_)
    diff_coeffs = np.array([(1j * k)**order for k in list(range(N)) + list(range(-N, 0))]) # it replicates the standard order of the Fourier coefficients in hat_v:
    # A[0] contains the zero-frequency term, A[1:n/2] contains the positive-frequency terms, 
    # and A[n/2+1:] contains the negative-frequency terms, in order of decreasingly negative frequency.
    hat_W = diff_coeffs * hat_V
    W_ = np.real(np.fft.ifft(hat_W))
    w = np.zeros_like(v)
    w[1:-1] = np.array([-W_[i] / np.sqrt(1 - x[i]**2) for i in range(1, N)])
    w[0] = 1./N * (np.sum([i**2 * np.real(hat_V[i]) for i in np.arange(1, N)]) + 1./2 * N**2 * np.real(hat_V[N]))
    w[N] = 1./N * (np.sum([(-1)**(i+1) * i**2 * np.real(hat_V[i]) for i in np.arange(1, N)]) + 1./2 * (-1)**(N+1) * N**2 * np.real(hat_V[N]))
#    if N % 2 != 0:
#        Exception('Only even number of nodes can be considered')
    return w
