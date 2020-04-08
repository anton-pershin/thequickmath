import unittest
import numpy as np
from thequickmath.differentiation.spectral import *
from functools import partial

class GoodSpectralDiffCheck(unittest.TestCase):
    test_periodic_fs = [
        lambda x: np.abs(np.sin(x))**3,
        lambda x: 1. / (1. + np.sin(x/2.)**2),
        lambda x: np.sin(10.*x),
    ]
    test_periodic_dfs = [
        lambda x: 3. * np.sin(x) * np.cos(x) * np.abs(np.sin(x)),
        lambda x: -np.sin(x) / (2.*(1. + np.sin(x/2.)**2)**2),
        lambda x: 10.*np.cos(10.*x),
    ]
    test_arbitrary_fs = [
        lambda x: np.abs(x**3),
        lambda x: 1. / (1. + x**2),
        lambda x: x**10,
    ]
    test_arbitrary_dfs = [
        lambda x: 3.*x*np.abs(x),
        lambda x: -2.*x / (1. + x**2)**2,
        lambda x: 10.*x**9,
    ]

    Ns = [2*n for n in range(4, 25)]

    def _fourier_xx_builder(self, N):
        return np.linspace(0., 2*np.pi, N, endpoint=False)

    def _cheb_xx_builder(self, N):
        return cheb_points(N)

    def test_first_fourier_diff_via_matrix_multiplication(self):
        print('Check Fourier differentiation via matrix multiplication')
        self._test_diff(self._fourier_xx_builder, fourier_diff_via_matrix_multiplication, self.test_periodic_fs, self.test_periodic_dfs)

    def test_first_fourier_diff_via_fourier_space(self):
        print('Check Fourier differentiation via FFT')
        self._test_diff(self._fourier_xx_builder, partial(fourier_diff_via_fft, order=1), self.test_periodic_fs, self.test_periodic_dfs)

    def test_first_cheb_diff_via_matrix_multiplication(self):
        print('Check Chebyshev differentiation via matrix multiplication')
        self._test_diff(self._cheb_xx_builder, cheb_diff_via_matrix_multiplication, self.test_arbitrary_fs, self.test_arbitrary_dfs)

    def test_first_cheb_diff_via_fourier_space(self):
        print('Check Chebyshev differentiation via FFT')
        self._test_diff(self._cheb_xx_builder, partial(cheb_diff_via_fft, order=1), self.test_arbitrary_fs, self.test_arbitrary_dfs)

    def _test_diff(self, xx_builder, diff_func, test_fs, test_dfs):
        print('These results should be compared against Spectral Methods in MATLAB, p.36 for Fourier differentiation and p. 58 for Chebyshev differentiation')
        for f, df in zip(test_fs, test_dfs):
            print('Errors for spectral derivatives of {}:'.format(f))
            for N in self.Ns:
                x = xx_builder(N)
                v = f(x)
                w_exact = df(x)
                w = diff_func(v)
                print('N = {}, ||w_exact - w|| = {}'.format(N, np.linalg.norm(w_exact - w, ord=np.inf)))

if __name__ == '__main__':
    unittest.main()