from __future__ import division
from algebraic_systems import *
from integration import *

from sympy.utilities.lambdify import lambdify
from sympy.abc import x
import numpy as np

class GalerkinMethod:
    '''
    Works on [-1, 1] domain
    '''

    def __init__(self):
        pass

    def set_operator(self, op):
        self.L = op

    def set_rhs(self, rhs):
        self.f = rhs

    def set_boundary_conditions(self, beta, B):
        '''
        Assume we have homogeneous Dirichlet BC at -1 and Mixed BC at 1
        Mixed BC has a form u'(1) + beta*u(1) = B
        '''
        self.beta = beta
        self.B = B

    def set_basis(self, basis):
        self.phi = basis

    def set_dimension(self, dim):
        self.dim = dim

    def build_task(self):
        '''
        Resultant task is a linear/nonlinear system of equations
        Let's suppose that only linear system of equations is returned
        '''
        A = np.zeros((self.dim, self.dim))
        b = np.zeros((self.dim,))
        
        print(lambdify(x, (self.L.p * self.phi[0]).diff(x), 'numpy'))
        p_phi_deriv = [lambdify(x, (self.L.p * self.phi[i]).diff(x), 'numpy') for i in range(self.dim)]
        q_phi = [lambdify(x, self.L.q * self.phi[i], 'numpy') for i in range(self.dim)]
        phi_deriv = [lambdify(x, self.phi[i].diff(x), 'numpy') for i in range(self.dim)]
        phi = [lambdify(x, self.phi[i], 'numpy') for i in range(self.dim)]
        p = lambdify(x, self.L.p, 'numpy')
        r = lambdify(x, self.L.r, 'numpy')
        f = lambdify(x, self.f, 'numpy')

        for i in range(self.dim):
            for j in range(self.dim):
                A[i, j] = integrate(lambda x: (p_phi_deriv[i](x) - q_phi[i](x))*phi_deriv[j](x) - r(x)*phi[i](x)*phi[j](x), -1, 1) \
                        + self.beta*p(1)*phi[i](1)*phi[j](1)
            b[i] = self.B * p(1) * phi[i](1) - integrate(lambda x: f(x) * phi[i](x), -1, 1)
        return LinearAlgebraicSystem(A, b)