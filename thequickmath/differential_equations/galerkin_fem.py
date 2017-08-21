from __future__ import division
from algebraic_systems import *
from integration import *

from sympy.utilities.lambdify import lambdify
from sympy.abc import xi, x
import numpy as np

class GalerkinFEMMethod:
    '''
    Works on [a, b] domain
    '''

    def __init__(self):
        pass

    def set_operator(self, op):
        self.L = op

    def set_rhs(self, rhs):
        self.f = rhs

    def set_boundary_conditions(self, u_0, u_N):
        '''
        Assume we have homogeneous Dirichlet BC at both boundaries
        '''
        self.u_0 = u_0
        self.u_N = u_N

    def set_universal_local_basis(self, basis):
        '''
        Local basis must be expressed in xi-space
        '''
        self.N = basis

    def set_mesh(self, mesh):
        self.x = mesh

    def build_task(self):
        '''
        Resultant task is a linear/nonlinear system of equations
        Let's suppose that only linear system of equations is returned
        '''
        nodes_num = 2
        dim = self.x.shape[0]
        elem_num = dim - 1
        A = np.zeros((dim, dim))
        b = np.zeros((dim,))

        def x_map(dx, x_1):
            return lambda xi: dx*xi + x_1

        N_xi = [lambdify(xi, self.N[i], 'numpy') for i in range(nodes_num)]
        N_deriv_xi = [lambdify(xi, self.N[i].diff(xi), 'numpy') for i in range(nodes_num)]
        p_x = lambdify(x, self.L.p, 'numpy')
        p_deriv_x = lambdify(x, self.L.p.diff(x), 'numpy')
        q_x = lambdify(x, self.L.q, 'numpy')
        r_x = lambdify(x, self.L.r, 'numpy')
        f_x = lambdify(x, self.f, 'numpy')

        def lhs_local_contribution(i, j, x_1, x_2):
            dx = x_2 - x_1
            x_ = x_map(dx, x_1)
            return integrate(lambda xi: (p_x(x_(xi))*N_deriv_xi[i](xi)*1/dx  + p_deriv_x(x_(xi))*N_xi[i](xi) - \
                                        q_x(x_(xi))*N_xi[i](xi))*N_deriv_xi[j](xi) - r_x(x_(xi))*N_xi[i](xi)*N_xi[j](xi)*dx, \
                                        0, 1)

        def rhs_local_contribution(i, x_1, x_2):
            dx = x_2 - x_1
            x_ = x_map(dx, x_1)
            return integrate(lambda xi: -N_xi[i](xi)*f_x(x_(xi))*dx, 0, 1)

        for elem_i in range(elem_num):
            x_1 = self.x[elem_i]
            x_2 = self.x[elem_i + 1]
            A_local, b_local = self._get_element_contribution(lhs_local_contribution, \
                                                        rhs_local_contribution, x_1, x_2)
            A[elem_i:elem_i + 2, elem_i:elem_i + 2] += A_local
            b[elem_i:elem_i + 2] += b_local

        # Drop first/last row and first column/last since they come from BCs
        # Since Dirichlet BCs are known at boundaries, drop the first and last row 
        # and eliminate the first and last columns by applings BCs
        b[1] -= A[1, 0] * self.u_0
        b[-2] -= A[-2, -1] * self.u_N
        return LinearAlgebraicSystem(A[1:-1, 1:-1], b[1:-1])

    def _get_element_contribution(self, lhs_int, rhs_int, x_1, x_2):
        nodes_num = 2
        A = np.zeros((nodes_num, nodes_num))
        b = np.zeros((nodes_num,))
        for i in range(nodes_num):
            for j in range(nodes_num):
                A[i, j] = lhs_int(i, j, x_1, x_2)
                b[i] = rhs_int(i, x_1, x_2)

        return A, b
