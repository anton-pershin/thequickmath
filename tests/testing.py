from __future__ import division
from numerics.galerkin import *
from numerics.galerkin_fem import *
from numerics.timesteppers import *
from basis import *
from mathoperator import *
from error_analysis import *

import matplotlib.pyplot as plt
import numpy as np
from sympy import sympify
from sympy.abc import *
from scipy.special import jn, yn

# TEST GALERKIN

def test_galerkin():
    '''
    Consider equation x d2u/dx2 + x^2 du/dx + x^3 u = x^3
    with BC u(-1) = 0 and u'(1) + 2 u(1) = 1
    '''

    beta_bc = 2
    B_bc = 1

    # Compute exact linear system
    alpha = lambda i, j: (i + 1)*j/(i + j) * (1**(i+j) - (-1)**(i+j)) if j != 0 else 0
    beta = lambda i, j: j/(i + j + 2) * (1**(i+j+2) - (-1)**(i+j+2)) if j != 0 else 0
    gamma = lambda i, j: 1/(i + j + 4) * (1**(i+j+4) - (-1)**(i+j+4))
    chi = lambda i: 1/(i + 4) * (1**(i+4) - (-1)**(i+4))

    dim = 4
    A_exact = np.zeros((dim, dim))
    b_exact = np.zeros((dim,))
    for i in range(dim):
        for j in range(dim):
            A_exact[i, j] = alpha(i, j) - beta(i, j) - gamma(i, j) + beta_bc
        b_exact[i] = B_bc - chi(i)

    # Compute approximated linear system
    g = GalerkinMethod()
    g.set_basis(BasisFunction.fromstring('x**n'))
    g.set_operator(SecondOrderLinearOperator.fromstrings(p='x', q='x**2', r='x**3'))
    g.set_rhs(sympify('x**3'))
    g.set_boundary_conditions(beta=beta_bc, B=B_bc)
    g.set_dimension(dim)
    las_approx = g.build_task()
    print(A_exact)
    print(las_approx.A)
    print('Max error for A is ' + str(np.max(las_approx.A - A_exact)))
    print('Max error for b is ' + str(np.max(las_approx.b - b_exact)))

    # Example 1.3 from Mark
    def galerkin_sol(m):
        g = GalerkinMethod()
        #basis_ = BasisFunction.fromstring('x**n')
        basis_ = BasisFunction.fromstring('(1 + x)**(n + 1)')
        g.set_basis(basis_)
        g.set_operator(SecondOrderLinearOperator.fromstrings(p='1', q='1', r='1'))
        g.set_rhs(sympify('x**2'))
        g.set_boundary_conditions(beta=2, B=1)
        g.set_dimension(m)
        las = g.build_task()
        c = np.linalg.solve(las.A, las.b)
        return lambdify(x, basis_.eval(c), 'numpy')

    u_exact = lambda x: x*(x - 2) + np.exp(-x/2.) * (2.812004351 * np.sin(np.sqrt(3)/2 * x) + 0.4977629886 * np.cos(np.sqrt(3)/2 * x))
    x_mesh = np.linspace(-1, 1, 1000)
    func_func_errors(u_exact, galerkin_sol, range(2, 8), x_mesh)

def test_galerkin_fem():
    x_mesh = np.linspace(0, 5, 5)
    x_pretty_mesh = np.linspace(0, 5, 100)
    
    def galerkin_fem_sol():
        '''
        Consider equation (x + 5) d2u/dx2 + du/dx + u = 2
        with BC u(0) = 1 and u(5) = 5/2
        '''
        g = GalerkinFEMMethod()
        #basis_ = BasisFunction.fromstring('x**n')
        basis_ = UniversalBasisFunction.linear_lagrangian()
        g.set_universal_local_basis(basis_)
        g.set_operator(SecondOrderLinearOperator.fromstrings(p='x + 5', q='1', r='1'))
        g.set_rhs(sympify('2'))
        g.set_boundary_conditions(u_0=1, u_N=5/2)
        g.set_mesh(x_mesh)
        las = g.build_task()
        print(las.A)
        print(las.b)
        u = np.linalg.solve(las.A, las.b)
        u_full = np.zeros_like(x_mesh)
        u_full[0] = 1
        u_full[1:-1] = u
        u_full[-1] = 5/2
#        print(u_full)
        return u_full, basis_.eval(u_full, x_mesh)

    #u_exact = lambda x: x*(x - 2) + np.exp(-x/2.) * (2.812004351 * np.sin(np.sqrt(3)/2 * x) + 0.4977629886 * np.cos(np.sqrt(3)/2 * x))
    u_vec, u_approx = galerkin_fem_sol()
    x_ = np.linspace(0, 5, 100)
    u_exact = 2.73305 * jn(0, 2*np.sqrt(x_ + 5)) + 0.57264 * yn(0, 2*np.sqrt(x_ + 5)) + 2
    u_evaled = np.vectorize(u_approx)
    plt.plot(x_pretty_mesh, u_evaled(x_pretty_mesh), '--', linewidth=2)
    plt.plot(x_mesh, u_vec, 's', linewidth=2)
    plt.plot(x_, u_exact, linewidth=2)
    plt.show()

def test_timesteppers():
    t_max = 6
    h = 0.02
    f = lambda t, y: -5*y
    y_exact = lambda t: np.exp(-5*t)
    y_approx = np.zeros((int(t_max//h) + 1,))
    t_max = (y_approx.shape[0] - 1) * h
    y_approx[:4] = np.array([y_exact(0), y_exact(h), y_exact(2*h), y_exact(3*h)])
    n = 3
    t_n = n*h
    while t_n + h <= t_max:
        y_approx[n+1] = am_predictor_corrector(y_approx[n-3:n+1], t_n, f, h)
        n += 1
        t_n = n*h

    simple_error(y_exact, y_approx, np.arange(0, t_max + h, h), log=False)


#test_galerkin_fem()
test_timesteppers()