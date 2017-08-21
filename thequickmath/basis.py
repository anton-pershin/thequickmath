from sympy.abc import x, xi, n
from sympy import sympify
from sympy.utilities.lambdify import lambdify
from nputils import *
class BasisFunction:
    '''
    Generic basis function which produces the basis using operator [].
    For example:
    basis = BasisFunction()
    '''
    def __init__(self, gen_expr):
        '''
        gen_expr is expected to be a sympified function of (x, n) when n corresponds to
        a successive number of basis function.
        '''
        self.gen_expr = gen_expr
        pass

    @classmethod
    def fromstring(cls, gen_expr):
        return cls(sympify(gen_expr))

    def __getitem__(self, n_val):
        '''
        Returns a function of basis corresponding to n.
        '''
        return self.gen_expr.subs(n, n_val)

    def eval(self, c):
        '''
        Takes vector of coefficients for linear combination
        and returns the resultant function spanned by the given basis
        '''
        u = c[0] * self[0]
        for n in range(1, c.shape[0]):
            u += c[n] * self[n]
        return u

class UniversalBasisFunction:
    def __init__(self, funcs):
        '''
        funcs is a list of sympy-functions of x corresponding to a node.
        '''
        self.funcs = funcs
        pass

    @classmethod
    def linear_lagrangian(cls):
        return cls([sympify('1 - xi'), sympify('xi')])

    def __getitem__(self, n_val):
        '''
        Returns a function of basis corresponding to a local node n.
        '''
        return self.funcs[n_val]

    def eval(self, u, x_mesh):
        '''
        Takes vector of nodal values (u) and returns lambdified approximate solution
        '''
        def sol(x):
            x_2_i = find_right_index(x_mesh, x)
            if x_2_i == 0:
                return u[0]
            if x_2_i >= x_mesh.shape[0]:
                return u[-1]
            x_1 = x_mesh[x_2_i - 1]
            x_2 = x_mesh[x_2_i]
            xi_x = lambda x: (x - x_1) / (x_2 - x_1)
            N_0 = lambdify(xi, self[0], 'numpy')
            N_1 = lambdify(xi, self[1], 'numpy')
            return u[x_2_i - 1]*N_0(xi_x(x)) + u[x_2_i]*N_1(xi_x(x))

        return sol