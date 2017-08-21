from sympy import sympify

class SecondOrderLinearOperator:
    '''
    L := p*d2/dx2 + q*d/dx + r
    '''
    def __init__(self, p, q, r):
        '''
        Sympified functions p(x), q(x), r(x) are expected 
        '''
        self.p = p
        self.q = q
        self.r = r

    @classmethod
    def fromstrings(cls, p, q, r):
        return cls(sympify(p), sympify(q), sympify(r))