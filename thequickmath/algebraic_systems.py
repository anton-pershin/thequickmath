class LinearAlgebraicSystem:
    '''
    Linear matrix equation: A * c = b
    '''
    def __init__(self, A, b):
        self.A = A
        self.b = b