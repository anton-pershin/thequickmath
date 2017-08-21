import numpy as np

def find_right_index(nparray, x):
    return np.searchsorted(nparray, x, side='right')