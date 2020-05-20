from typing import Union
from bisect import bisect_right

import numpy as np


SupportsArithmetic = Union[int, float, np.ndarray]


class NamedAttributesContainer(object):
    def __init__(self, elements, elements_names):
        self.elements = list(elements)
        self.elements_names = list(elements_names)

    def set_elements_names(self, elements_names):
        if len(elements_names) != len(self.elements):
            raise DimensionsDoNotMatch('Number of elements and number of elements names do not match')
        self.elements_names = list(elements_names)
        for i in range(len(self.elements)):
            setattr(self, self.elements_names[i], self.elements[i])

    def make_subcontainer(self, elem_indexes):
        subcontainer_elements = []
        subcontainer_elements_names = []
        for i in range(len(self.elements)):
            if i in elem_indexes:
                subcontainer_elements.append(self.elements[i])
                subcontainer_elements_names.append(self.elements_names[i])

        subcontainer = NamedAttributesContainer(subcontainer_elements, subcontainer_elements_names)
        subcontainer.set_elements_names(subcontainer_elements_names)
        return subcontainer

    def change_order(self, elem_indexes):
        if len(self.elements) != 1: # exception for 1d 
            if len(elem_indexes) != len(self.elements):
                raise DimensionsDoNotMatch('Number of indexes with new order and number of elements do not match')
            self.elements[:] = [self.elements[i] for i in elem_indexes]
            if self.elements_names != []:
                self.elements_names[:] = [self.elements_names[i] for i in elem_indexes]

    def update_attributed_elements(self):
        for i in range(len(self.elements_names)):
            setattr(self, self.elements_names[i], self.elements[i])

    def convert_names_to_indexes_if_necessary(self, names):
        indexes = []
        if isinstance(names[0], str):
            indexes = []
            for name in names:
                indexes.append(self.elements_names.index(name))
        else:
            indexes = names

        return indexes

class DimensionsDoNotMatch(Exception):
    pass

class LabeledValue(object):
    def __init__(self, val, label):
        self.val = val
        self.label = label

    def __str__(self):
        return str(self.val)

#class LabeledList(object):
#    def __init__(self):
#        self.values = []
#        self.label = None
#
#    def append(self, labeled_value):
#        if self.label is None:
#            self.label = labeled_value.label
#        elif self.label != labeled_value.label:
#            raise LabelsDoNotMatch('List label and appended value label do not match')
#
#        self.values.append(labeled_value.val)

class LabeledList(object):
    def __init__(self, values, label):
        self.values = values
        self.label = label


class LabelsDoNotMatch(Exception):
    pass


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))


def np_index(np_array, val):
    """
    Returns index corresponding to the nearest element. Outdated function, should be removed. Use
    index_for_closest_element, index_for_almost_exact_coincidence or index_for_almost_exact_coincidence_unsorted
    """
    return np.abs(np_array - val).argmin()


def index_for_closest_element(seq_sorted, val):
    """
    Returns index corresponding to the closest element to val in the sorted sequence seq_sorted.
    """
    if len(seq_sorted) == 0:
        raise ValueError('Sequence is empty')
    i = bisect_right(seq_sorted, val)  # find next value after "the rightmost value less than or equal to val"
    if i == len(seq_sorted):
        return i - 1
    else:
        return np.argmin([seq_sorted[i - 1], seq_sorted[i]]) + i - 1


def index_for_almost_exact_coincidence(seq_sorted, val, rtol=1e-05, atol=1e-08):
    """
    Returns index corresponding to *almost* exact coincidence of val with an element in the sorted sequence seq_sorted.
    For the explanation of arguments rtol, atol, see numpy.isclose docs.
    """
    if len(seq_sorted) == 0:
        raise ValueError('Sequence is empty')
    i = bisect_right(seq_sorted, val)  # find next value after "the rightmost value less than or equal to val"
    i -= 1
    if not np.isclose(seq_sorted[i], val, rtol=rtol, atol=atol):
        if i + 1 == len(seq_sorted):
            raise ValueError('Bad value is given as input ({}). '
                             'Closest value in a sequence is {}'.format(val, seq_sorted[i]))
        i += 1
        if not np.isclose(seq_sorted[i], val, rtol=rtol, atol=atol):
            raise ValueError('Bad value is given as input ({}). '
                             'Closest values in a sequence are {} and {}'.format(val, seq_sorted[i - 1], seq_sorted[i]))
    return i


def index_for_almost_exact_coincidence_unsorted(seq, val, rtol=1e-05, atol=1e-08):
    """
    Returns index corresponding to *almost* exact coincidence of val with an element in the sequence seq.
    For the explanation of arguments rtol, atol, see numpy.isclose docs.
    """
    if len(seq) == 0:
        raise ValueError('Sequence is empty')
    for i, seq_val in enumerate(seq):
        if np.isclose(seq_val, val, rtol=rtol, atol=atol):
            return i
    raise ValueError('No close value is found in the sequence (given value is {})'.format(val))


def local_maxima_indices(np_array, threshold):
    '''
    Returns a list of indices corresponding to local maxima of the given array subjected to the threshold
    '''
    maxima = []
    for i in range(1, len(np_array) - 1): # local minima in array
        if np_array[i] > np_array[i - 1] and np_array[i] > np_array[i + 1] and np.abs(np_array[i]) > threshold:
            maxima.append(i)
    return maxima

def map_onto_grid(ys, xs, xs_new):
    '''
    Returns new ys such that it is now mapped on grid xs_new using linear interpolation
    '''
    ys_new = np.zeros_like(xs_new)
    left_i = 1
    for i in range(len(xs_new)):
        x = xs_new[i]
        while left_i < len(xs) - 1 and xs[left_i] < x:
            left_i += 1
        left_i -= 1
        x_l = xs[left_i]
        x_r = xs[left_i + 1]
        y_l = ys[left_i]
        y_r = ys[left_i + 1]
        ys_new[i] = 1./(x_r - x_l) * (y_r*(x - x_l) - y_l*(x - x_r))
    return ys_new


def relative_error(approx_val: SupportsArithmetic, exact_val: SupportsArithmetic) -> SupportsArithmetic:
    """
    Returns relative error |approx_val - exact_val| / |exact_val|
    """
    return np.abs(approx_val - exact_val) / np.abs(exact_val)