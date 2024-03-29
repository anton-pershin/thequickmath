import os
import re

import numpy as np
import netCDF4
import h5py

from thequickmath.misc import *


class Space(NamedAttributesContainer):
    def __init__(self, coords):
        #self.coords = list(coords)
        NamedAttributesContainer.__init__(self, coords, [])

#    def __del__(self):
#        print('____ delete space ____')

    def dim(self):
        return len(self.elements)

    def set_xyz_naming(self):
        if self.dim() != 3:
            raise DimensionsDoNotMatch('XYZ naming is possible only for 3-dimensional space')

        self.set_elements_names(['x', 'y', 'z'])

    def make_subspace(self, indexes):
        subsontainer = self.make_subcontainer(indexes)
        subspace = Space(subsontainer.elements)
        subspace.set_elements_names(subsontainer.elements_names)
        return subspace


class Field(NamedAttributesContainer):
    """
    Base class for field representation

    If Field instance has only one component (element), then one can access the attributes/methods of this component as
    if they were the attributes/methods of the field instance (e.g., field.element.method transforms into
    field.method in this case).

    :todo: need to implement method swapcoords for Field. Sometimes it is useful to change the leading dimension.
    """
    def __init__(self, elements, space):
        self.space = space
        NamedAttributesContainer.__init__(self, elements, [])

#    def __del__(self):
#        print('____ delete field ____')

    def __getattr__(self, item):
        if len(self.elements) == 1:  # if there is only one element, we just proxy __getattr__ to this elements
            return getattr(self.elements[0], item)
        raise AttributeError('No attribute {} found'.format(item))

    def __add__(self, rhs):
        if isinstance(rhs, Field): 
            sum_elements = [elem_self + elem_rhs for elem_self, elem_rhs in zip(self.elements, rhs.elements)]
        else:
            sum_elements = [elem + rhs for elem in self.elements]
        sum_field = Field(sum_elements, self.space)
        sum_field.grab_namings(self)
        return sum_field

    def __sub__(self, rhs):
        if isinstance(rhs, Field): 
            diff_elements = [elem_self - elem_rhs for elem_self, elem_rhs in zip(self.elements, rhs.elements)]
        else:
            diff_elements = [elem - rhs for elem in self.elements]
        diff_field = Field(diff_elements, self.space)
        diff_field.grab_namings(self)
        return diff_field

    def __mul__(self, rhs):
        mul_elements = [elem * rhs for elem in self.elements]
        mul_field = Field(mul_elements, self.space)
        mul_field.grab_namings(self)
        return mul_field

    __rmul__ = __mul__

    def __div__(self, rhs):
        div_elements = [np.divide(elem_self + 1, elem_rhs + 1) for elem_self, elem_rhs in zip(self.elements, rhs.elements)]
        div_field = Field(div_elements, self.space)
        div_field.grab_namings(self)
        return div_field

    def __getitem__(self, index):
        elems = []
        space_elems = []
        space_names = []
        if isinstance(index, tuple): # not one-dimensional space
            for coord, coord_name, coord_index in zip(self.space.elements, self.space.elements_names, index):
                if isinstance(coord_index, slice): # add the coordinate into space only if it is a slice
                    space_elems.append(coord[coord_index])
                    space_names.append(coord_name)
        elif isinstance(index, slice): # one-dimensional space:
            space_elems.append(self.space.elements[0][index])
            space_names.append(self.space.elements_names[0][index])
#        else: # one point in one-dimensional space
#            space_elems.append(self.space.elements[0][index])
#            space_names.append(self.space.elements_names[0][index])

        if len(space_elems) == 0 and len(self.elements) == 1: # just return value
            return self.elements[0][index]
            
        for elem in self.elements:
            elems.append(elem[index])

        space = Space(space_elems)
        space.set_elements_names(space_names)
        f = Field(elems, space)
        f.set_elements_names(self.elements_names)
        return f

    def __str__(self):
        space_dependence = '(' + ', '.join(self.space.elements_names) + ')'
        components_descr = '\n\t'.join([elem + space_dependence for elem in self.elements_names])
        space_descr = '\n\t'.join(['{} (dimension: {})'.format(elem_name, len(elem))
                                   for elem, elem_name in zip(self.space.elements, self.space.elements_names)])
        return 'Field instance has {} components:\n\t{}\n' \
               'where each component is defined on the space with {} coordinates:\n\t{}\n' \
               'Total dimension is {}'.format(len(self.elements), components_descr, len(self.space.elements),
                                              space_descr,
                                              len(self.elements)*np.prod([len(elem) for elem in self.space.elements]))

    def grab_namings(self, another_field):
        self.space.set_elements_names(another_field.space.elements_names)
        self.set_elements_names(another_field.elements_names)

    def set_uvw_naming(self):
        if len(self.elements) != 3:
            raise DimensionsDoNotMatch('UVW naming is possible only for a vector field with 3 elements')

        self.set_elements_names(['u', 'v', 'w'])

    def make_subfield(self, elems):
        indexes = self.convert_names_to_indexes_if_necessary(elems)
        subcontainer = self.make_subcontainer(indexes)
        space_ = Space(self.space.elements)
        space_.set_elements_names(self.space.elements_names)
        subfield = Field(subcontainer.elements, space_)
        subfield.set_elements_names(subcontainer.elements_names)
        return subfield

    def change_order(self, elems):
        indexes = self.space.convert_names_to_indexes_if_necessary(elems)
        self.elements[:] = [np.transpose(elem, indexes) for elem in self.elements]
        NamedAttributesContainer.change_order(self, indexes)
        self.space.change_order(indexes)
        self.update_attributed_elements()
        self.space.update_attributed_elements()

    def change_space_order(self, coords):
        indexes = self.space.convert_names_to_indexes_if_necessary(coords)
        self.elements[:] = [np.transpose(elem, indexes) for elem in self.elements]
        self.space.change_order(indexes)
        self.space.update_attributed_elements()


def zeros_like(f: Field) -> Field:
    space_ = Space(f.space.elements)
    space_.set_elements_names(f.space.elements_names)
    elements_made_of_zeros = [np.zeros_like(e) for e in f.elements]
    zero_field = Field(elements_made_of_zeros, space_)
    zero_field.set_elements_names(f.elements_names)
    return zero_field


def L2_norms(field, normalize):
    V = 1
    if normalize:
        for i in range(len(field.space.elements)):
            coord = field.space.elements[i]
            V *= abs(coord[0] - coord[len(coord)-1])

    L2_norms = []
    for i in range(len(field.elements)):
        val = np.sqrt(integrate_field(np.power(field.elements[i], 2), field.space) / V)
        L2_norms.append(val)

    return L2_norms


def L2_norm(raw_array, coord, normalize=True):
    V = 1
    if normalize:
        V *= abs(coord[0] - coord[len(coord)-1])

    val = np.sqrt(integrate_field(np.power(raw_array, 2), Space([coord])) / V)
    #L2_norms.append(LabeledValue(val, '||' + self.elements_names[i] + '||'))

    return val


def norms(fields_, elem, normalize=True):
    fields = []
    if not type(Field) and is_sequence(fields_):
        fields = list(fields_)
    else:
        fields.append(fields_)

    L2_norms = []
    for field in fields:
        elem_index = field.convert_names_to_indexes_if_necessary(elem)[0]
        V = 1
        if normalize:
            for i in range(len(field.space.elements)):
                coord = field.space.elements[i]
                V *= abs(coord[0] - coord[len(coord)-1])
        val = np.sqrt(integrate_field(np.power(field.elements[elem_index], 2), field.space) / V)
        L2_norms.append(val)
        #L2_norms.append(LabeledValue(val, '||' + self.elements_names[0] + '||'))

    if is_sequence(fields_):
        return np.array(L2_norms)
    else:
        return L2_norms[0]


# TODO: must be generalized
#def filter(self, coord, rule):
def filter(field, coord, filtering_capacity):
    index = field.space.convert_names_to_indexes_if_necessary([coord])[0]
    spacing = int(1 / filtering_capacity)
    indexes_to_filter = list(range(1, field.space.elements[index].shape[0], spacing))
    filtered_coord_array = np.delete(field.space.elements[index], indexes_to_filter)
    filtered_coords = []
    for i in range(len(field.space.elements)):
        if i == index:
            filtered_coords.append(np.delete(field.space.elements[index], indexes_to_filter))
        else:
            filtered_coords.append(field.space.elements[i])

    filtered_raw_fields = [np.delete(raw_field, indexes_to_filter, axis=index) for raw_field in field.elements]
    filtered_space = Space(filtered_coords)
    #filtered_space.set_elements_names(field.space.elements_names)
    filtered_field = Field(filtered_raw_fields, filtered_space)
    #filtered_field.set_elements_names(field.elements_names)
    filtered_field.grab_namings(field)
    return filtered_field


def average(field, elems, along):
    indexes = field.convert_names_to_indexes_if_necessary(elems)
    coord_index = field.space.convert_names_to_indexes_if_necessary([along])[0]
    averaged_subfield = field.make_subfield(elems)
    averaged_raw_fields = []
    for raw_field in averaged_subfield.elements:
        averaged_raw_fields.append(np.mean(raw_field, coord_index))
    averaged_subfield.elements = averaged_raw_fields
    all_indexes_expect_coord_index = list(range(coord_index)) + list(range(coord_index + 1, len(averaged_subfield.space.elements)))
    averaged_subfield.space = averaged_subfield.space.make_subspace(all_indexes_expect_coord_index)
    averaged_subfield.update_attributed_elements()
    return averaged_subfield


def at_index(field, coord, index):
    #indexes = field.convert_names_to_indexes_if_necessary(elems)
    coord_index = field.space.convert_names_to_indexes_if_necessary([coord])[0]
    all_indexes_expect_coord_index = list(range(coord_index)) + list(range(coord_index + 1, len(field.space.elements)))
    access_list = []
    for i in range(len(field.space.elements)):
        if i != coord_index:
            access_list.append(slice(0, field.space.elements[i].shape[0]))
        else:
            access_list.append(index)
    return field[tuple(access_list)]


def at(field, coord, value):
    #indexes = field.convert_names_to_indexes_if_necessary(elems)
    coord_index = field.space.convert_names_to_indexes_if_necessary([coord])[0]

    # As grid is not supposed to be even, use brute force to find the closest index at the given coordinate
    # Binary search is possible for sure, but there is no need -- it is quick enough
    value_index = np.searchsorted(field.space.elements[coord_index], (value)) # employs binary search, finds index BEFORE which value should be inserted => need to check the previous index
    if value_index == 0 or value_index == len(field.space.elements[coord_index]):
        print('Value found only at the edge of space. It might be wrong.')
    if value_index != 0:
        if np.abs(value - field.space.elements[coord_index][value_index - 1]) < np.abs(value - field.space.elements[coord_index][value_index]):
            value_index -= 1

    return at_index(field, coord, value_index)
#    access_list = []
#    for i in range(len(field.space.elements)):
#        if i != coord_index:
#            access_list.append(slice(0, field.space.elements[i].shape[0]))
#        else:
#            access_list.append(value_index)
#
#    raw_subfields = []
#    for raw_field in field.elements:
#        raw_subfields.append(raw_field[tuple(access_list)])
#
#    subfield = Field(raw_subfields, subspace)
#    subfield.set_elements_names(field.elements_names)
#    return subfield


def enlarge_field(field, coord, new_maximum, trying_to_extrapolate=False):
    # Two ways of enlarging -- extrapolation and filling by zeros.
    # TODO: trying_to_extrapolate is ignored now. Should be added
    # IMPORTANT: equispaced mesh is assumed along coord direction
    index = field.space.convert_names_to_indexes_if_necessary([coord])[0]
    step = field.space.elements[index][1] - field.space.elements[index][0]
    old_maximum = field.space.elements[index][-1] 
    number_of_points_on_each_edge = int((new_maximum - old_maximum) / step / 2)
    edge_start = field.space.elements[index][-1] + step
    edge_end = edge_start + step * (2 * number_of_points_on_each_edge - 1)
    
    enlarged_space = Space(field.space.elements)
    enlarged_space.elements[index] = np.append(enlarged_space.elements[index], np.linspace(edge_start, edge_end, 2 * number_of_points_on_each_edge))

    edge_padding_shape = list(field.elements[0].shape)
    edge_padding_shape[index] = number_of_points_on_each_edge
    edge_padding = np.zeros(tuple(edge_padding_shape))
    enlarged_raw_fields = [np.concatenate((edge_padding, elem, edge_padding), axis=index) for elem in field.elements]
    #enlarged_raw_fields = [np.lib.pad(elem, ((0,0), (0,0), (number_of_points_on_each_edge, number_of_points_on_each_edge)), 'constant') for elem in field.elements]
    enlarged_field = Field(enlarged_raw_fields, enlarged_space)
    enlarged_field.grab_namings(field)

    return enlarged_field


def enlarge_field_one_side(field, coord, new_maximum, trying_to_extrapolate=False):
    # Two ways of enlarging -- extrapolation and filling by zeros.
    # TODO: trying_to_extrapolate is ignored now. Should be added
    # IMPORTANT: equispaced mesh is assumed along coord direction
    index = field.space.convert_names_to_indexes_if_necessary([coord])[0]
    step = field.space.elements[index][1] - field.space.elements[index][0]
    old_maximum = field.space.elements[index][-1] 
    number_of_points_on_each_edge = int((new_maximum - old_maximum) / step / 2)
    edge_start = field.space.elements[index][-1] + step
    edge_end = edge_start + step * (2 * number_of_points_on_each_edge - 1)
    
    enlarged_space = Space(field.space.elements)
    enlarged_space.elements[index] = np.append(enlarged_space.elements[index], np.linspace(edge_start, edge_end, 2 * number_of_points_on_each_edge))

    edge_padding_shape = list(field.elements[0].shape)
    edge_padding_shape[index] = 2 * number_of_points_on_each_edge
    edge_padding = np.zeros(tuple(edge_padding_shape))
    enlarged_raw_fields = [np.concatenate((elem, edge_padding), axis=index) for elem in field.elements]
    #enlarged_raw_fields = [np.lib.pad(elem, ((0,0), (0,0), (number_of_points_on_each_edge, number_of_points_on_each_edge)), 'constant') for elem in field.elements]
    enlarged_field = Field(enlarged_raw_fields, enlarged_space)
    enlarged_field.grab_namings(field)

    return enlarged_field


def map_to_equispaced_mesh(field, details_capacity_list):
    if field.space.dim() != 2:
        raise DimensionsDoNotMatch('Mapping to equispaced mesh is possible only for 2-dimensional space')

    new_coord_arrays = []
    for orig_coord_array, details_capacity in zip(field.space.elements, details_capacity_list):
        deltas = np.abs(orig_coord_array - np.roll(orig_coord_array, 1))
        max_delta = np.max(deltas[1:])
        min_delta = np.min(deltas[1:])
        equispaced_delta = min_delta + (1 - details_capacity) * (max_delta - min_delta)
        min_value = np.min(orig_coord_array)
        max_value = np.max(orig_coord_array)
        equispaced_number = (np.max(orig_coord_array) - np.min(orig_coord_array)) / equispaced_delta + 1
        equispaced_array = np.linspace(min_value, max_value, equispaced_number)
        new_coord_arrays.append(equispaced_array)

    equispaced_space = Space(new_coord_arrays)
    new_elements = []
    for u in field.elements:
        new_elements.append(map_to_2d_mesh(u, field.space, equispaced_space))
    new_field = Field(new_elements, equispaced_space)
    new_field.grab_namings(field)
    return new_field


def map_to_1d_mesh(raw_field_1d, old_space, new_space):
    """
    Maps raw_field_1d corresponding to old_space into a space defined by new_space using linear interpolation. 
    If new_space is wider than old_space, constant extrapolation is used.
    Returns a new 1D field 
    """
    # Indices mappings: new_space_array_index -> original_array_nearest_left_index
    x_i_map = build_left_index_map_between_arrays(old_space.elements[0], new_space.elements[0])

    new_x = new_space.elements[0]
    new_field = np.zeros((new_x.shape[0],))
    for i in range(len(new_x)):
        x = new_x[i]
        if x < old_space.elements[0][0]:
            new_field[i] = raw_field_1d[0]
        elif x > old_space.elements[0][-1]:
            new_field[i] = raw_field_1d[-1]
        else:
            x_l = old_space.elements[0][x_i_map[i]]
            x_r = old_space.elements[0][x_i_map[i] + 1]
            u_l = raw_field_1d[x_i_map[i]]
            u_r = raw_field_1d[x_i_map[i] + 1]
            new_field[i] = (x_r - x) / (x_r - x_l) * u_l \
                         + (x - x_l) / (x_r - x_l) * u_r
    return new_field


def map_to_2d_mesh(raw_field_2d, old_space, new_space):
    """
    Maps raw_field_2d corresponding to old_space into a space defined by new_space. Returns a new 2D field
    """
    # Indices mappings: new_space_array_index -> original_array_nearest_left_index
    x_i_map = build_left_index_map_between_arrays(old_space.elements[0], new_space.elements[0])
    y_i_map = build_left_index_map_between_arrays(old_space.elements[1], new_space.elements[1])

    new_x = new_space.elements[0]
    new_y = new_space.elements[1]
    new_field = np.zeros((new_x.shape[0], new_y.shape[0]))
    for i in range(len(new_x)):
        x = new_x[i]
        x_l = old_space.elements[0][x_i_map[i]]
        x_r = old_space.elements[0][x_i_map[i] + 1]
        for j in range(len(new_y)):
            y = new_y[j]
            y_l = old_space.elements[1][y_i_map[j]]
            y_r = old_space.elements[1][y_i_map[j] + 1]
#            if j != len(new_y) - 1:
#                y_r = old_space.elements[1][y_i_map[j] + 1]
#            if i == len(new_x) - 1 and j == len(new_y) - 1: # "corner" of domain
#                new_field[i, j] = raw_field_2d[x_i_map[i], y_i_map[j]]
#                continue
            u_ll = raw_field_2d[x_i_map[i], y_i_map[j]]
            u_rl = raw_field_2d[x_i_map[i] + 1, y_i_map[j]]
            u_lr = raw_field_2d[x_i_map[i], y_i_map[j] + 1]
            u_rr = raw_field_2d[x_i_map[i] + 1, y_i_map[j] + 1]
            # bilinear interpolation:
            new_field[i, j] = (x_r - x) * (y_r - y) / (x_r - x_l) / (y_r - y_l) * u_ll \
                            + (x_r - x) * (y - y_l) / (x_r - x_l) / (y_r - y_l) * u_lr \
                            + (x - x_l) * (y_r - y) / (x_r - x_l) / (y_r - y_l) * u_rl \
                            + (x - x_l) * (y - y_l) / (x_r - x_l) / (y_r - y_l) * u_rr

#            if i == len(new_x) - 1: # linear interpolation along the y-axis
#                new_field[i, j] = (y_r - y) / (y_r - y_l) * u_ll \
#                                + (y - y_l) / (y_r - y_l) * u_lr
#            elif j == len(new_y) - 1: # linear interpolation along the x-axis
#                new_field[i, j] = (x_r - x) / (x_r - x_l) * u_ll \
#                                + (x - x_l) / (x_r - x_l) * u_rl
#            else: # bilinear interpolation
#                new_field[i, j] = (x_r - x) * (y_r - y) / (x_r - x_l) / (y_r - y_l) * u_ll \
#                                + (x_r - x) * (y - y_l) / (x_r - x_l) / (y_r - y_l) * u_lr \
#                                + (x - x_l) * (y_r - y) / (x_r - x_l) / (y_r - y_l) * u_rl \
#                                + (x - x_l) * (y - y_l) / (x_r - x_l) / (y_r - y_l) * u_rr
    return new_field


def build_left_index_map_between_arrays(orig_coord_array, new_coord_array):
    def search_for_next_left_index(array, start_index, value):
        left_index_ = start_index
        for i in range(start_index, len(array)):
            if array[i] > value:
                left_index_ = i - 1
                break
        return left_index_
    
    last_left_index = 0
    left_indexes_array = [0 for i in range(len(new_coord_array))]
    #left_indexes_array[-1] = len(orig_coord_array) - 1
    for i in range(0, len(new_coord_array)):
        left_indexes_array[i] = search_for_next_left_index(orig_coord_array, last_left_index, new_coord_array[i])
        last_left_index = left_indexes_array[i]
    return left_indexes_array


def integrate_field(raw_field, space):
    # Introduce flat high-dimensional array. 
    # To visualize it, imagine a sequence of projections (cube to square, square to line, line to point)
    # which are flattened into one-dimensional array
    dims = [space.elements[i].shape[0] for i in range(len(space.elements))] # [N0, N1, N2, N3, N4]
#    print('dims:')
#    print dims
    flat_array_dims = [1] + [np.prod(dims[:i]) for i in range(1, len(dims))] # [1, N0, N0*N1, N0*N1*N2, N0*N1*N2*N3]
#    print('flat_array_dims:')
#    print flat_array_dims
    flat_array = np.zeros((np.sum(flat_array_dims),))
    flat_array_offsets = np.array([np.sum(flat_array_dims[i:len(flat_array_dims)]) for i in range(1, len(flat_array_dims))] + [0]) # [N0 + N0*N1 + N0*N1*N2 + N0*N1*N2*N3, N0*N1 + N0*N1*N2 + N0*N1*N2*N3, N0*N1*N2 + N0*N1*N2*N3, N0*N1*N2*N3, 0]
#    print('flat_array_offsets:')
#    print flat_array_offsets

    # NEED TO CREATE ARRAY OF INDEXES SHIFTS FOR EVERY DIMENSION
    # For dim_i = 4: [N1*N2*N3, N2*N3, N3, 1]
    # For dim_i = 3: [N1*N2, N2, 1]
    # For dim_i = 2: [N1, 1]
    # For dim_i = 1: [1]
    # For dim_i = 0: []

    indexes_shifts = [] # index is dimension. Element is np.array of shifts
    for i in range(len(dims)):
        if i != 0:
            shifts = np.array([np.prod(dims[j:i]) for j in range(1, i)], np.int32) # np.array([]) gives float64 by default, so need to set explicitly    print('indexes_shifts:')
            indexes_shifts.append(np.concatenate((shifts, [1])))
        else:
            indexes_shifts.append(np.array([], np.int32))

#    print('indexes_shifts:')
#    print indexes_shifts

    last_dim = len(raw_field.shape) - 1

    # This func merely generates nested loops
    # indexes_list must be initilized by zeros
    def recurs_integration(dim_i, indexes_array):
        #sum_array_offset is formed from previous dimensions. 
        # Suppose we have dimensions i = 0,1,2,3,4
        # We are looping through i = 0, 1, 2, 3. For i=4 we integrate along the whole coordinate 
        # for every combination ijkl where i,j,k,l in corresp. dimensions. We should suggest such layout
        # that the last looping will arange integrated values subsequently. Then we have flat_index:
        # flat_index = i * N1 * N2 * N3 + j * N2 * N3 + k * N3 + l
        # Obivously, maximum index here is (N0-1)*N1*N2*N3 + (N1-1)*N2*N3 + (N2-1)*N3 + N3-1 = N0*N1*N2*N3 - 1. Denote it (plus 1) flat_index_offset_0123
        # When the loop over l finishes, before going to k + 1 we integrate along i=3 coordinate.
        # At that point we have ijk indexes and subseqently located points corresponding to i=3 coordinate starting
        # at flat_index = i * N1 * N2 * N3 + j * N2 * N3 + k * N3 and taking N3 points. Integrated value should be stored at
        # flat_index = flat_index_offset_0123 + i * N1 * N2 + j * N2 + k. Maximum index then will be 
        # flat_index_offset_012 = flat_index_offset_0123 + (N0-1)*N1*N2 + (N1-1)*N2 + N2-1 + 1 = 
        # flat_index_offset_0123 + N0*N1*N2 = N0*N1*N2*N3 + N0*N1*N2
        # And so on. The results will be stored in the last element of flat array

        flat_array_projection_offset = flat_array_offsets[dim_i]
        coord_offset = np.dot(indexes_shifts[dim_i], indexes_array)
        dim_N = dims[dim_i]

        if dim_i == last_dim: # last coordinate -- integrate along the last coordinate
            flat_array[flat_array_projection_offset + coord_offset] = np.trapz(raw_field[tuple(indexes_array + [slice(0, dim_N)])], space.elements[dim_i])
        else:
#            print('\n')
#            print(dim_i * '\t' + 'Looping dimension ' + str(dim_i))
#            print(dim_i * '\t' + 'indexes_array:')
#            print(dim_i * '\t' + str(indexes_array))
#            print(dim_i * '\t' + 'flat_array_projection_offset = ' + str(flat_array_projection_offset))
#            print(dim_i * '\t' + 'coord_offset = ' + str(coord_offset))

            for n in range(dim_N):
#                print(dim_i * '\t' + 'Go recursively to dimension ' + str(dim_i + 1) + ' with index_array ' + str(indexes_array + [n]) + ', projection offset ' + str(flat_array_offsets[dim_i + 1]) + ' and indexes shifts ' + str(indexes_shifts[dim_i + 1]))
                recurs_integration(dim_i + 1, indexes_array + [n]) # jump to the next dimension and shift offset correspondingly
#                print(dim_i * '\t' + 'Integrate results from the higher dimension')

            higher_dim_proj_offset = flat_array_offsets[dim_i + 1]
            higher_dim_coord_offset = np.dot(indexes_shifts[dim_i + 1][:-1], indexes_array) # essentially, higher_dim_coord_offset is an address of the first element of higher dim projection
            flat_array[flat_array_projection_offset + coord_offset] = \
                np.trapz(flat_array[higher_dim_proj_offset + higher_dim_coord_offset : higher_dim_proj_offset + higher_dim_coord_offset + dim_N], space.elements[dim_i])

    recurs_integration(0, [])
    return flat_array[-1]


def find_likely_period(fields):
    # Take the first field as a reference point. Compare in terms of second norms all subsequent fields to it
    ref_field = fields[0][0]
    ref_norm = np.sqrt(integrate_field(np.power(ref_field.u, 2), ref_field.space))
    smallest_diff_time = 0
    smallest_diff_norm = 0.
    for i in range(1, len(fields)):
        diff_norm = np.sqrt(integrate_field(np.power(ref_field.elements[0] - fields[i][0].elements[0], 2), fields[i][0].space))
        #print('%ith time init: L2_diff = %f' % (i, val))
        #val = np.sum(abs(fields[i][0].elements[0]))
        print('%ith time init: percent of diff = %f' % (i, diff_norm / ref_norm * 100))
        if diff_norm < smallest_diff_norm:
            smallest_diff_norm = diff_norm
            smallest_diff_time = i
    return smallest_diff_time


def find_likely_period(field, coord, ref_value):
    coord_index = field.space.convert_names_to_indexes_if_necessary([coord])[0]
    ref_index = np_index(field.space.elements[coord_index], ref_value)
    ref_subfield = at_index(field, coord_index, ref_index)
    to_one_dim = lambda field_ : np.sqrt(integrate_field(np.power(field_.elements[0], 2), field_.space))
    ref_map = to_one_dim(ref_subfield)
    #smallest_diff_coord = 0
    diff_map = np.array([to_one_dim(at_index(field, coord_index, i) - ref_subfield) \
        for i in range(ref_index, field.space.elements[coord_index].shape[0])])
    min_indices = local_maxima_indices(-diff_map, -1000)
    return [ref_index + i for i in min_indices], diff_map


def ke(field):
    return np.sum([norms(field, elem_name)**2 for elem_name in field.elements_names])


def max_pointwise_ke(field):
    ke_raw_field = np.zeros(field.elements[0].shape)
    for raw_field in field.elements:
        ke_raw_field += np.power(raw_field, 2)
    return np.amax(ke_raw_field)


def read_field(filename):
    _, extension = os.path.splitext(filename)
    if extension == '.h5':
        f = h5py.File(filename, 'r')
        u_dataset = f['data']['u']

        u_numpy = u_dataset[0,:,:,:]
        v_numpy = u_dataset[1,:,:,:]
        w_numpy = u_dataset[2,:,:,:]

        # Reverse order for the y-coordinate (chflow gives it from 1 to -1 instead of from -1 to 1)
        u_numpy = u_numpy[:,::-1,:]
        v_numpy = v_numpy[:,::-1,:]
        w_numpy = w_numpy[:,::-1,:]

        x_dataset = f['geom']['x']
        y_dataset = f['geom']['y']
        z_dataset = f['geom']['z']
        x_numpy = x_dataset[:]
        y_numpy = y_dataset[:]
        z_numpy = z_dataset[:]

        # Reverse order for the y-coordinate (chflow gives it from 1 to -1 instead of from -1 to 1)
        y_numpy = y_numpy[::-1]

        attrs = dict(f.attrs)
    elif extension == '.nc':
        f = netCDF4.Dataset(filename, 'r', format='NETCDF4')
        original_names = {
            'field': {
                'u': 'Velocity_X',
                'v': 'Velocity_Y',
                'w': 'Velocity_Z',
            },
            'space': {
                'x': 'X',
                'y': 'Y',
                'z': 'Z',
            },
        }

        u_numpy = np.array(f[original_names['field']['u']])
        v_numpy = np.array(f[original_names['field']['v']])
        w_numpy = np.array(f[original_names['field']['w']])

        # Reverse order for the y-coordinate (chflow gives it from 1 to -1 instead of from -1 to 1). Also array axes are Z, Y, X, so change them to X, Y, Z
        u_numpy = u_numpy[:,::-1,:]
        v_numpy = v_numpy[:,::-1,:]
        w_numpy = w_numpy[:,::-1,:]
        u_numpy = np.transpose(u_numpy)
        v_numpy = np.transpose(v_numpy)
        w_numpy = np.transpose(w_numpy)

        x_numpy = np.array(f[original_names['space']['x']])
        y_numpy = np.array(f[original_names['space']['y']])
        z_numpy = np.array(f[original_names['space']['z']])

        # Reverse order for the y-coordinate (chflow gives it from 1 to -1 instead of from -1 to 1)
        y_numpy = y_numpy[::-1]

        attrs = dict(f.__dict__)
        attrs['__META__'] = {
            'original_names': original_names,
        }
    else:
        raise ValueError('Bad extension of file "{}" containing field data.'.format(filename))

    space = Space([x_numpy, y_numpy, z_numpy])
    space.set_xyz_naming()
    field = Field([u_numpy, v_numpy, w_numpy], space)
    field.set_uvw_naming()
    return field, attrs


def read_fields(path, file_prefix='u', file_postfix='.h5', start_time = 0, end_time = None, time_step=1):
    files_list = os.listdir(path)
    found_files = []
    if end_time is None:
        end_time = len(files_list) * time_step # impossible to have more time units than number of files

    checker = range(start_time, end_time + time_step, time_step)
    max_time_found = 0
    for file_ in files_list:
        match = re.match(file_prefix + '(?P<time>[0-9]+)' + file_postfix, file_)
        if match is not None:
            time = int(match.group('time'))
            if time >= start_time and time <= end_time:
                if time > max_time_found:
                    max_time_found = time
                checker.remove(time)
                found_files.append(match.string)

    end_time = max_time_found
    if end_time == 0: # nothing is found
        return [], []
    if checker != []:
        if checker.index(end_time + time_step) != 0:
            raise BadFilesOrder('Time order based on files is broken. Probably, some of the files are missed')

    fields = []
    attrs = []
    for t in range(start_time, end_time + time_step, time_step):
        field, attr = read_field(path + '/' + file_prefix + str(t) + file_postfix)
        fields.append(field)
        attrs.append(attr)
    return fields, attrs


def write_field(field, attrs, filename):
    _, extension = os.path.splitext(filename)
    if extension == '.h5':
        f = h5py.File(filename, 'w')
        # Copy attributes
        for key, value in attrs.items():
            f.attrs[key] = value

        data = f.create_group('data')
        geom = f.create_group('geom')
        data['u'] = np.stack(tuple(elem[:,::-1,:] for elem in field.elements), axis=0)
        for i in range(len(field.space.elements_names)):
            # Dirty hack -- the y-coordinate is to be inverted in chflow, so invert it here
            if field.space.elements_names[i] == 'y':
                geom[field.space.elements_names[i]] = field.space.elements[i][::-1]
            else:
                geom[field.space.elements_names[i]] = field.space.elements[i]
        f.close()
    elif extension == '.nc':
        f = netCDF4.Dataset(filename, 'w', format='NETCDF4')
        original_names = attrs['__META__']['original_names']
        f.createDimension(original_names['space']['x'], len(field.space.x))
        f.createDimension(original_names['space']['y'], len(field.space.y))
        f.createDimension(original_names['space']['z'], len(field.space.z))
        vars = {}
        vars['x'] = f.createVariable(original_names['space']['x'], 'f8', (original_names['space']['x'],))
        vars['y'] = f.createVariable(original_names['space']['y'], 'f8', (original_names['space']['y'],))
        vars['z'] = f.createVariable(original_names['space']['z'], 'f8', (original_names['space']['z'],))
        vars['u'] = f.createVariable(original_names['field']['u'], 'f8', (original_names['space']['x'],
                                                                          original_names['space']['y'],
                                                                          original_names['space']['z'],))
        vars['v'] = f.createVariable(original_names['field']['v'], 'f8', (original_names['space']['x'],
                                                                          original_names['space']['y'],
                                                                          original_names['space']['z'],))
        vars['w'] = f.createVariable(original_names['field']['w'], 'f8', (original_names['space']['x'],
                                                                          original_names['space']['y'],
                                                                          original_names['space']['z'],))
        # Save space coordinates
        for coord_name in field.space.elements_names:
            if coord_name == 'y':
                vars[coord_name][:] = getattr(field.space, coord_name)[::-1]
            else:
                vars[coord_name][:] = getattr(field.space, coord_name)
        # Save components of the field
        for comp_name in field.elements_names:
            vars[comp_name][:, :, :] = getattr(field, comp_name)
        # Save attributes
        for k, v in attrs.items():
            if k != '__META__':
                setattr(f, k, v)
        f.close()
    else:
        raise ValueError('Bad extension of file "{}" containing field data.'.format(filename))


def call_by_portions(path, func, start_time=0, end_time=None, portion_size=100):
    '''
    func should take a field

    '''
    results = []
    stop = False
    while not stop:
        print('Processing {} to {}...'.format(start_time, start_time + portion_size - 1))
        fields, attrs = read_fields(path, start_time=start_time, end_time=start_time + portion_size - 1)
        for field_ in fields:
            results.append(func(field_))
        start_time += portion_size
        if end_time is not None:
            if start_time >= end_time:
                stop = True
            elif end_time - start_time < portion_size:
                portion_size = end_time - start_time
        elif len(fields) < portion_size:
            stop = True

    return results


def check_reflection_symmetry_xyz(field):
    # We except that the space is meshed such that there is a symmetry of distribution about the midplane
    for i, raw_field in enumerate(field.elements):
        indexes_str = ','.join([coord_name for coord_name in field.space.elements_names])
        neg_indexes_str = ','.join(['-' + coord_name for coord_name in field.space.elements_names])
        max_symm_error = np.max(get_reflection_symmetry_estimate(raw_field, field.space))
        print('Symmetry {}({}) -> -{}({}): {}'.format(field.elements_names[i], indexes_str,
                                                      field.elements_names[i], neg_indexes_str, max_symm_error))


def get_reflection_symmetry_estimate(raw_field, space):
    # Check simple symmetry u(x,y,z) -> +-u(-x, -y, -z)
    # x and z are periodic, so the last mesh point is usually not uncluded
    neg_indexes = []
    pos_indexes = []
    quarter_space_coords = []
    for raw_coord, coord_name in zip(space.elements, space.elements_names):
        points_num = raw_coord.shape[0]
        neg_start_index = 0
        if coord_name == 'x' or coord_name == 'z':
            points_num += 1
            neg_start_index = 1
        neg_origin_index = points_num // 2 - 1
        pos_origin_index = 0
        if points_num % 2 == 0:
            pos_origin_index = neg_origin_index + 1
        else:
            pos_origin_index = neg_origin_index + 2
        neg_indexes.append(slice(neg_start_index, neg_origin_index + 1))
        pos_indexes.append(slice(-1, pos_origin_index - 1, -1))
        quarter_space_coords.append(raw_coord[neg_indexes[-1]])
    quarter_space = Space(quarter_space_coords)
    symm_err_field = Field([np.abs(raw_field[pos_indexes] + raw_field[neg_indexes])], quarter_space)
    return symm_err_field
    #return np.abs(raw_field[pos_indexes] + raw_field[neg_indexes])


def fourier_representation(field):
#    u_decomp = []
#    y_mid = 0.0
#    f_mid = at(field, 'y', y_mid)
#    for u_i in range(len(f_mid.elements)):
#        u_z = np.zeros((len(f_mid.space.x), len(f_mid.space.z)))
#        for z_i in range(len(f_mid.space.z)):
#            u_f = np.fft.fft(f_mid.elements[u_i][:, z_i])
#            for u_f_i in range(len(u_f)):
#                u_z[u_f_i, z_i] = 2 * np.abs(u_f[u_f_i])
#        u_decomp.append([L2_norm(u_z[i, :], f_mid.space.z) for i in range(u_z.shape[0])])
#    return np.array(u_decomp[0]), np.array(u_decomp[1]), np.array(u_decomp[2])
    u_decomp = []
    u_z = np.zeros((len(field.space.x), len(field.space.z)))
    u_y = np.zeros((len(field.space.x), len(field.space.y)))
    for u_i in range(len(field.elements)):
        for y_i in range(len(field.space.y)):
            for z_i in range(len(field.space.z)):
                u_f = np.fft.fft(field.elements[u_i][:, y_i, z_i])
                for u_f_i in range(len(u_f)):
                    u_z[u_f_i, z_i] = 2 * np.abs(u_f[u_f_i])
            u_y[:, y_i] = np.sqrt(1/u_z.shape[1] * np.sum(u_z**2, axis=1))
        #u_decomp.append([L2_norm(u_z[i, :], f_mid.space.z) for i in range(u_z.shape[0])])
        u_decomp.append(np.sqrt(1/u_y.shape[1] * np.sum(u_y**2, axis=1)))
    return u_decomp[0][:len(u_decomp[0])//2], u_decomp[1][:len(u_decomp[0])//2], u_decomp[2][:len(u_decomp[0])//2]


def fourier_representation_z(field):
    u_decomp = []
    u_z = np.zeros((len(field.space.x), len(field.space.z)))
    u_y = np.zeros((len(field.space.x), len(field.space.y)))
    for u_i in range(len(field.elements)):
        for z_i in range(len(field.space.z)):
            for y_i in range(len(field.space.y)):
                u_f = np.fft.fft(field.elements[u_i][:, y_i, z_i])
                for u_f_i in range(len(u_f)):
                    u_y[u_f_i, y_i] = 2 * np.abs(u_f[u_f_i])
            u_z[:, z_i] = np.sqrt(1/u_y.shape[1] * np.sum(u_y**2, axis=1))
        u_decomp.append(np.copy(u_z[:u_z.shape[0]//2, :]))
    return np.stack(u_decomp)


def fourier_representation_midplane(field):
    u_decomp = []
    u_z = np.zeros((len(field.space.x), len(field.space.z)))
    y_i = len(field.space.y) // 2
    for u_i in range(len(field.elements)):
        for z_i in range(len(field.space.z)):
            u_f = np.fft.fft(field.elements[u_i][:, y_i, z_i])
            u_z[:, z_i] = np.abs(u_f)
        u_decomp.append(np.copy(u_z[:u_z.shape[0]//2, :]))
    return np.stack(u_decomp)


def pointwise_fourier_representation(field, y_i, z_i):
    u_decomp_cos = []
    u_decomp_sin = []
    u_z = np.zeros((len(field.space.x), len(field.space.z)))
    u_y = np.zeros((len(field.space.x), len(field.space.y)))
    for u_i in range(len(field.elements)):
        u_f = np.fft.fft(field.elements[u_i][:, y_i, z_i])
        #u_decomp.append(2 * np.abs(u_f[:len(field.space.z)//2]))
        u_decomp_cos.append(2 * np.real(u_f[:len(field.space.z)//2]))
        u_decomp_sin.append(2 * np.imag(u_f[:len(field.space.z)//2]))
    return np.stack(u_decomp_cos), np.stack(u_decomp_sin)


def complex_fourier_representation_z(field):
    u_decomp = []
    u_z = np.zeros((len(field.space.x), len(field.space.z)), dtype=complex)
    u_y = np.zeros((len(field.space.x), len(field.space.y)), dtype=complex)
    for u_i in range(len(field.elements)):
        for z_i in range(len(field.space.z)):
            for y_i in range(len(field.space.y)):
                u_f = np.fft.fft(field.elements[u_i][:, y_i, z_i])
                for u_f_i in range(len(u_f)):
                    u_y[u_f_i, y_i] = u_f[u_f_i]
            max_wall_normal_magnitude_on_zero_mode_index = np.abs(u_y[0, :]).argmax()
            u_z[:, z_i] = u_y[:, max_wall_normal_magnitude_on_zero_mode_index]
        u_decomp.append(np.copy(u_z[:u_z.shape[0]//2, :]))
    return np.stack(u_decomp)


def scale(field, c_x, c_z):
    '''Scales the field according to a rule [u, v, w](x, y, z) -> [u, v, w](c_x*x, y, c_z*z)
    '''
    scaled_space = Space([c_x*field.space.x, c_z*field.space.z])
    scaled_space.set_elements_names(['x', 'z'])
    return map_field(field, scaled_space)


def shift(field, alpha, gamma):
    '''Shifts the field according to a rule [u, v, w](x, y, z) -> [u, v, w](x + alpha, y, z + beta)
    The x- and z-directions are assumed to be periodic.
    '''
    # We first roll the arrays and after that interpolate them
    def normalize_shift(shift, period):
        shift -= (shift // period)*period
        if shift < 0:
            shift += period
        return shift
    spatial_period = lambda x_: x_[-1] + (x_[1] - x_[0])
    rolling_ii = []
    rolled_xz = []
    shifted_xz = []
    for coord, shift in zip((field.space.x, field.space.z), (alpha, gamma)):
        normalized_shift = normalize_shift(shift, spatial_period(coord))
        coord_i_rolling = build_left_index_map_between_arrays(coord, [normalized_shift,])[0]
        rolling_ii.append(-coord_i_rolling)
        rolled_xz.append(coord + coord_i_rolling*(coord[1] - coord[0]))
        shifted_xz.append(coord + normalized_shift)
    rolled_space = Space([rolled_xz[0], field.space.y, rolled_xz[1]])
    rolled_elements = [np.roll(u, rolling_ii, (0, 2)) for u in field.elements]
    rolled_field = Field(rolled_elements, rolled_space)
    rolled_field.grab_namings(field)
    shifted_space = Space(shifted_xz)
    shifted_space.set_elements_names(['x', 'z'])
    return map_field(rolled_field, shifted_space)


def map_field(field, new_2d_space):
    old_2d_space = Space([field.space.x, field.space.z])
    mapped_elements = [np.zeros_like(u) for u in field.elements]
    for u, mapped_u in zip(field.elements, mapped_elements):
        for i in range(u.shape[1]):
            mapped_u[:,i,:] = map_to_2d_mesh(u[:,i,:], old_2d_space, new_2d_space)
    mapped_field = Field(mapped_elements, Space([new_2d_space.x, field.space.y, new_2d_space.z]))
    mapped_field.grab_namings(field)
    return mapped_field


class BadFilesOrder(Exception):
    pass


if __name__ == '__main__':
    randomly_spaced_array = np.array([0., 0.1, 0.3, 0.32, 0.33, 0.5, 0.6, 0.62, 0.8, 1.])
    equispaced_array = np.linspace(0., 1., 20)
    mapping = map_differently_spaced_arrays(randomly_spaced_array, equispaced_array)
    correct_mapping = [0, 0, 1, 1, 1, 1, 2, 4, 4, 4, 5, 5, 7, 7, 7, 7, 8, 8, 8, 9]
    print('Randomly spaced array:')
    print(randomly_spaced_array)
    print('Equispaced array:')
    print(equispaced_array)
    print('Calculated mapping:')
    print(mapping)
    print('Correct mapping:')
    print(correct_mapping)
    print('Not matched indexes:')
    print([i for i,j in zip(mapping, correct_mapping) if i != j])

    from test_fields import get_simple_1D_field, get_simple_2D_field, get_simple_3D_field, get_wave_field, get_randomly_spaced_wave_field
    from plotting import plot_filled_contours
    import matplotlib.pyplot as plt
    test_field = get_randomly_spaced_wave_field()
    equispaced_field = test_field.map_to_equispaced_mesh((1., 1.))
    plot_filled_contours(equispaced_field)
    #plot_filled_contours(test_field)
    plt.show()

    #wave_field = get_wave_field()
    wave_field = get_simple_3D_field()
    val = integrate_field(wave_field.elements[0], wave_field.space)
    print(val)
