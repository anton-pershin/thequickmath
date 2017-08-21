from __future__ import division
from math import *
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import h5py
import numpy as np
from aux import NamedAttributesContainer, DimensionsDoNotMatch, LabeledValue, is_sequence

# TODO: implement method swapcoords for Field. Sometimes it is useful to change the leading dimension.

class Space(NamedAttributesContainer):
    def __init__(self, coords):
        #self.coords = list(coords)
        NamedAttributesContainer.__init__(self, coords, [])

    def __del__(self):
        print('____ delete space ____')

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
    '''
    Base class for field representation
    '''
    def __init__(self, elements, space):
        self.space = space
        NamedAttributesContainer.__init__(self, elements, [])

    def __del__(self):
        print('____ delete field ____')

    def __add__(self, rhs):
        sum_elements = [elem_self + elem_rhs for elem_self, elem_rhs in zip(self.elements, rhs.elements)]
        sum_field = Field(sum_elements, self.space)
        sum_field.grab_namings(self)
        return sum_field

    def __sub__(self, rhs):
        diff_elements = [elem_self - elem_rhs for elem_self, elem_rhs in zip(self.elements, rhs.elements)]
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
        space = []
        for elem in self.elements:
            elems.append(elem[index])
        if isinstance(index, tuple): # not one-dimensional space
            for coord, coord_index in zip(self.space.elements, index):
                space.append(coord[coord_index])
        elif isinstance(index, slice): # one-dimensional space:
            space.append(self.space.elements[0][index])
        else: # one point in one-dimensional space
            space.append(self.space.elements[0][index])
        f = Field(elems, Space(space))
        f.grab_namings(self)
        return f

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
    if is_sequence(fields_):
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
    all_indexes_expect_coord_index = range(coord_index) + range(coord_index + 1, len(averaged_subfield.space.elements))
    averaged_subfield.space = averaged_subfield.space.make_subspace(all_indexes_expect_coord_index)
    return averaged_subfield

def at(field, coord, value):
    #indexes = field.convert_names_to_indexes_if_necessary(elems)
    coord_index = field.space.convert_names_to_indexes_if_necessary([coord])[0]
    all_indexes_expect_coord_index = range(coord_index) + range(coord_index + 1, len(field.space.elements))
    subspace = field.space.make_subspace(all_indexes_expect_coord_index)

    # As grid is not supposed to be even, use brute force to find the closest index at the given coordinate
    # Binary search is possible for sure, but there is no need -- it is quick enough
    value_index = np.searchsorted(field.space.elements[coord_index], (value)) # employs binary search, finds index BEFORE which value should be inserted => need to check the previous index
    if value_index == 0 or value_index == len(field.space.elements[coord_index]):
        print('Value found only at the edge of space. It might be wrong.')
    if value_index != 0:
        if np.abs(value - field.space.elements[coord_index][value_index - 1]) < np.abs(value - field.space.elements[coord_index][value_index]):
            value_index -= 1
    access_list = []
    for i in range(len(field.space.elements)):
        if i != coord_index:
            access_list.append(slice(0, field.space.elements[i].shape[0]))
        else:
            access_list.append(value_index)

    raw_subfields = []
    for raw_field in field.elements:
        raw_subfields.append(raw_field[tuple(access_list)])

    subfield = Field(raw_subfields, subspace)
    subfield.set_elements_names(field.elements_names)
    return subfield

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

def map_to_equispaced_mesh(field, details_capacity_list):
    if field.space.dim() != 2:
        raise DimensionsDoNotMatch('Mapping to equispaced mesh is possible only for 2-dimensional space')

    new_coord_arrays = []
    indexes_mappings = [] # i -> mapping for ith coord
                          # mapping for ith coord: equispaced_array_index -> original_array_nearest_left_index
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
        indexes_mappings.append(map_differently_spaced_arrays(orig_coord_array, equispaced_array))

    # TODO: need to rewrite in matrix form
    new_x = new_coord_arrays[0]
    new_y = new_coord_arrays[1]
    x_indexes_mapping = indexes_mappings[0]
    y_indexes_mapping = indexes_mappings[1]
    new_elements = [np.zeros((new_x.shape[0], new_y.shape[0])) for i in range(len(field.elements))]
    for i in range(len(new_x)):
        x = new_x[i]
        x_l = field.space.elements[0][x_indexes_mapping[i]]
        if i != len(new_x) - 1:
            x_r = field.space.elements[0][x_indexes_mapping[i] + 1]
        for j in range(len(new_y)):
            y = new_y[j]
            y_l = field.space.elements[1][y_indexes_mapping[j]]
            if j != len(new_y) - 1:
                y_r = field.space.elements[1][y_indexes_mapping[j] + 1]

            for u, new_u in zip(field.elements, new_elements):
                if i == len(new_x) - 1 and j == len(new_y) - 1: # "corner" of domain
                    new_u[i, j] = u[x_indexes_mapping[i], y_indexes_mapping[j]]
                    continue
                #print(new_x.shape)
                #print(new_y.shape)
                #print('Shape')
                #print(u.shape)
                #print('x_indexes_mapping')
                #print(x_indexes_mapping)
                #print('y_indexes_mapping')
                #print(y_indexes_mapping)
                #print('i = ' + str(i) + ', j = ' + str(j))

                if i == len(new_x) - 1: # linear interpolation along the y-axis
                    new_u[i, j] = (y_r - y) / (y_r - y_l) * u[x_indexes_mapping[i], y_indexes_mapping[j]] \
                                + (y - y_l) / (y_r - y_l) * u[x_indexes_mapping[i], y_indexes_mapping[j] + 1]
                elif j == len(new_y) - 1: # linear interpolation along the x-axis
                    new_u[i, j] = (x_r - x) / (x_r - x_l) * u[x_indexes_mapping[i], y_indexes_mapping[j]] \
                                + (x - x_l) / (x_r - x_l) * u[x_indexes_mapping[i] + 1, y_indexes_mapping[j]]
                else: # bilinear interpolation
                    new_u[i, j] = (x_r - x) * (y_r - y) / (x_r - x_l) / (y_r - y_l) * u[x_indexes_mapping[i], y_indexes_mapping[j]] \
                                + (x_r - x) * (y - y_l) / (x_r - x_l) / (y_r - y_l) * u[x_indexes_mapping[i], y_indexes_mapping[j] + 1] \
                                + (x - x_l) * (y_r - y) / (x_r - x_l) / (y_r - y_l) * u[x_indexes_mapping[i] + 1, y_indexes_mapping[j]] \
                                + (x - x_l) * (y - y_l) / (x_r - x_l) / (y_r - y_l) * u[x_indexes_mapping[i] + 1, y_indexes_mapping[j] + 1]

    equispaced_space = Space(new_coord_arrays)
    new_field = Field(new_elements, equispaced_space)
    new_field.grab_namings(field)
    return new_field

def map_differently_spaced_arrays(orig_coord_array, new_coord_array):
    def search_for_next_left_index(array, start_index, value):
        left_index_ = start_index
        for i in range(start_index, len(array)):
            if array[i] > value:
                left_index_ = i - 1
                break
        return left_index_
    
    left_index = 0
    left_indexes_array = [0 for i in range(new_coord_array.shape[0])]
    # boundaries are also mapped exactly
    left_indexes_array[0] = 0
    left_indexes_array[-1] = len(orig_coord_array) - 1
    for i in range(1, len(new_coord_array) - 1):
        left_indexes_array[i] = search_for_next_left_index(orig_coord_array, left_index, new_coord_array[i])
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

def ke(field):
    return np.sum([norms(field, elem_name)**2 for elem_name in field.elements_names])

def max_pointwise_ke(field):
    ke_raw_field = np.zeros(field.elements[0].shape)
    for raw_field in field.elements:
        ke_raw_field += np.power(raw_field, 2)
    return np.amax(ke_raw_field)

def read_field(filename):
    f = h5py.File(filename, 'r')
    u_dataset = f['data']['u']
    u_numpy = u_dataset[0,:,:,:]
    v_numpy = u_dataset[1,:,:,:]
    w_numpy = u_dataset[2,:,:,:]

    x_dataset = f['geom']['x']
    y_dataset = f['geom']['y']
    z_dataset = f['geom']['z']
    x_numpy = x_dataset[:]
    y_numpy = y_dataset[:]
    z_numpy = z_dataset[:]

    # Reverse order for the y-coordinate (chflow gives it from 1 to -1 instead of from -1 to 1)
    space = Space([x_numpy, y_numpy[::-1], z_numpy])
    space.set_xyz_naming()
    field = Field([u_numpy[:,::-1,:], v_numpy[:,::-1,:], w_numpy[:,::-1,:]], space)
    field.set_uvw_naming()
    return field, dict(f.attrs)

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
    f = h5py.File(filename, 'w')
    # Copy attributes
    for key, value in attrs.iteritems():
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
        print('Symmetry {}({}) -> -{}({}): {}'.format(field.elements_names[i], indexes_str, \
                                                    field.elements_names[i], neg_indexes_str, max_symm_error))

def get_reflection_symmetry_estimate(raw_field, space):
    # Check simple symmetry u(x,y,z) -> +-u(-x, -y, -z)
    # x and z are periodic, so the last mesh point is usually not uncluded
    neg_indexes = []
    pos_indexes = []
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
    return np.abs(raw_field[pos_indexes] + raw_field[neg_indexes])

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
    print val