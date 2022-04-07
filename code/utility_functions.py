import numpy as np
import math

def line_length(line):
    '''
    Function to return length of line
    @param line: iterable containing two two-ordinate iterables, e.g. 2 x 2 array or 2-tuple of 2-tuples

    @return length: Distance between start & end points in native units
    '''
    return math.sqrt(math.pow(line[1][0] - line[0][0], 2.0) +
                     math.pow(line[1][1] - line[0][1], 2.0))

def coords2distance(coordinate_array):
    '''
    From geophys_utils, transect_utils

    Function to calculate cumulative distance in metres from native (lon/lat) coordinates
    @param coordinate_array: Array of shape (n, 2) or iterable containing coordinate pairs

    @return distance_array: Array of shape (n) containing cumulative distances from first coord
    '''
    coord_count = coordinate_array.shape[0]
    distance_array = np.zeros((coord_count,), coordinate_array.dtype)
    cumulative_distance = 0.0
    distance_array[0] = cumulative_distance
    last_point = coordinate_array[0]

    for coord_index in range(1, coord_count):
        point = coordinate_array[coord_index]
        distance = line_length((point, last_point))
        cumulative_distance += distance
        distance_array[coord_index] = cumulative_distance
        last_point = point

    return distance_array


def thickness_to_depth(thickness):
    """
    Function for calculating depth top from a thickness array
    :param depth: an array of thicknesses
    :return:
    a flat array of depth
    """
    # Create a new thickness array
    depth = np.zeros(shape=thickness.shape,
                               dtype=float)
    # Iterate through the depth array
    depth[1:] = np.cumsum(thickness[:-1])

    return depth

def estimate_uncertainty(interpreted_depths, multiplicative_noise = np.nan, additive_noise = np.nan,
                         use = "layer_thickness", thickness = None):

    if use == "interpreted_depth":
        return interpreted_depths * multiplicative_noise + additive_noise
    elif use == "layer_thickness":
        if thickness is None:
            print("If using layer_thickness, you must define the thicknesses of the AEM layers")
        layer_top_depth = thickness_to_depth(thickness)
        min_inds = np.abs(layer_top_depth - interpreted_depths[:,None]).argmin(axis = 1)
        layer_thickness = thickness[min_inds]
        return layer_thickness * multiplicative_noise + additive_noise
    else:
        print("The 'use' keyword argument must be either 'layer_thickness' or 'interpreted_depth'")
        return None
