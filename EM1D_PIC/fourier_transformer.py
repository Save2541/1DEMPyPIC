import numpy
import scipy.fft

from . import qol
from . import digital_filtering


def get_data_array(key, data):
    """
    Get array from data
    :param key: name of the requested array
    :param data: loaded data file
    :return: requested array
    """
    if key == "rho":
        return data["rho_list"]
    else:
        return data[key]


def fourier_transform_2d(key, data, k_array, is_smooth=True):
    """
    Perform a 2D Fourier Transform on a data array
    :param key: name of the array to be fourier transformed
    :param data: loaded data file
    :param k_array: 2D array of k values
    :param is_smooth: flag to perform digital filtering on the fourier transformed array
    :return: fourier transformed array (top half only)
    """
    # GET VARIABLES
    (dx, dt, nt) = qol.read_almanac(data, "dx", "dt", "nt")
    # GET DATA ARRAY TO BE FOURIER TRANSFORMED
    array = get_data_array(key, data)
    fourier_plot = numpy.abs(dx * dt * scipy.fft.rfft2(array))
    # Y-AXIS LENGTH
    y_max = nt // 2 + 1
    if is_smooth:
        smoothed_plot = digital_filtering.smooth(fourier_plot, k_array, dx)
        return smoothed_plot[0:y_max]
    else:
        return fourier_plot[0:y_max]
