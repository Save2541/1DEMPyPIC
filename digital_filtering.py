import numpy

import constants


def smooth(fourier_plot, sample_k, dx, weight=constants.w_smooth):
    """
    Apply smoothing function to fourier-transformed plot
    :param fourier_plot: fourier-transformed plot to be smoothed
    :param sample_k: a list of sample frequencies k
    :param dx: grid size
    :param weight: weight of smoothing
    :return: smoothed plot
    """

    return (1 + 2 * weight * numpy.cos(sample_k * dx)) / (1 + 2 * weight) * fourier_plot
