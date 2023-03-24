import numpy
import scipy.fft

from . import digital_filtering


def solve_field_x(grids, dx, ksqi_over_epsilon, sample_k):
    """
    Find Ex using Poisson's Equation.
    :param grids: list of grid cells
    :param dx: grid size
    :param ksqi_over_epsilon: list of values of k-squared-inverse divided by epsilon naught
    :param sample_k: list of sample frequencies k to be sent to the smoothing function
    :return: none
    """

    # FOURIER TRANSFORM, FROM rho(x) TO rho(k)
    rho_n_list = dx * scipy.fft.rfft(grids.rho)
    rho_n_list = digital_filtering.smooth(rho_n_list, sample_k, dx)
    # POISSON'S EQUATION, FROM rho(k) to phi(k)
    phi_n_list = numpy.concatenate(
        ([0], rho_n_list[1:] * ksqi_over_epsilon))
    # INVERSE FOURIER TRANSFORM, phi(k) TO phi(x)
    phi_list = 1 / dx * scipy.fft.irfft(phi_n_list)
    # SOLVE FOR Ex FROM phi(x)
    grids.ex = (numpy.roll(phi_list, 1) - numpy.roll(phi_list, -1)) / (2 * dx)


def solve_field(grids, dt, epsilon, sqrt_mu_over_epsilon):
    """
    Find Ey, Bz, Ez, and By using Maxwell's Equation
    :param grids: list of grid cells
    :param dt: duration of a time step
    :param epsilon: permittivity of free space (epsilon_0)
    :param sqrt_mu_over_epsilon: square root of mu naught over epsilon naught
    :return: none
    """
    dt_over_4 = 0.25 * dt

    # SOLVE RIGHT-GOING FIELD QUANTITY F = 1/2*(Ey + Bz)
    grids.f_right = numpy.roll(grids.f_right_old - dt_over_4 * grids.jy_old, 1) - dt_over_4 * grids.jy

    # SOLVE LEFT-GOING FIELD QUANTITY F = 1/2*(Ey - Bz)
    grids.f_left = numpy.roll(grids.f_left_old - dt_over_4 * grids.jy_old, -1) - dt_over_4 * grids.jy

    # SOLVE Ey
    grids.ey = (grids.f_right + grids.f_left) / epsilon

    # SOLVE Bz
    grids.bz = (grids.f_right - grids.f_left) * sqrt_mu_over_epsilon

    # SOLVE RIGHT-GOING FIELD QUANTITY G = 1/2*(Ez - By)
    grids.g_right = numpy.roll(grids.g_right_old - dt_over_4 * grids.jz_old, 1) - dt_over_4 * grids.jz

    # SOLVE LEFT-GOING FIELD QUANTITY G = 1/2*(Ez + By)
    grids.g_left = numpy.roll(grids.g_left_old - dt_over_4 * grids.jz_old, 1) - dt_over_4 * grids.jz

    # SOLVE Ez
    grids.ez = (grids.g_right + grids.g_left) / epsilon

    # SOLVE By
    grids.by = (grids.g_left - grids.g_right) * sqrt_mu_over_epsilon
