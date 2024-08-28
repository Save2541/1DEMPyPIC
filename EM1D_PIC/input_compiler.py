import math

import numpy

from . import constants
from . import qol
from . import user_input
from . import multiprocessor


def derive_parameters(sp_list, theta=user_input.theta, b0=user_input.b0, dx=user_input.dx, dt=user_input.dt,
                      ng=user_input.ng):
    """
    Derive parameters to put in specie list.
    :param sp_list: list of specie parameters
    :param theta: b0 angle
    :param b0: external magnetic field
    :param dx: grid size
    :param dt: time step
    :param ng: number of grid cells
    :return: a dictionary of useful numbers in mks
    """
    # DERIVED QUANTITIES
    theta = math.radians(theta)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    bz0 = b0 * cos_theta  # magnetic field along z (T)
    bx0 = b0 * sin_theta  # magnetic field along x (T)
    sp_list.qm = sp_list.charge / sp_list.mass  # charge-to-mass ratios (C/kg)
    sp_list.wp = numpy.sqrt(
        sp_list.density * sp_list.qm * sp_list.charge / constants.epsilon)  # plasma frequencies (rad/s)
    sp_list.wc = abs(sp_list.qm) * b0  # cyclotron frequencies (rad/s)
    sp_list.kt = constants.kb * sp_list.temperature  # thermal energies kT's (J)
    sp_list.vth = numpy.sqrt(2 * sp_list.kt / sp_list.mass)  # thermal velocities (m/s)
    sum_z2n_over_t = numpy.sum(sp_list.charge ** 2 * sp_list.density / sp_list.temperature)
    lambda_d = math.sqrt(constants.epsilon * constants.kb / sum_z2n_over_t)  # Debye length (m)
    #lambda_d = sp_list.vth[0] / math.sqrt(2) / sp_list.wp[0]  # Electron Debye length (m)
    rho_mass = numpy.sum(sp_list.density * sp_list.mass)  # mass density
    # GRID SIZES
    dx = dx * lambda_d  # spatial grid size
    length = ng * dx  # length of the system
    dt = dt * dx / constants.c  # duration of time step
    # GET WAVE NUMBERS FOR INITIAL DENSITY WAVES
    for name in sp_list.name:
        d_wv = sp_list.init_d_wv[name]
        nw = d_wv["number of waves"]
        d_wv["wave number (k)"] = qol.number_of_waves_to_wave_number(nw, length)
        for component in sp_list.init_v_wv[name]:
            v_wv = sp_list.init_v_wv[name][component]
            nw = v_wv["number of waves"]
            v_wv["wave number (k)"] = qol.number_of_waves_to_wave_number(nw, length)

    return {"dx": dx, "length": length, "dt": dt, "theta": theta, "sin_theta": sin_theta, "cos_theta": cos_theta,
            "bz0": bz0, "bx0": bx0, "lambda_d": lambda_d, "rho_mass": rho_mass}


def compile_input(sp_list, specie_names, size, rank, ng=user_input.ng):
    """
    Compile input into usable parameters in specie list
    :param sp_list: specie list
    :param specie_names: input dictionary describing plasma composition
    :param size: number of processors
    :param rank: rank of processor
    :param ng: number of grid cells
    :return: a dictionary of useful numbers in mks
    """
    iterator = 0
    for specie_name, parameters in specie_names.items():
        sp_list.name.append(specie_name)
        sp_list.mass[iterator] = parameters["mass"]
        sp_list.real_mass[iterator] = parameters["mass"]
        sp_list.charge[iterator] = parameters["charge"]
        sp_list.real_charge[iterator] = parameters["charge"]
        sp_list.density[iterator] = parameters["number density"]
        sp_list.temperature[iterator] = parameters["temperature"]
        sp_list.drift_velocity[iterator] = parameters["drift velocity"]
        np_per_grid = parameters["number of simulated particles per grid cell"]
        np = np_per_grid * ng
        sp_list.np_all[iterator] = np
        sp_list.np[iterator] = multiprocessor.get_ranked_np(np, size, rank)
        sp_list.np_per_grid[iterator] = multiprocessor.get_ranked_np(np_per_grid, size, rank)
        if parameters.get("output", True):
            sp_list.out_sp.append(iterator)
        sp_list.init_d_wv[specie_name] = parameters["initial density wave"]
        sp_list.init_v_wv[specie_name] = parameters["initial velocity wave"]
        iterator += 1
    return derive_parameters(sp_list)
