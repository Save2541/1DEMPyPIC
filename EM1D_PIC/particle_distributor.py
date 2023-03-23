import math

import numpy

from . import qol


def distribute_positions(sp_list, almanac, rng):
    """
    Distribute particle positions
    :param sp_list: list of specie parameters
    :param almanac: dictionary of useful quantities
    :param rng: random number generator
    :return: dictionary of particle positions
    """
    # DEFAULT DENSITY DISTRIBUTIONS (UNIFORM)
    x_list = {}
    for i in range(0, sp_list.n_sp):
        x_list[sp_list.name[i]] = rng.uniform(0, almanac["length"], size=sp_list.np[i])

    # INITIAL DENSITY WAVES
    xx = numpy.linspace(0, almanac["length"], int(1E6))  # evenly spaced choices of x
    for specie_key in x_list:
        nw = sp_list.init_d_wv[specie_key]["number of waves"]
        amplitude = sp_list.init_d_wv[specie_key]["amplitude"]
        prob_list = 1 + amplitude * numpy.sin(qol.number_of_waves_to_wave_number(nw, almanac[
            "length"]) * xx)  # probability of particles to be in each grid cell
        prob_list /= numpy.sum(prob_list)  # normalized probability distribution
        number = len(x_list[specie_key])  # number of particles
        x_list[specie_key] = numpy.random.choice(xx, number, p=prob_list)
    return x_list


def distribute_velocities(x_list, sp_list, almanac, rng):
    """
    Distribute particle velocities
    :param x_list: particle position list
    :param sp_list: list of specie parameters
    :param almanac: dictionary of useful quantities
    :param rng: random number generator
    :return: dictionary of particle velocities
    """
    # DEFAULT VELOCITY DISTRIBUTIONS (GAUSSIAN)
    v_list = {}
    for i in range(0, sp_list.n_sp):
        u_0 = sp_list.drift_velocity[i]
        uxp_0 = u_0 * almanac["cos_theta"]
        ub0_0 = u_0 * almanac["sin_theta"]
        sigma = sp_list.vth[i] / math.sqrt(2)
        number = sp_list.np[i]
        v_list[sp_list.name[i]] = {
            "vxp": rng.normal(uxp_0, sigma, size=number),
            "vy": rng.normal(0, sigma, size=number),
            "vb0": rng.normal(ub0_0, sigma, size=number)}

    # INITIAL VELOCITY WAVES
    for specie_key in v_list:
        for v_key in v_list[specie_key]:
            nw = sp_list.init_v_wv[specie_key][v_key]["number of waves"]
            amplitude = sp_list.init_v_wv[specie_key][v_key]["amplitude"]
            v_list[specie_key][v_key] = v_list[specie_key][v_key] + amplitude * numpy.sin(
                qol.number_of_waves_to_wave_number(nw, almanac["length"]) * x_list[specie_key])
    return v_list
