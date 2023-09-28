import math

import numpy

from . import qol


def distribute_positions(sp_list, almanac, rng, enforce_local_uniformity, grids=None):
    """
    Distribute particle positions
    :param sp_list: list of specie parameters
    :param almanac: dictionary of useful quantities
    :param rng: random number generator
    :param enforce_local_uniformity: if True, every grid will have the same number of particles
    :param grids: simulation grids
    :return: dictionary of particle positions
    """
    # DEFAULT DENSITY DISTRIBUTIONS (UNIFORM)
    x_list = {}
    if enforce_local_uniformity:
        # INITIALIZE ARRAYS
        for i in range(0, sp_list.n_sp):
            x_list[sp_list.name[i]] = numpy.zeros(sp_list.np[i])
        # GET NUMBER OF PARTICLES PER GRID
        np_per_grid = sp_list.np_per_grid
        # SET COUNTERS
        start, finish = numpy.zeros_like(np_per_grid), numpy.zeros_like(np_per_grid)
        # LOOP THROUGH THE GRIDS, EACH GRID GETTING THE SAME NUMBER OF PARTICLES
        for grid_x in grids.x:
            finish += np_per_grid
            for i in range(0, sp_list.n_sp):
                x_list[sp_list.name[i]][start[i]:finish[i]] = rng.uniform(grid_x, grid_x + almanac["dx"],
                                                                          size=np_per_grid[i])
            start += np_per_grid
        # SHUFFLE LIST (PROBABLY NOT NEEDED)
        for specie in x_list:
            rng.shuffle(x_list[specie])
    else:
        for i in range(0, sp_list.n_sp):
            x_list[sp_list.name[i]] = rng.uniform(0, almanac["length"], size=sp_list.np[i])

    # INITIAL DENSITY WAVES
    def initialize_density_waves():
        xx = numpy.linspace(0, almanac["length"], int(1E6), endpoint=False)  # evenly spaced choices of x
        for specie_key in x_list:
            nw = sp_list.init_d_wv[specie_key]["number of waves"]
            amplitude = sp_list.init_d_wv[specie_key]["amplitude"]
            prob_list = 1 + amplitude * numpy.sin(qol.number_of_waves_to_wave_number(nw, almanac[
                "length"]) * xx)  # probability of particles to be in each grid cell
            prob_list /= numpy.sum(prob_list)  # normalized probability distribution
            number = len(x_list[specie_key])  # number of particles
            x_list[specie_key] = numpy.random.choice(xx, number, p=prob_list)

    # CHECK IF DENSITY WAVES NEED TO BE INITIALIZED
    for sp_key in x_list:
        if sp_list.init_d_wv[sp_key]["number of waves"] != 0 and sp_list.init_d_wv[sp_key]["amplitude"] != 0:
            initialize_density_waves()
            break

    return x_list


def distribute_velocities(x_list, sp_list, almanac, rng):
    """
    Distribute particle velocities
    :param x_list: list of particle positions
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
