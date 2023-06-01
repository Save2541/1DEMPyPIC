import numpy

from . import user_input
from . import multiprocessor


def init_weigh_to_grid(species, grids, dx, comm, ng=user_input.ng):
    """
    Initial weighting of particles to grid values (x to rho, hat function)
    :param species: particle list
    :param grids: grid list
    :param dx: grid size
    :param comm: mpi comm
    :param ng: number of grids
    :return: none
    """
    index = 0
    for specie in species:  # loop for each specie
        # UPDATE NEAREST GRIDS
        specie.update_nearest_grid(dx)
        # GET NEAREST GRIDS
        nearest_left_grid, nearest_right_grid = specie.nearest_grids(ng)
        # GET POSITIONS
        x_left_grid = grids.x[nearest_left_grid]  # get the positions of the nearest left grids
        xi = specie.x  # get particle positions
        coeff = 1 / dx ** 2  # calculate a coefficient to be used
        d_den_right = coeff * (xi - x_left_grid)  # calculate densities to be assigned to the nearest right grids
        d_den_left = coeff * dx - d_den_right  # calculate densities to be assigned to the nearest left grids
        # ADD DENSITIES TO CORRESPONDING GRIDS
        grids.den[index] = numpy.bincount(nearest_left_grid, weights=d_den_left, minlength=ng) + numpy.bincount(nearest_right_grid, weights=d_den_right, minlength=ng)
        grids.rho += specie.q * grids.den[index]
        index += 1
    multiprocessor.gather_rho(grids, comm)
    multiprocessor.gather_den(grids, comm)


def weigh_to_grid(grids, species, dx, sin_theta, cos_theta, comm, ng=user_input.ng):
    """
    Update grid values based on particle values i.e. (x,v) to (rho,j)
    :param grids: list of grid cells
    :param species: list of particles divided into species
    :param dx: grid size
    :param sin_theta: sine of theta, where theta is the angle between B_0 and the z axis
    :param cos_theta: cosine of theta, where theta is the angle between B_0 and the z axis
    :param comm: mpi comm
    :param ng: number of grid cells
    :return: none
    """
    """UPDATE GRID VALUES BASED ON PARTICLE VALUES"""

    # REINITIALIZE CURRENT DENSITIES
    grids.jy_old = numpy.zeros(ng)
    grids.jy = numpy.zeros(ng)
    grids.jz_old = numpy.zeros(ng)
    grids.jz = numpy.zeros(ng)

    # REINITIALIZE RHO AND DEN
    grids.rho = numpy.zeros(ng)
    grids.den = numpy.zeros_like(grids.den)

    index = 0

    for specie in species:  # loop for each specie
        # GET PREVIOUS NEAREST GRIDS AND GRID POSITIONS
        old_nearest_left_grid, old_nearest_right_grid = specie.nearest_grids_old(ng)
        old_x_left_grid = grids.x[old_nearest_left_grid]
        # GET CURRENT NEAREST GRIDS AND CORRESPONDING GRID POSITIONS
        nearest_left_grid, nearest_right_grid = specie.nearest_grids(ng)
        x_left_grid = grids.x[nearest_left_grid]
        # GET PARTICLE CHARGE
        qc = specie.q
        # GET PARTICLE POSITIONS
        xi = specie.x
        # GET PARTICLE POSITIONS FROM THE PREVIOUS TIME STEP
        xi_old = specie.x_old

        # WEIGHING FUNCTIONS

        def weigh_old(values, destination):
            """
            Weigh a value to grids before particle movement
            :param values: value list to be weighted
            :param destination: grid quantity to be updated
            :return: updated grid quantity
            """
            d_right = values * (xi_old - old_x_left_grid)
            d_left = values * dx - d_right
            weighted = destination + numpy.bincount(old_nearest_left_grid, weights=d_left,
                                                    minlength=ng) + numpy.bincount(old_nearest_right_grid,
                                                                                   weights=d_right, minlength=ng)
            return weighted

        def weigh_current(values, destination):
            """
            Weigh a value to grids after particle movement
            :param values: value list to be weighted
            :param destination: grid quantity to be updated
            :return: updated grid quantity
            """
            d_right = values * (xi - x_left_grid)
            d_left = values * dx - d_right
            weighted = destination + numpy.bincount(nearest_left_grid, weights=d_left, minlength=ng) + numpy.bincount(
                nearest_right_grid, weights=d_right, minlength=ng)
            return weighted

        # WEIGH Jy

        value = qc / dx * specie.vy / dx
        grids.jy_old = weigh_old(value, grids.jy_old)
        grids.jy = weigh_current(value, grids.jy)

        # WEIGH Jz

        value = qc / dx * (specie.vb0 * cos_theta - specie.vxp * sin_theta) / dx
        grids.jz_old = weigh_old(value, grids.jz_old)
        grids.jz = weigh_current(value, grids.jz)

        # WEIGH RHO

        value = 1 / dx / dx
        grids.den[index] = weigh_current(value, grids.den[index])
        grids.rho += qc * grids.den[index]
        index += 1

    # GET ARGUMENTS
    args = (grids, comm)

    # GATHER RHO
    multiprocessor.gather_rho(*args)

    # GATHER NUMBER DENSITY
    multiprocessor.gather_den(*args)

    # GATHER J
    multiprocessor.gather_j(*args)


def weigh_to_grid_es(grids, species, dx, comm, ng=user_input.ng):
    """
    Update rho based on x (for ES code)
    :param grids: list of grid cells
    :param species: list of particles divided into species
    :param dx: grid size
    :param comm: mpi comm
    :param ng: number of grid cells
    :return: none
    """
    # REINITIALIZE RHO
    grids.rho = numpy.zeros(ng)
    grids.den = numpy.zeros_like(grids.den)
    index = 0
    for specie in species:  # loop for each specie
        # GET CURRENT NEAREST GRIDS AND CORRESPONDING GRID POSITIONS
        nearest_left_grid, nearest_right_grid = specie.nearest_grids(ng)
        x_left_grid = grids.x[nearest_left_grid]
        # GET PARTICLE CHARGE
        qc = specie.q
        # GET PARTICLE POSITIONS
        xi = specie.x

        # WEIGHING FUNCTIONS

        def weigh_current(values, destination):
            """
            Weigh a value to grids after particle movement
            :param values: value list to be weighted
            :param destination: grid quantity to be updated
            :return: updated grid quantity
            """
            d_right = values * (xi - x_left_grid)
            d_left = values * dx - d_right
            weighted = destination + numpy.bincount(nearest_left_grid, weights=d_left, minlength=ng) + numpy.bincount(
                nearest_right_grid, weights=d_right, minlength=ng)
            return weighted

        # WEIGH RHO

        value = 1 / dx / dx
        grids.den[index] = weigh_current(value, grids.den[index])
        grids.rho += qc * grids.den[index]
        index += 1

    # GET ARGUMENTS
    args = (grids, comm)

    # GATHER RHO
    multiprocessor.gather_rho(*args)

    # GATHER NUMBER DENSITY
    multiprocessor.gather_den(*args)
