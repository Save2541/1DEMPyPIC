import numpy


def weigh_to_grid(grids, species, ng, dx, sin_theta, cos_theta):
    """
    Update grid values based on particle values i.e. (x,v) to (rho,j)
    :param grids: list of grid cells
    :param species: list of particles divided into species
    :param ng: number of grid cells
    :param dx: grid size
    :param sin_theta: sine of theta, where theta is the angle between B_0 and the z axis
    :param cos_theta: cosine of theta, where theta is the angle between B_0 and the z axis
    :return: none
    """
    """UPDATE GRID VALUES BASED ON PARTICLE VALUES"""

    # STORE OLD FIELD QUANTITIES
    grids.f_right_old = grids.f_right
    grids.f_left_old = grids.f_left
    grids.g_right_old = grids.g_right
    grids.g_left_old = grids.g_left

    # REINITIALIZE CURRENT DENSITIES
    grids.jy_old = numpy.zeros(ng)
    grids.jy = numpy.zeros(ng)
    grids.jz_old = numpy.zeros(ng)
    grids.jz = numpy.zeros(ng)

    # REINITIALIZE RHO
    grids.rho = numpy.zeros(ng)

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

        value = qc / dx / dx
        grids.rho = weigh_current(value, grids.rho)
