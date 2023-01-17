import numpy

import user_input


def interpolate(value, nearest_left_grid_point, nearest_right_grid_point, x_right_weight):
    """
    Interpolate grid quantities to particles
    :param value: grid quantity
    :param nearest_left_grid_point: nearest left grid points of the particles
    :param nearest_right_grid_point: nearest right grid points of the particles
    :param x_right_weight: interpolation weight
    :return: grid quantity interpolated to particles
    """
    value_left = value[nearest_left_grid_point]
    new_value = value_left + x_right_weight * (value[nearest_right_grid_point] - value_left)
    return new_value


def store_old_velocities(specie):
    """
    Store old velocities before calculating new velocities
    :param specie: list of particles
    :return: none
    """
    specie.vy_old = specie.vy
    specie.vxp_old = specie.vxp
    specie.vb0_old = specie.vb0


def get_args_interpolate(specie, grids, dx, ng=user_input.ng):
    """
    Get the required arguments for the interpolate function
    :param specie: list of particles
    :param grids: list of grid cells
    :param dx: grid size
    :param ng: number of grid cells
    :return: arguments for the interpolate function (the nearest left grids, nearest right grids, weight)
    """
    # GET NEAREST GRIDS
    nearest_left_grid_point, nearest_right_grid_point = specie.nearest_grids(ng)
    # GET POSITIONS
    xi = specie.x  # get particle positions
    x_left_grid = grids.x[nearest_left_grid_point]  # get nearest left grid positions
    # WEIGHT FOR INTERPOLATION
    x_right_weight = (xi - x_left_grid) / dx
    # RETURN ARGUMENTS TO BE SENT TO THE INTERPOLATION FUNCTION
    return nearest_left_grid_point, nearest_right_grid_point, x_right_weight


def solve_equations_of_motion(specie, dx, dt, length, ex, ey, ez, bx0, bz, sin_theta, cos_theta):
    """
    Solve equations of motion and move particles. Also store old positions and update particles' nearest grids.
    :param specie: list of particles
    :param dx: grid size
    :param dt: time step
    :param length: length of system (x)
    :param ex: electric field x
    :param ey: electric field y
    :param ez: electric field z
    :param bx0: magnetic field x, always constant
    :param bz: magnetic field z
    :param sin_theta: sine of theta, where theta is the angle between B_0 and the z axis
    :param cos_theta: cosine of theta, where theta is the angle between B_0 and the z axis
    :return: none
    """
    # STORE PREVIOUS POSITIONS
    specie.x_old = specie.x
    # GET THE CHARGE PER MASS RATIO
    q_by_m = specie.qm
    # PROJECT MAGNETIC FIELD TO THE b0 DIRECTION (b0 direction = (sin(theta), cos(theta)))
    bb0 = bz * cos_theta + bx0 * sin_theta
    # CALCULATE ROTATION ANGLE
    d_theta = (- q_by_m * dt) * bb0
    sin_d_theta = numpy.sin(d_theta)
    cos_d_theta = numpy.cos(d_theta)
    # CALCULATE HALF ACCELERATION IN xp AND y DIRECTION
    half_acceleration_xp = (q_by_m * dt / 2) * (ex * cos_theta - ez * sin_theta)
    half_acceleration_y = (q_by_m * dt / 2) * ey
    # CALCULATE FULL ACCELERATION IN b0 DIRECTION
    full_acceleration_b0 = (q_by_m * dt) * (ex * sin_theta + ez * cos_theta)
    # ADD HALF ACCELERATIONS TO CORRESPONDING VELOCITIES
    vxp_1 = specie.vxp_old + half_acceleration_xp
    vy_1 = specie.vy_old + half_acceleration_y
    # APPLY ROTATION AND HALF ACCELERATIONS
    specie.vxp = (cos_d_theta * vxp_1 - sin_d_theta * vy_1) + half_acceleration_xp
    specie.vy = (sin_d_theta * vxp_1 + cos_d_theta * vy_1) + half_acceleration_y
    # APPLY FULL ACCELERATION FOR vb0
    specie.vb0 = specie.vb0_old + full_acceleration_b0
    # MOVE X
    specie.x = numpy.fmod(specie.vx(sin_theta, cos_theta) * dt + specie.x_old, length)
    # MAKE SURE THAT 0 < X < LENGTH
    specie.x[specie.x < 0] += length
    # UPDATE NEAREST GRIDS
    specie.update_nearest_grid(dx)


def move_back_v(species, grids, dx, dt, b0, e_ext, sin_theta, cos_theta, ng=user_input.ng):
    """
    Find v at time t = -dt/2.
    :param species: particle list
    :param grids: grid list
    :param dx: grid size
    :param dt: time step
    :param b0: external magnetic field
    :param e_ext: external electric field
    :param sin_theta: sine of b0 angle
    :param cos_theta: cosine of b0 angle
    :param ng: number of grid cells
    :return: none
    """
    for specie in species:
        coefficient = specie.qm * (-dt / 2)  # calculate a coefficient to be used
        d_theta = - b0 * coefficient  # calculate rotation angle due to magnetic field
        cos_d_theta = numpy.cos(d_theta)  # cosine of d_theta
        sin_d_theta = numpy.sin(d_theta)  # sine of d_theta
        # GET NEAREST GRIDS
        nearest_left_grid_point, nearest_right_grid_point = specie.nearest_grids(ng)
        # GET POSITIONS
        xi = specie.x  # get particle positions
        x_right_grid = grids.x[nearest_left_grid_point] + dx  # get the positions of nearest right grids
        # CALCULATE SELF-CONSISTENT ELECTRIC FIELD
        e_sc = (x_right_grid - xi) / dx * grids.ex[nearest_left_grid_point] + (
                xi - grids.x[nearest_left_grid_point]) / dx * grids.ex[nearest_right_grid_point]
        # CALCULATE THE THREE VELOCITY COMPONENTS
        specie.vxp_old = (cos_d_theta * specie.vxp - sin_d_theta * specie.vy) + coefficient * e_sc * cos_theta
        specie.vy_old = sin_d_theta * specie.vxp + cos_d_theta * specie.vy + coefficient * e_ext
        specie.vb0_old = specie.vb0 + coefficient * e_sc * sin_theta


def move_particles_init(species, grids, dx, dt, length, bx0, sin_theta, cos_theta, bz0, e_ext,
                        is_electromagnetic=user_input.is_electromagnetic):
    """
    Particle mover for the first time step
    :param species: list of particles divided into species
    :param grids: list of grid cells
    :param dx: grid size
    :param dt: duration of a time step
    :param length: length of the system
    :param bx0: magnetic field in the x-direction (always constant)
    :param sin_theta: sine of theta, where theta is the angle between B_0 and the z axis
    :param cos_theta: cosine of theta, where theta is the angle between B_0 and the z axis
    :param bz0: magnetic field in the z-direction (constant for ES code)
    :param e_ext: electric field in the y-direction (constant for ES code)
    :param is_electromagnetic: flag for EM code (True or False)
    :return: none
    """
    """Particle mover for the first time step"""
    for specie in species:  # loop for each specie

        # GET ARGUMENTS FOR THE INTERPOLATION FUNCTION
        args_interpolate = get_args_interpolate(specie, grids, dx)

        # INTERPOLATE FIELD QUANTITIES FROM GRIDS TO PARTICLES
        ex = interpolate(grids.ex, *args_interpolate)

        if is_electromagnetic:
            # INTERPOLATE ELECTROMAGNETIC FIELDS
            bz = interpolate(grids.bz, *args_interpolate)
            ey = interpolate(grids.ey, *args_interpolate)
            ez = interpolate(grids.ez, *args_interpolate)
        else:
            # SET VALUES FOR ELECTROSTATIC
            bz = bz0
            ey = e_ext
            ez = 0

        # SOLVE EQUATIONS OF MOTION AND MOVE PARTICLES
        solve_equations_of_motion(specie, dx, dt, length, ex, ey, ez, bx0, bz, sin_theta, cos_theta)


def move_particles_em(species, grids, dx, dt, length, bx0, sin_theta, cos_theta):
    """
    Particle mover for electromagnetic code
    :param species: list of particles divided into species
    :param grids: list of grid cells
    :param dx: grid size
    :param dt: duration of a time step
    :param length: length of the system
    :param bx0: magnetic field in the x-direction (always constant)
    :param sin_theta: sine of theta, where theta is the angle between B_0 and the z axis
    :param cos_theta: cosine of theta, where theta is the angle between B_0 and the z axis
    :return: none
    """
    for specie in species:  # loop for each specie
        # STORE PREVIOUS VELOCITIES
        store_old_velocities(specie)

        # GET ARGUMENTS FOR THE INTERPOLATION FUNCTION
        args_interpolate = get_args_interpolate(specie, grids, dx)

        # INTERPOLATE FIELD QUANTITIES FROM GRIDS TO PARTICLES
        ex = interpolate(grids.ex, *args_interpolate)

        # INTERPOLATE ELECTROMAGNETIC FIELDS
        bz = interpolate(grids.bz, *args_interpolate)
        ey = interpolate(grids.ey, *args_interpolate)
        ez = interpolate(grids.ez, *args_interpolate)

        # SOLVE EQUATIONS OF MOTION AND MOVE PARTICLES
        solve_equations_of_motion(specie, dx, dt, length, ex, ey, ez, bx0, bz, sin_theta, cos_theta)


def move_particles_es(species, grids, dx, dt, length, bx0, sin_theta, cos_theta, bz0, e_ext):
    """
    Particle mover for electrostatic code
    :param species: list of particles divided into species
    :param grids: list of grid cells
    :param dx: grid size
    :param dt: duration of a time step
    :param length: length of the system
    :param bx0: magnetic field in the x-direction (always constant)
    :param sin_theta: sine of theta, where theta is the angle between B_0 and the z axis
    :param cos_theta: cosine of theta, where theta is the angle between B_0 and the z axis
    :param bz0: magnetic field in the z-direction (constant for ES code)
    :param e_ext: electric field in the y-direction (constant for ES code)
    :return: none
    """
    for specie in species:  # loop for each specie
        # STORE PREVIOUS VELOCITIES
        store_old_velocities(specie)

        # GET ARGUMENTS FOR THE INTERPOLATION FUNCTION
        args_interpolate = get_args_interpolate(specie, grids, dx)

        # INTERPOLATE FIELD QUANTITIES FROM GRIDS TO PARTICLES
        ex = interpolate(grids.ex, *args_interpolate)

        # SOLVE EQUATIONS OF MOTION AND MOVE PARTICLES
        solve_equations_of_motion(specie, dx, dt, length, ex, e_ext, 0, bx0, bz0, sin_theta, cos_theta)