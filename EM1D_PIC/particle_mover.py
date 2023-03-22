import numpy

from . import user_input


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
    value_left += x_right_weight * (value[nearest_right_grid_point] - value_left)

    return value_left


def store_old_velocities(specie):
    """
    Store old velocities before calculating new velocities
    :param specie: list of particles
    :return: none
    """
    specie.vy_old = specie.vy
    specie.vxp_old = specie.vxp
    specie.vb0_old = specie.vb0


def get_args_interpolate(specie, grids, dx, ng=None):
    """
    Get the required arguments for the interpolate function
    :param specie: list of particles
    :param grids: list of grid cells
    :param dx: grid size
    :param ng: number of grid cells
    :return: arguments for the interpolate function (the nearest left grids, nearest right grids, weight)
    """
    if ng is None:
        ng = user_input.ng
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
    specie.x_old = specie.x  # SCALARS
    # GET THE CHARGE PER MASS RATIO TIMES dt
    q_by_m_dt = specie.qm * dt  # SCALARS
    half_q_by_m_dt = 0.5 * q_by_m_dt  # SCALARS
    # PROJECT MAGNETIC FIELD TO THE b0 DIRECTION (b0 direction = (sin(theta), cos(theta)))
    bb0 = bz * cos_theta + bx0 * sin_theta  # ARRAYS
    # CALCULATE ROTATION ANGLE
    d_theta = - q_by_m_dt * bb0  # ARRAYS
    sin_d_theta = numpy.sin(d_theta)  # ARRAYS
    cos_d_theta = numpy.cos(d_theta)  # ARRAYS
    # CALCULATE HALF ACCELERATION IN xp AND y DIRECTION
    half_acceleration_xp = half_q_by_m_dt * (ex * cos_theta - ez * sin_theta)  # ARRAYS
    half_acceleration_y = half_q_by_m_dt * ey  # ARRAYS
    # CALCULATE FULL ACCELERATION IN b0 DIRECTION
    full_acceleration_b0 = q_by_m_dt * (ex * sin_theta + ez * cos_theta)  # ARRAYS
    # ADD HALF ACCELERATIONS TO CORRESPONDING VELOCITIES
    vxp_1 = specie.vxp_old + half_acceleration_xp  # ARRAYS
    vy_1 = specie.vy_old + half_acceleration_y  # ARRAYS
    # APPLY ROTATION AND HALF ACCELERATIONS
    specie.vxp = cos_d_theta * vxp_1 - sin_d_theta * vy_1 + half_acceleration_xp
    specie.vy = sin_d_theta * vxp_1 + cos_d_theta * vy_1 + half_acceleration_y
    # APPLY FULL ACCELERATION FOR vb0
    specie.vb0 = specie.vb0_old + full_acceleration_b0
    # MOVE X
    specie.x = numpy.fmod(specie.vx(sin_theta, cos_theta) * dt + specie.x_old, length)
    # MAKE SURE THAT 0 < X < LENGTH
    specie.x[specie.x < 0] += length
    # UPDATE NEAREST GRIDS
    specie.update_nearest_grid(dx)


def _get_specie_vb0(specie, q_by_m_dt, ex, sin_theta, ez, cos_theta):
    return specie.vb0_old + q_by_m_dt * (ex * sin_theta + ez * cos_theta)


def _get_specie_vy(sin_d_theta, vxp_1, cos_d_theta, vy_1, half_acceleration_y):
    return sin_d_theta * vxp_1 + cos_d_theta * vy_1 + half_acceleration_y


def _get_specie_vxp(cos_d_theta, vxp_1, sin_d_theta, vy_1, half_acceleration_xp):
    # APPLY ROTATION AND HALF ACCELERATIONS
    return cos_d_theta * vxp_1 - sin_d_theta * vy_1 + half_acceleration_xp


def _get_vxp_1(specie, half_acceleration_xp):
    return specie.vxp_old + half_acceleration_xp


def _get_vy_1(specie, half_acceleration_y):
    return specie.vy_old + half_acceleration_y


def _get_sin_and_cos_d_theta(bz, cos_theta, bx0, sin_theta, q_by_m_dt):
    # PROJECT MAGNETIC FIELD TO THE b0 DIRECTION (b0 direction = (sin(theta), cos(theta)))
    bb0 = bz * cos_theta + bx0 * sin_theta  # ARRAYS
    # CALCULATE ROTATION ANGLE
    d_theta = - q_by_m_dt * bb0  # ARRAYS
    return numpy.sin(d_theta), numpy.cos(d_theta)


def _get_half_acceleration_xp(half_q_by_m_dt, ex, cos_theta, ez, sin_theta):
    # CALCULATE HALF ACCELERATION IN xp DIRECTION
    return half_q_by_m_dt * (ex * cos_theta - ez * sin_theta)  # ARRAYS


def _get_half_acceleration_y(half_q_by_m_dt, ey):
    # CALCULATE HALF ACCELERATION IN y DIRECTION
    return half_q_by_m_dt * ey


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


def move_particles_em(species, grids, dx, dt, length, bx0, sin_theta, cos_theta, pool=None):
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
    :param pool: multiprocessing pool
    :return: none
    """
    for specie in species:  # loop for each specie
        # STORE PREVIOUS VELOCITIES
        store_old_velocities(specie)

        # GET ARGUMENTS FOR THE INTERPOLATION FUNCTION
        args_interpolate = get_args_interpolate(specie, grids, dx)

        if pool is None:
            # INTERPOLATE FIELD QUANTITIES FROM GRIDS TO PARTICLES
            ex = interpolate(grids.ex, *args_interpolate)

            # INTERPOLATE ELECTROMAGNETIC FIELDS
            bz = interpolate(grids.bz, *args_interpolate)
            ey = interpolate(grids.ey, *args_interpolate)
            ez = interpolate(grids.ez, *args_interpolate)
        else:
            # set up pool workers
            ex = pool.apply_async(interpolate, args=(grids.ex, *args_interpolate))
            bz = pool.apply_async(interpolate, args=(grids.bz, *args_interpolate))
            ey = pool.apply_async(interpolate, args=(grids.ey, *args_interpolate))
            ez = pool.apply_async(interpolate, args=(grids.ez, *args_interpolate))

            # get results from pool workers
            ex = ex.get()
            bz = bz.get()
            ey = ey.get()
            ez = ez.get()

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
