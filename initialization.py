def initialization():
    # INITIAL WEIGHTING (x to rho, hat function)

    for specie in species:  # loop for each specie
        nearest_left_grid, nearest_right_grid = specie.nearest_grids()  # get the indices of the nearest left and
        # right grids of the particles
        x_left_grid = grids.x[nearest_left_grid]  # get the positions of the nearest left grids
        xi = specie.x  # get particle positions
        coeff = specie.q / dx / dx  # calculate a coefficient to be used
        d_rho_right = coeff * (xi - x_left_grid)  # calculate densities to be assigned to the nearest right grids
        d_rho_left = coeff * dx - d_rho_right  # calculate densities to be assigned to the nearest left grids
        # ADD DENSITIES TO CORRESPONDING GRIDS
        numpy.add.at(grids.rho, nearest_left_grid, d_rho_left)
        numpy.add.at(grids.rho, nearest_right_grid, d_rho_right)
    # INITIAL FIELD SOLVER (Find Ex)
    rho_n_list = dx * scipy.fft.rfft(grids.rho)  # fourier transform, rho(x) to rho(k)
    phi_n_list = numpy.concatenate(
        ([0], rho_n_list[1:] * ksqi_list / epsilon))  # calculate phi(k) from rho(k)
    phi_list = 1 / dx * scipy.fft.irfft(phi_n_list)  # inverse fourier transform, phi(k) to phi(x)
    # CALCULATE E(x) FROM phi(x)
    grids.ex = (numpy.roll(phi_list, 1) - numpy.roll(phi_list, -1)) / 2 / dx

    # INITIAL PARTICLE MOVER (MOVE BACK V FROM t = 0 to t = - dt/2, hat function)
    for specie in species:
        coefficient = specie.qm * (-dt / 2)  # calculate a coefficient to be used
        d_theta = - b0 * coefficient  # calculate rotation angle due to magnetic field
        cos_d_theta = numpy.cos(d_theta)  # cosine of d_theta
        sin_d_theta = numpy.sin(d_theta)  # sine of d_theta
        nearest_left_grid_point, nearest_right_grid_point = specie.nearest_grids()  # get the indices of the nearest
        # left and right grids of the particles
        xi = specie.x  # get particle positions
        x_right_grid = grids.x[nearest_left_grid_point] + dx  # get the positions of nearest right grids
        # CALCULATE SELF-CONSISTENT ELECTRIC FIELD
        e_sc = (x_right_grid - xi) / dx * grids.ex[nearest_left_grid_point] + (
                xi - grids.x[nearest_left_grid_point]) / dx * grids.ex[nearest_right_grid_point]
        # CALCULATE THE THREE VELOCITY COMPONENTS
        specie.vxp_old = (cos_d_theta * specie.vxp - sin_d_theta * specie.vy) + coefficient * e_sc * cos_theta
        specie.vy_old = sin_d_theta * specie.vxp + cos_d_theta * specie.vy + coefficient * e_ext
        specie.vb0_old = specie.vb0 + coefficient * e_sc * sin_theta

    # STORE INITIAL DATA FOR PHASE SPACE PLOT (ELECTRONS ONLY)
    x_data[0] = species[0].x[plot_particles_id]
    v_data[0] = species[0].vx[plot_particles_id]

    # FIRST STEP IN TIME
    move_particles("init")
    weigh_to_grid()
    solve_field()